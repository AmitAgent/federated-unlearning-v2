"""FedEraser: history-based federated client unlearning.

Ported unchanged in structure from the first project
(``src/unlearning/federaser.py``, ``FedEraserEngine.run_federaser``), which
implements Wang et al., "FedEraser: Enabling Efficient Client-Level Data
Removal from Federated Learning Models", IEEE/ACM IWQoS 2021.

How it works
------------
During the ORIGINAL federated training the server keeps, for every round, each
client's update (its trained weights minus the global weights it started the
round from). To forget a client the server then *replays* that history without
the client:

1. Start from the initial global weights W_0.
2. For each round t = 1..T:
   * "Fast-forward" round: apply the mean of the RETAINED clients' stored
     updates for that round.
   * "Milestone" round (every ``milestone_interval`` rounds, and always the
     last): each retained client runs a few real calibration steps from the
     current unlearned weights. The server rescales each tensor of that new
     update to the norm of the stored historical update, then applies the mean.
3. The weights after round T are the unlearned model.

Only the milestone rounds need real client computation, which is where the
speed-up over retraining comes from. The price is storage: the per-round,
per-client updates must have been recorded during training.

Differences from the first project (all because v2's models differ)
--------------------------------------------------------------------
* v2's models are transformers with LayerNorm only. The first project's
  ResNet-specific steps are therefore not carried over: frozen BatchNorm,
  training only ``layer4`` + classifier, and re-initialising the forgotten
  client's classifier prototype. Calibration here updates every floating
  point parameter instead.
* Calibration uses v2's own client optimizer (``make_optimizer``) and, by
  default, the learning rate the original FL run used. The first project did
  the same (its calibration lr equalled its FL lr).
* The aggregate is a plain mean over retained clients, as in the first
  project. ``weighted=True`` uses sample-weighted means (what v2's FedAvg
  does) instead.

Nothing here touches stored experiment files, ``start_unlearning`` or the
other methods. History is recorded by passing ``RoundLogger()`` as
``fedavg_train(..., round_callback=...)``.
"""
from __future__ import annotations

import copy
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

DEFAULT_CALI_STEPS = 4
DEFAULT_CALI_EPOCHS = 1
DEFAULT_MILESTONE_INTERVAL = 2
_EPS = 1e-8


def _is_float(t: torch.Tensor) -> bool:
    return t.is_floating_point()


class RoundLogger:
    """Records what FedEraser needs while ``fedavg_train`` runs normally.

    Pass an instance as ``round_callback``. It only reads; it never changes
    the training it observes. Stored on CPU so recording does not use GPU
    memory.

    Memory: roughly ``rounds x clients x model size`` (floating point
    parameters only). That storage cost is part of FedEraser's trade-off.
    """

    def __init__(self) -> None:
        self.initial_state: dict | None = None
        self.rounds: list[dict] = []

    def __call__(self, round_num, global_before, client_reports, global_after) -> None:
        if self.initial_state is None:
            self.initial_state = {k: v.detach().cpu().clone() for k, v in global_before.items()}
        clients = {}
        for report in client_reports:
            state = report["state_dict"]
            delta = {
                k: (state[k].detach().cpu() - global_before[k].detach().cpu())
                for k in global_before
                if _is_float(global_before[k])
            }
            clients[int(report["client_id"])] = {
                "num_samples": int(report["num_samples"]),
                "delta": delta,
            }
        self.rounds.append({"round": int(round_num), "clients": clients})

    def to_dict(self) -> dict:
        if self.initial_state is None:
            raise ValueError("RoundLogger recorded no rounds")
        return {"initial_state": self.initial_state, "rounds": self.rounds}


def history_size_bytes(history: dict) -> int:
    """Bytes of tensor data the recorded history holds (the storage cost)."""
    total = sum(v.numel() * v.element_size() for v in history["initial_state"].values())
    for r in history["rounds"]:
        for c in r["clients"].values():
            total += sum(v.numel() * v.element_size() for v in c["delta"].values())
    return total


def save_history(history: dict, path) -> None:
    torch.save(history, path)


def load_history(path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def _calibrate(
    state: dict, model_id, channels, num_classes, dataset, indices,
    epochs, steps, lr, batch_size, device,
) -> tuple[dict, int]:
    """A few real local training steps from `state`; returns (delta, steps_run).

    Mirrors the first project's calibration call: ``cali_epochs`` epochs,
    stopped as soon as ``cali_steps`` optimizer steps have run in total.
    """
    from core.federated.model import build_model, make_optimizer

    model = build_model(model_id, channels=channels, num_classes=num_classes).to(device)
    model.load_state_dict(state)
    model.train()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True)
    optimizer = make_optimizer(model_id, model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    done = 0
    for _ in range(max(1, epochs)):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(images), labels).backward()
            optimizer.step()
            done += 1
            if done >= steps:
                break
        if done >= steps:
            break

    local = model.state_dict()
    delta = {k: (local[k] - state[k]).detach().clone() for k in state if _is_float(state[k])}
    return delta, done


def rescale_to_history(cali_delta: dict, hist_delta: dict) -> dict:
    """Keep the calibration direction, restore the historical magnitude.

    Per tensor: ``cali / ||cali|| * ||hist||`` (the first project's rule,
    including its 1e-8 stabiliser).
    """
    out = {}
    for k, c in cali_delta.items():
        h = hist_delta[k].to(c.device)
        out[k] = (c / (c.norm() + _EPS)) * (h.norm() + _EPS)
    return out


def run_federaser(
    history: dict,
    model_id: str,
    channels: int,
    num_classes: int,
    dataset,
    clients: list[dict],
    target_client_id: int,
    lr: float,
    cali_steps: int = DEFAULT_CALI_STEPS,
    cali_epochs: int = DEFAULT_CALI_EPOCHS,
    milestone_interval: int = DEFAULT_MILESTONE_INTERVAL,
    batch_size: int = 32,
    weighted: bool = False,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Unlearn `target_client_id` by replaying `history` without it.

    `history` is ``RoundLogger.to_dict()`` (or ``load_history``). `clients` is
    the partition's client list (each with ``client_id`` and ``indices``).
    The target client's data is never read; only retained clients calibrate.

    Returns the same shape the established baselines return
    (``final_state_dict``, ``steps_completed``) plus FedEraser details.
    """
    rounds = history["rounds"]
    if not rounds:
        raise ValueError("history has no rounds")
    if cali_steps < 1:
        raise ValueError("cali_steps must be >= 1")
    if milestone_interval < 1:
        raise ValueError("milestone_interval must be >= 1")
    by_id = {c["client_id"]: c for c in clients}
    if target_client_id not in by_id:
        raise ValueError(f"client {target_client_id} is not in the partition")
    retained = [cid for cid in sorted(by_id) if cid != target_client_id and by_id[cid]["indices"]]
    if not retained:
        raise ValueError("FedEraser needs at least one retained client with data")

    started = time.time()
    torch.manual_seed(seed)
    T = len(rounds)
    milestones = {t for t in range(T) if (t + 1) % milestone_interval == 0 or t == T - 1}

    state = {k: v.clone().to(device) for k, v in history["initial_state"].items()}
    calibration_steps_total = 0

    for t in range(T):
        recorded = rounds[t]["clients"]
        usable = [cid for cid in retained if cid in recorded]
        if not usable:
            continue  # no retained client contributed this round
        hist = {cid: {k: v.to(device) for k, v in recorded[cid]["delta"].items()} for cid in usable}

        if t in milestones:
            deltas = {}
            for cid in usable:
                cali, steps = _calibrate(
                    state, model_id, channels, num_classes, dataset, by_id[cid]["indices"],
                    cali_epochs, cali_steps, lr, batch_size, device,
                )
                calibration_steps_total += steps
                deltas[cid] = rescale_to_history(cali, hist[cid])
        else:
            deltas = hist

        if weighted:
            n = {cid: recorded[cid]["num_samples"] for cid in usable}
            total = float(sum(n.values()))
            w = {cid: n[cid] / total for cid in usable}
        else:
            w = {cid: 1.0 / len(usable) for cid in usable}

        new_state = copy.copy(state)
        for k in deltas[usable[0]]:
            agg = sum(deltas[cid][k].float() * w[cid] for cid in usable)
            new_state[k] = state[k] + agg.to(state[k].dtype)
        state = new_state

    return {
        "final_state_dict": {k: v.detach().cpu().clone() for k, v in state.items()},
        "steps_completed": calibration_steps_total,
        "rounds_replayed": T,
        "milestone_rounds": sorted(t + 1 for t in milestones),
        "retained_clients": retained,
        "duration_seconds": time.time() - started,
    }
