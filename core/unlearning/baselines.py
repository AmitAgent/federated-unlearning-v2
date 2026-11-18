"""Established unlearning baselines to compare against GA+KD.

Both are standard, deliberately simple, and run from the SAME trained
global model and are scored by the SAME evaluation code as the other
modes in `core.unlearning.compare_modes`.

GradDiff (a.k.a. NegGrad+): every step does gradient ASCENT on a batch of
the target client's data and gradient DESCENT on a batch of the retained
clients' data, in one combined loss:

    loss = -CE(target batch) + retain_weight * CE(retain batch)

It is a single-stage method (no distillation). It is the usual answer to
"why not just keep the retained loss low while forgetting?".

Retain-only fine-tuning: no forgetting objective at all. Keep training on
the retained clients' data and hope the target's influence washes out.
It is the sanity baseline: if it matches your method, the method is not
doing real forgetting.

Neither gives any formal unlearning guarantee. These are our own compact
implementations of the well-known ideas; they are not copied from any
particular library, and their exact settings here are choices, not
published defaults.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from core.unlearning.gradient_ascent import DEFAULT_GRAD_CLIP_NORM


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _prepare(global_state, model_id, channels, num_classes, device, seed):
    from core.federated.model import build_model

    torch.manual_seed(seed)
    model = build_model(model_id, channels=channels, num_classes=num_classes).to(device)
    model.load_state_dict(global_state)
    return model


def run_graddiff(
    global_state: dict,
    model_id: str,
    channels: int,
    num_classes: int,
    dataset,
    target_indices: list[int],
    retained_indices: list[int],
    steps: int = 30,
    lr: float = 0.01,
    batch_size: int = 32,
    retain_weight: float = 1.0,
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    if not target_indices or not retained_indices:
        raise ValueError("graddiff needs both target and retained indices")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    model = _prepare(global_state, model_id, channels, num_classes, device, seed)
    target_it = _cycle(DataLoader(Subset(dataset, target_indices), batch_size=batch_size, shuffle=True))
    retain_it = _cycle(DataLoader(Subset(dataset, retained_indices), batch_size=batch_size, shuffle=True))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    ce = nn.CrossEntropyLoss()

    model.train()
    for _ in range(steps):
        t_x, t_y = (v.to(device) for v in next(target_it))
        r_x, r_y = (v.to(device) for v in next(retain_it))
        opt.zero_grad()
        loss = -ce(model(t_x), t_y) + retain_weight * ce(model(r_x), r_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        opt.step()
    return {"final_state_dict": copy.deepcopy(model.state_dict()), "steps_completed": steps}


def run_retain_finetune(
    global_state: dict,
    model_id: str,
    channels: int,
    num_classes: int,
    dataset,
    target_indices: list[int],  # unused; kept so both baselines share a signature
    retained_indices: list[int],
    steps: int = 200,
    lr: float = 0.01,
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    if not retained_indices:
        raise ValueError("retain_finetune needs retained indices")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    model = _prepare(global_state, model_id, channels, num_classes, device, seed)
    retain_it = _cycle(DataLoader(Subset(dataset, retained_indices), batch_size=batch_size, shuffle=True))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    ce = nn.CrossEntropyLoss()

    model.train()
    for _ in range(steps):
        x, y = (v.to(device) for v in next(retain_it))
        opt.zero_grad()
        ce(model(x), y).backward()
        opt.step()
    return {"final_state_dict": copy.deepcopy(model.state_dict()), "steps_completed": steps}


BASELINES = {"graddiff": run_graddiff, "retain_finetune": run_retain_finetune}
