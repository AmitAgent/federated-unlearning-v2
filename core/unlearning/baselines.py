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

NegGrad+ (Kurmanji, Triantafillou, Hayes, Triantafillou, "Towards Unbounded
Machine Unlearning", NeurIPS 2023, eq. 4): the published form of the same
idea, with an explicit weight beta between the two terms:

    loss = beta * CE(retain batch) - (1 - beta) * CE(forget batch)

The paper tunes beta close to 1 (0.95 on its small-scale setting, 0.9999 on
its large-scale one) and states that small beta "result in explosion quickly".
`run_graddiff` above is the same loss with beta = 0.5 (retain_weight 1.0),
which is the regime the paper warns about. Use `run_neggrad_plus` for a
comparison that follows the paper.

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


def run_neggrad_plus(
    global_state: dict,
    model_id: str,
    channels: int,
    num_classes: int,
    dataset,
    target_indices: list[int],
    retained_indices: list[int],
    steps: int = 100,
    lr: float = 0.01,
    batch_size: int = 64,
    beta: float = 0.95,
    weight_decay: float = 5e-4,
    momentum: float = 0.0,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """NegGrad+ from Kurmanji et al. (NeurIPS 2023), eq. 4.

    loss = beta * CE(retain batch) - (1 - beta) * CE(forget batch), plain SGD
    (lr 0.01 and weight decay as in the paper's appendix), no gradient
    clipping and no early stopping, as in the paper. beta in (0, 1]:
    beta = 1 is retain-only fine-tuning, smaller beta forgets harder and is
    less stable.

    What the paper does NOT fix, and we therefore choose: the batch size, and
    the budget (the paper trains 5-10 epochs; here the budget is `steps`, to
    stay comparable with the other methods in compare_modes). The paper tunes
    beta per setting for high forget error without hurting retain error, so
    treat beta as a hyperparameter to sweep, not a constant.
    """
    if not target_indices or not retained_indices:
        raise ValueError("neggrad_plus needs both target and retained indices")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not (0.0 < beta <= 1.0):
        raise ValueError("beta must be in (0, 1]")
    model = _prepare(global_state, model_id, channels, num_classes, device, seed)
    target_it = _cycle(DataLoader(Subset(dataset, target_indices), batch_size=batch_size, shuffle=True))
    retain_it = _cycle(DataLoader(Subset(dataset, retained_indices), batch_size=batch_size, shuffle=True))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()

    model.train()
    for _ in range(steps):
        t_x, t_y = (v.to(device) for v in next(target_it))
        r_x, r_y = (v.to(device) for v in next(retain_it))
        opt.zero_grad()
        loss = beta * ce(model(r_x), r_y) - (1.0 - beta) * ce(model(t_x), t_y)
        loss.backward()
        opt.step()
    return {"final_state_dict": copy.deepcopy(model.state_dict()), "steps_completed": steps}


BASELINES = {
    "graddiff": run_graddiff,
    "neggrad_plus": run_neggrad_plus,
    "retain_finetune": run_retain_finetune,
}
