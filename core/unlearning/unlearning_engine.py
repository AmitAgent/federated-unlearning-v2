"""Orchestrates the unlearning pipeline (Phase 08 + Phase 09).

Exposes the two conceptual ComputeBackend operations from the
Dashboard Spec (Ch.71.8):

    start_unlearning(client_id)
    get_unlearning_status()

`start_unlearning` runs both stages of the pipeline in sequence:

    1. Gradient Ascent (Phase 08)   -- forget the target client
    2. Knowledge Distillation (Phase 09) -- repair collateral damage
       on every other ("retained") client, using the pre-ascent
       global model as the teacher

and persists the resulting status (and both stages' weights) so a
later `get_unlearning_status` call can pick the full result back up
without re-running anything. Evaluation/MIA (Phase 10) is out of
scope here -- this pipeline stops once the final unlearned model is
saved.
"""
from __future__ import annotations

import time

from core.datasets.base import get_dataset
from core.federated.storage import load_fl_history, load_model_checkpoint
from core.partitioning.storage import load_partition
from core.provenance import software_info
from core.unlearning.calibration import CalibrationConfig
from core.unlearning.gradient_ascent import run_gradient_ascent
from core.unlearning.knowledge_distillation import run_knowledge_distillation
from core.unlearning.storage import (
    load_unlearning_status,
    save_final_checkpoint,
    save_ga_checkpoint,
    save_unlearning_status,
)


#: The Phase 10 evaluation uses test indices [0, 200) as MIA non-members.
#: Calibration reference samples start well past that so the yardstick
#: used to *control* unlearning is never the data used to *score* it.
CALIBRATION_REFERENCE_OFFSET = 1000


def _evenly_spaced(pool: list[int], count: int) -> list[int]:
    """Up to `count` indices spread evenly across `pool` (deterministic)."""
    if count >= len(pool):
        return list(pool)
    stride = len(pool) / count
    return [pool[int(i * stride)] for i in range(count)]


def class_matched_reference_indices(
    test_data, target_labels: list[int], num_classes: int, size: int,
    offset: int = CALIBRATION_REFERENCE_OFFSET,
) -> list[int]:
    """Held-out test indices whose label mix matches the target client's.

    The plain calibration yardstick is a generic slice of the test set. On a
    label-skewed client (e.g. 65% one digit) that yardstick is mostly *other*
    digits, so "looks like unseen data" is judged against the wrong
    population. Here the reference reproduces the target's class proportions,
    so the KS test asks: does the model treat this client's kind of sample
    like samples of the same classes it has never trained on?

    Deterministic; skips the first `offset` test samples (MIA non-members).
    """
    from core.unlearning.heterogeneity import label_distribution, labels_of

    dist = label_distribution(target_labels, num_classes)
    test_labels = labels_of(test_data, list(range(len(test_data))))
    pools: dict[int, list[int]] = {k: [] for k in range(num_classes)}
    for i in range(offset, len(test_data)):
        pools[test_labels[i]].append(i)
    raw = [d * size for d in dist]
    counts = [int(r) for r in raw]
    # largest-remainder rounding so counts sum to `size`
    for k in sorted(range(num_classes), key=lambda k: raw[k] - counts[k], reverse=True)[: size - sum(counts)]:
        counts[k] += 1
    chosen: list[int] = []
    for k in range(num_classes):
        if counts[k] and pools[k]:
            chosen.extend(_evenly_spaced(pools[k], counts[k]))
    return sorted(chosen)


def _per_client_accuracy(state: dict, checkpoint: dict, dataset, clients: list[dict], device: str) -> dict:
    """Accuracy of one model checkpoint on EVERY client's local data.

    Used for the Client Influence chart: the same function is applied to
    the original, post-ascent and final models so every bar in a column
    comes from a single, consistent checkpoint. Keys are strings to match
    the JSON round-trip.
    """
    from core.federated.model import build_model
    from core.unlearning.knowledge_distillation import _client_eval_loaders, _evaluate_accuracy

    model = build_model(
        checkpoint["model_id"], channels=checkpoint["channels"], num_classes=checkpoint["num_classes"]
    ).to(device)
    model.load_state_dict(state)
    loaders = _client_eval_loaders(dataset, clients, 256)
    return {str(cid): _evaluate_accuracy(model, loader, device) for cid, loader in loaders.items()}


def start_unlearning(
    experiment_id: str,
    client_id: int,
    steps: int = 30,
    lr: float = 0.01,
    batch_size: int = 32,
    kd_epochs: int = 5,
    kd_lr: float = 0.01,
    kd_temperature: float = 4.0,
    kd_alpha: float = 0.5,
    kd_batch_size: int = 64,
    device: str = "cpu",
    calibrated: bool = False,
    ks_target: float = 0.1,
    max_retain_drop: float = 0.05,
    reference_size: int = 500,
    guard_size: int = 1000,
    adaptive: bool = False,
    class_matched: bool = False,
) -> dict:
    """Run the full Gradient Ascent -> Knowledge Distillation pipeline
    against `client_id`'s local data.

    Requires that this experiment has already been through Phase 03
    (partitioned) and Phase 05 (trained, with a saved checkpoint).
    """
    started_at = time.time()

    history = load_fl_history(experiment_id)
    partition = load_partition(experiment_id)
    checkpoint = load_model_checkpoint(experiment_id)

    clients = partition["clients"]
    if not (0 <= client_id < len(clients)):
        raise ValueError(
            f"No client {client_id} in this experiment's partition "
            f"(num_clients={partition['num_clients']})"
        )
    target_indices = clients[client_id]["indices"]
    retained_clients = [c for c in clients if c["client_id"] != client_id]

    dataset = get_dataset(partition["dataset"]).get_train()

    # Opt-in heterogeneity-adaptive controller: measure how distinctive the
    # target client is, then override the strength settings. It needs the
    # calibrated stopping rule (it sets that rule's threshold), so it turns
    # calibration on. Without `adaptive`, nothing here runs.
    adaptive_plan = None
    if adaptive:
        from core.unlearning.adaptive import plan_adaptive_unlearning

        adaptive_plan = plan_adaptive_unlearning(
            dataset=dataset,
            clients=clients,
            target_client_id=client_id,
            num_classes=partition["num_classes"],
            steps=steps,
            kd_epochs=kd_epochs,
            kd_alpha=kd_alpha,
        )
        chosen = adaptive_plan["chosen"]
        steps = chosen["steps"]
        kd_epochs = chosen["kd_epochs"]
        kd_alpha = chosen["kd_alpha"]
        ks_target = chosen["ks_target"]
        calibrated = True
        # Adaptive mode judges "forgotten" against held-out samples with the
        # target client's own class mix, not a generic test slice.
        class_matched = True
    if class_matched:
        calibrated = True  # class-matched reference only makes sense with calibration

    calibration = None
    if calibrated:
        test_data = get_dataset(partition["dataset"]).get_test()
        reference_end = min(len(test_data), CALIBRATION_REFERENCE_OFFSET + reference_size)
        if reference_end - CALIBRATION_REFERENCE_OFFSET < 50:
            raise ValueError("test split too small to hold a calibration reference set")
        retained_pool = [i for c in retained_clients for i in c["indices"]]
        if class_matched:
            from core.unlearning.heterogeneity import labels_of

            reference_indices = class_matched_reference_indices(
                test_data, labels_of(dataset, target_indices), partition["num_classes"], reference_size
            )
            if len(reference_indices) < 50:
                raise ValueError("class-matched calibration reference set is too small")
        else:
            reference_indices = list(range(CALIBRATION_REFERENCE_OFFSET, reference_end))
        calibration = CalibrationConfig(
            reference_dataset=test_data,
            reference_indices=reference_indices,
            guard_dataset=dataset,
            guard_indices=_evenly_spaced(retained_pool, guard_size),
            ks_target=ks_target,
            max_retain_drop=max_retain_drop,
        )

    # The pre-ascent global model is Knowledge Distillation's teacher
    # (Phase 09's goal): capture it now, before Gradient Ascent
    # overwrites the working weights below.
    pretrained_state = checkpoint["state_dict"]
    # Consistent "Before" snapshot: the original global model on all clients.
    acc_original = _per_client_accuracy(pretrained_state, checkpoint, dataset, clients, device)

    status = {
        "experiment_id": experiment_id,
        "target_client_id": client_id,
        "model": history["model"],
        "dataset": history["dataset"],
        "stage": "gradient_ascent",
        "status": "running",
        # Phase 15 (Spec Ch.44): record exactly how this run was
        # configured and which code ran it, so the result can be
        # reproduced. Written up front so even a failed run keeps it.
        "method": "gradient_ascent+knowledge_distillation",
        "hyperparameters": {
            "gradient_ascent": {"steps": steps, "lr": lr, "batch_size": batch_size},
            "knowledge_distillation": {
                "epochs": kd_epochs,
                "lr": kd_lr,
                "temperature": kd_temperature,
                "alpha": kd_alpha,
                "batch_size": kd_batch_size,
            },
        },
        "seed": history.get("seed"),
        "software": software_info(),
    }
    if calibrated:
        # `method` is left unchanged so existing dashboards keep working;
        # the extension is recorded alongside the hyperparameters.
        status["hyperparameters"]["calibration"] = {
            "ks_target": ks_target,
            "max_retain_drop": max_retain_drop,
            "reference_size": reference_size,
            "guard_size": guard_size,
            "class_matched_reference": class_matched,
        }
    if adaptive_plan is not None:
        status["hyperparameters"]["adaptive"] = adaptive_plan
    save_unlearning_status(experiment_id, status)

    # -- Stage 1: Gradient Ascent (Phase 08) --------------------------
    ga_started_at = time.time()
    try:
        ga_result = run_gradient_ascent(
            global_state=pretrained_state,
            model_id=checkpoint["model_id"],
            channels=checkpoint["channels"],
            num_classes=checkpoint["num_classes"],
            dataset=dataset,
            target_indices=target_indices,
            steps=steps,
            lr=lr,
            batch_size=batch_size,
            device=device,
            calibration=calibration,
        )
    except Exception as exc:
        status = {
            **status,
            "status": "failed",
            "error": str(exc),
            "duration_seconds": time.time() - started_at,
        }
        save_unlearning_status(experiment_id, status)
        raise

    acc_after_ascent = _per_client_accuracy(
        ga_result["final_state_dict"], checkpoint, dataset, clients, device
    )

    save_ga_checkpoint(
        experiment_id,
        state_dict=ga_result["final_state_dict"],
        model_id=checkpoint["model_id"],
        channels=checkpoint["channels"],
        num_classes=checkpoint["num_classes"],
    )

    gradient_ascent_summary = {
        "status": "complete",
        "steps_completed": ga_result["steps_completed"],
        "steps_requested": ga_result["steps_requested"],
        "stopped_early": ga_result["stopped_early"],
        "initial_loss": ga_result["initial_loss"],
        "initial_accuracy": ga_result["initial_accuracy"],
        "final_loss": ga_result["final_loss"],
        "final_accuracy": ga_result["final_accuracy"],
        "step_history": ga_result["step_history"],
        "duration_seconds": time.time() - ga_started_at,
    }
    if "calibration" in ga_result:
        gradient_ascent_summary["calibration"] = ga_result["calibration"]

    status = {
        **status,
        "stage": "knowledge_distillation",
        "gradient_ascent": gradient_ascent_summary,
    }
    save_unlearning_status(experiment_id, status)

    # -- Stage 2: Knowledge Distillation (Phase 09) -------------------
    kd_started_at = time.time()
    try:
        kd_result = run_knowledge_distillation(
            teacher_state=pretrained_state,
            student_state=ga_result["final_state_dict"],
            model_id=checkpoint["model_id"],
            channels=checkpoint["channels"],
            num_classes=checkpoint["num_classes"],
            dataset=dataset,
            retained_clients=retained_clients,
            epochs=kd_epochs,
            lr=kd_lr,
            temperature=kd_temperature,
            alpha=kd_alpha,
            batch_size=kd_batch_size,
            device=device,
        )
    except Exception as exc:
        status = {
            **status,
            "status": "failed",
            "error": str(exc),
            "duration_seconds": time.time() - started_at,
        }
        save_unlearning_status(experiment_id, status)
        raise

    save_final_checkpoint(
        experiment_id,
        state_dict=kd_result["final_state_dict"],
        model_id=checkpoint["model_id"],
        channels=checkpoint["channels"],
        num_classes=checkpoint["num_classes"],
    )

    acc_final = _per_client_accuracy(
        kd_result["final_state_dict"], checkpoint, dataset, clients, device
    )

    knowledge_distillation_summary = {
        "status": "complete",
        "epochs_completed": kd_result["epochs_completed"],
        "epochs_requested": kd_result["epochs_requested"],
        "temperature": kd_result["temperature"],
        "alpha": kd_result["alpha"],
        "initial_retained_accuracy": kd_result["initial_retained_accuracy"],
        "final_retained_accuracy": kd_result["final_retained_accuracy"],
        "per_client_accuracy_before": kd_result["per_client_accuracy_before"],
        "per_client_accuracy_after": kd_result["per_client_accuracy_after"],
        "epoch_history": kd_result["epoch_history"],
        "duration_seconds": time.time() - kd_started_at,
    }

    status = {
        **status,
        "status": "complete",
        "knowledge_distillation": knowledge_distillation_summary,
        # Every client (target included) measured on the same three
        # checkpoints, so before/after bars are directly comparable.
        "client_influence": {
            "original": acc_original,
            "after_ascent": acc_after_ascent,
            "final": acc_final,
        },
        "duration_seconds": time.time() - started_at,
    }
    save_unlearning_status(experiment_id, status)
    return status


def get_unlearning_status(experiment_id: str) -> dict:
    """Return the most recently persisted unlearning status.

    Mirrors the conceptual `get_unlearning_status()` from Ch.71.8.
    Raises FileNotFoundError if `start_unlearning` hasn't been called
    yet for this experiment.
    """
    return load_unlearning_status(experiment_id)
