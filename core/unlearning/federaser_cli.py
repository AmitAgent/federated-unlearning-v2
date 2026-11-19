"""Run FedEraser on a stored experiment and add it to the method comparison.

    python -m core.unlearning.federaser_cli --experiment-id <id>

FedEraser needs the per-round client updates of the original training run,
which v2 does not store. This command therefore:

1. Re-runs the experiment's original FedAvg loop (same partition, seed, model,
   rounds, local epochs, lr and batch size from ``fl_history.json``) with a
   read-only recorder attached, and saves the recording to
   ``experiments/<id>/fl_round_log.pt``. Skipped if that file already exists
   (use ``--rerecord`` to redo it). On CPU the re-run reproduces the stored
   model exactly; this is checked and reported. In a real deployment the
   server would record during the first run and this step would not exist.
2. Runs FedEraser (``core.unlearning.federaser``) for the target client.
3. Scores the result with the SAME ``evaluate_model_state`` used for every other
   method and merges one ``federaser`` row into
   ``experiments/<id>/mode_comparison.json`` (created if absent).

It does not touch the stored global model, the stored unlearning files, or
any other row of the comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from core.datasets.base import get_dataset
from core.evaluation.evaluation_engine import (
    DEFAULT_NON_MEMBER_SAMPLE_SIZE,
    evaluate_model_state,
)
from core.evaluation.storage import load_retraining_results
from core.federated.server import fedavg_train
from core.federated.storage import load_fl_history, load_model_checkpoint
from core.partitioning.storage import load_partition
from core.unlearning.federaser import (
    DEFAULT_CALI_EPOCHS,
    DEFAULT_CALI_STEPS,
    DEFAULT_MILESTONE_INTERVAL,
    RoundLogger,
    history_size_bytes,
    load_history,
    run_federaser,
    save_history,
)
from core.unlearning.storage import EXPERIMENTS_DIR, load_unlearning_status

LOG_NAME = "fl_round_log.pt"
KIND = "FedEraser (Wang et al., IWQoS 2021) - needs recorded round history"


def _gap(a: float, b: float) -> float:
    return round(abs(a - b), 6)


def record_round_log(experiment_id: str, device: str = "cpu", force: bool = False) -> dict:
    """Return ``{"history", "path", "recorded", "recording_seconds", "matches_stored_model", "max_abs_diff"}``."""
    path = EXPERIMENTS_DIR / experiment_id / LOG_NAME
    if path.exists() and not force:
        return {
            "history": load_history(path), "path": path, "recorded": False,
            "recording_seconds": None, "matches_stored_model": None, "max_abs_diff": None,
        }

    fl = load_fl_history(experiment_id)
    partition = load_partition(experiment_id)
    logger = RoundLogger()
    t0 = time.time()
    _, final_state = fedavg_train(
        experiment_id=experiment_id,
        partition=partition,
        model_id=fl["model"],
        rounds=fl["rounds"],
        local_epochs=fl["local_epochs"],
        seed=fl["seed"],
        lr=fl.get("lr", 0.01),
        batch_size=fl.get("batch_size", 32),
        device=device,
        round_callback=logger,
    )
    seconds = time.time() - t0
    history = logger.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_history(history, path)

    stored = load_model_checkpoint(experiment_id)["state_dict"]
    diff = max(
        float((final_state[k].detach().cpu().float() - stored[k].detach().cpu().float()).abs().max())
        for k in stored
    )
    return {
        "history": history, "path": path, "recorded": True,
        "recording_seconds": round(seconds, 1),
        "matches_stored_model": diff == 0.0, "max_abs_diff": diff,
    }


def run_for_experiment(
    experiment_id: str,
    client_id: int | None = None,
    device: str = "cpu",
    rerecord: bool = False,
    cali_steps: int = DEFAULT_CALI_STEPS,
    cali_epochs: int = DEFAULT_CALI_EPOCHS,
    milestone_interval: int = DEFAULT_MILESTONE_INTERVAL,
    weighted: bool = False,
) -> dict:
    try:
        retrain = load_retraining_results(experiment_id)
    except FileNotFoundError:
        retrain = None
        print(
            "WARNING: no retraining baseline for this experiment, so gap-to-retrain "
            "cannot be computed.", file=sys.stderr,
        )
    if client_id is None:
        client_id = (
            retrain["target_client_id"] if retrain is not None
            else load_unlearning_status(experiment_id)["target_client_id"]
        )
    have_reference = retrain is not None and retrain["target_client_id"] == client_id
    ref = retrain["metrics"] if have_reference else None

    rec = record_round_log(experiment_id, device=device, force=rerecord)
    if rec["recorded"] and not rec["matches_stored_model"]:
        print(
            f"WARNING: the recorded re-run differs from the stored model "
            f"(max abs diff {rec['max_abs_diff']:.3g}); expected on GPU because of "
            f"non-deterministic kernels. FedEraser only uses the recorded updates.",
            file=sys.stderr,
        )

    fl = load_fl_history(experiment_id)
    partition = load_partition(experiment_id)
    clients = partition["clients"]
    target = next(c for c in clients if c["client_id"] == client_id)
    retained = [c for c in clients if c["client_id"] != client_id]
    base_ckpt = load_model_checkpoint(experiment_id)
    ds = get_dataset(partition["dataset"])
    train_data, test_data = ds.get_train(), ds.get_test()
    non_members = list(range(min(DEFAULT_NON_MEMBER_SAMPLE_SIZE, len(test_data))))

    res = run_federaser(
        history=rec["history"], model_id=base_ckpt["model_id"], channels=base_ckpt["channels"],
        num_classes=base_ckpt["num_classes"], dataset=train_data, clients=clients,
        target_client_id=client_id, lr=fl.get("lr", 0.01), cali_steps=cali_steps,
        cali_epochs=cali_epochs, milestone_interval=milestone_interval,
        batch_size=fl.get("batch_size", 32), weighted=weighted, device=device, seed=fl["seed"],
    )
    m = evaluate_model_state(
        res["final_state_dict"], base_ckpt["model_id"], base_ckpt["channels"],
        base_ckpt["num_classes"], train_data, test_data, target, retained,
        non_members, device=device,
    )
    row = {
        "global_accuracy": m["global_accuracy"],
        "forget_client_accuracy": m["forget_client_accuracy"],
        "retained_client_accuracy": m["retained_client_accuracy"],
        "mia_success": m["mia_success"],
        # Same keys as the other baseline rows so the table and dashboard read it.
        "ga_steps_completed": res["steps_completed"],
        "ga_final_loss": None,
        "kd_epochs": 0,
        "kd_alpha": None,
        # Unlearning time only: replay + calibration. Recording the history is
        # part of ordinary training in a real deployment, reported separately.
        "duration_seconds": round(res["duration_seconds"], 1),
        "kind": KIND,
        "federaser": {
            "rounds_replayed": res["rounds_replayed"],
            "milestone_rounds": res["milestone_rounds"],
            "calibration_steps_per_client": cali_steps,
            "milestone_interval": milestone_interval,
            "weighted": weighted,
            "history_storage_mb": round(history_size_bytes(rec["history"]) / 1e6, 2),
            "history_recording_seconds": rec["recording_seconds"],
            "history_matches_stored_model": rec["matches_stored_model"],
        },
    }
    if ref is not None:
        row["gap_to_retrain"] = {
            "forget_accuracy": _gap(row["forget_client_accuracy"], ref["forget_client_accuracy"]),
            "retained_accuracy": _gap(row["retained_client_accuracy"], ref["retained_client_accuracy"]),
            "global_accuracy": _gap(row["global_accuracy"], ref["global_accuracy"]),
            "mia_success": _gap(row["mia_success"], ref["mia_success"]),
        }

    out_path = EXPERIMENTS_DIR / experiment_id / "mode_comparison.json"
    if out_path.exists():
        result = json.loads(out_path.read_text())
    else:
        result = {
            "experiment_id": experiment_id,
            "target_client_id": client_id,
            "dataset": partition["dataset"],
            "retrained_reference": ref if have_reference else None,
            "modes": {},
            "caveat": "single seed, single target client; a data point, not a proof of superiority",
        }
    result.setdefault("modes", {})["federaser"] = row
    out_path.write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--client-id", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--rerecord", action="store_true", help="Redo the history recording even if it exists.")
    p.add_argument("--cali-steps", type=int, default=DEFAULT_CALI_STEPS)
    p.add_argument("--cali-epochs", type=int, default=DEFAULT_CALI_EPOCHS)
    p.add_argument("--milestone-interval", type=int, default=DEFAULT_MILESTONE_INTERVAL)
    p.add_argument("--weighted", action="store_true", help="Sample-weighted mean instead of plain mean.")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    try:
        result = run_for_experiment(
            args.experiment_id, args.client_id, args.device, args.rerecord,
            args.cali_steps, args.cali_epochs, args.milestone_interval, args.weighted,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"comparison": result}))
        return 0
    from core.unlearning.compare_modes import format_table

    print(format_table(result))
    print("\nSaved experiments/%s/mode_comparison.json" % args.experiment_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
