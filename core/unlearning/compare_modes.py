"""Compare fixed, calibrated and adaptive unlearning on one experiment.

    python -m core.unlearning.compare_modes --experiment-id <id>

Runs the three modes against the same target client, scores every
resulting model with the SAME code that scores the original and retrained
models (`evaluate_model_state`), and measures each one's distance to the
retrained-without-the-client reference. Writes
``experiments/<id>/mode_comparison.json`` and prints a table.

The stored unlearning files (status, GA model, final model) are backed up
first and restored afterwards, so the Professor Demo experiment is left
exactly as it was.

Honest limits: one seed, one target client, one dataset per run. A single
row proves nothing about superiority; it is a measured data point.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

from core.datasets.base import get_dataset
from core.evaluation.evaluation_engine import (
    DEFAULT_NON_MEMBER_SAMPLE_SIZE,
    evaluate_model_state,
)
from core.evaluation.storage import load_retraining_results
from core.federated.storage import load_model_checkpoint
from core.partitioning.storage import load_partition
from core.unlearning.baselines import BASELINES
from core.unlearning.storage import (
    EXPERIMENTS_DIR,
    load_final_checkpoint,
    load_unlearning_status,
)
from core.unlearning.unlearning_engine import start_unlearning

MODES = {
    "fixed": {},
    "calibrated": {"calibrated": True},
    "adaptive": {"adaptive": True},
}
_BACKED_UP = ("unlearning_status.json", "unlearning_ga_model.pt", "unlearning_final_model.pt")


def _gap(a: float, b: float) -> float:
    return round(abs(a - b), 6)


def compare_modes(
    experiment_id: str,
    client_id: int | None = None,
    device: str = "cpu",
    which: str = "all",
) -> dict:
    """which: 'all' (our 3 modes + baselines), 'core' (our 3 modes), or
    'baselines' (only the established baselines, merged into any existing
    mode_comparison.json)."""
    if which not in ("all", "core", "baselines"):
        raise ValueError("which must be 'all', 'core' or 'baselines'")
    exp_dir = EXPERIMENTS_DIR / experiment_id
    try:
        retrain = load_retraining_results(experiment_id)
    except FileNotFoundError:
        retrain = None
        print(
            "WARNING: no retraining baseline for this experiment, so gap-to-retrain "
            "cannot be computed. Run `python -m core.evaluation.retraining_cli "
            f"--experiment-id {experiment_id}` first for the full comparison."
        )
    if client_id is None:
        if retrain is not None:
            client_id = retrain["target_client_id"]
        else:
            client_id = load_unlearning_status(experiment_id)["target_client_id"]
    have_reference = retrain is not None and retrain["target_client_id"] == client_id
    ref = retrain["metrics"] if have_reference else None

    partition = load_partition(experiment_id)
    clients = partition["clients"]
    target = next(c for c in clients if c["client_id"] == client_id)
    retained = [c for c in clients if c["client_id"] != client_id]
    base_ckpt = load_model_checkpoint(experiment_id)
    ds = get_dataset(partition["dataset"])
    train_data, test_data = ds.get_train(), ds.get_test()
    non_members = list(range(min(DEFAULT_NON_MEMBER_SAMPLE_SIZE, len(test_data))))

    out_path = exp_dir / "mode_comparison.json"
    rows: dict[str, dict] = {}
    if which == "baselines" and out_path.exists():
        rows = json.loads(out_path.read_text()).get("modes", {})

    backup = Path(tempfile.mkdtemp(prefix="unlearn-backup-"))
    for name in _BACKED_UP:
        if (exp_dir / name).exists():
            shutil.copy2(exp_dir / name, backup / name)

    try:
        for mode, kwargs in (MODES.items() if which != "baselines" else []):
            status = start_unlearning(experiment_id, client_id, device=device, **kwargs)
            final = load_final_checkpoint(experiment_id)
            m = evaluate_model_state(
                final["state_dict"], base_ckpt["model_id"], base_ckpt["channels"],
                base_ckpt["num_classes"], train_data, test_data, target, retained,
                non_members, device=device,
            )
            ga = status["gradient_ascent"]
            hp = status["hyperparameters"]
            row = {
                "global_accuracy": m["global_accuracy"],
                "forget_client_accuracy": m["forget_client_accuracy"],
                "retained_client_accuracy": m["retained_client_accuracy"],
                "mia_success": m["mia_success"],
                "ga_steps_completed": ga["steps_completed"],
                "ga_final_loss": ga["final_loss"],
                "kd_epochs": status["knowledge_distillation"]["epochs_completed"],
                "kd_alpha": status["knowledge_distillation"]["alpha"],
                "duration_seconds": round(status["duration_seconds"], 1),
            }
            if "adaptive" in hp:
                row["adaptive"] = {
                    "target_js_divergence": hp["adaptive"]["target_js_divergence"],
                    "heterogeneity": hp["adaptive"]["heterogeneity"],
                    "chosen": hp["adaptive"]["chosen"],
                }
            if ref is not None:
                row["gap_to_retrain"] = {
                    "forget_accuracy": _gap(row["forget_client_accuracy"], ref["forget_client_accuracy"]),
                    "retained_accuracy": _gap(row["retained_client_accuracy"], ref["retained_client_accuracy"]),
                    "global_accuracy": _gap(row["global_accuracy"], ref["global_accuracy"]),
                    "mia_success": _gap(row["mia_success"], ref["mia_success"]),
                }
            rows[mode] = row

        if which != "core":
            retained_pool = [i for c in retained for i in c["indices"]]
            for name, fn in BASELINES.items():
                t0 = time.time()
                res = fn(
                    global_state=base_ckpt["state_dict"], model_id=base_ckpt["model_id"],
                    channels=base_ckpt["channels"], num_classes=base_ckpt["num_classes"],
                    dataset=train_data, target_indices=target["indices"],
                    retained_indices=retained_pool, device=device,
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
                    "ga_steps_completed": res["steps_completed"],
                    "ga_final_loss": None,
                    "kd_epochs": 0,
                    "kd_alpha": None,
                    "duration_seconds": round(time.time() - t0, 1),
                    "kind": "established baseline",
                }
                if ref is not None:
                    row["gap_to_retrain"] = {
                        "forget_accuracy": _gap(row["forget_client_accuracy"], ref["forget_client_accuracy"]),
                        "retained_accuracy": _gap(row["retained_client_accuracy"], ref["retained_client_accuracy"]),
                        "global_accuracy": _gap(row["global_accuracy"], ref["global_accuracy"]),
                        "mia_success": _gap(row["mia_success"], ref["mia_success"]),
                    }
                rows[name] = row
    finally:
        for name in _BACKED_UP:
            if (backup / name).exists():
                shutil.copy2(backup / name, exp_dir / name)
        shutil.rmtree(backup, ignore_errors=True)

    result = {
        "experiment_id": experiment_id,
        "target_client_id": client_id,
        "dataset": partition["dataset"],
        "retrained_reference": ref if have_reference else None,
        "modes": rows,
        "caveat": "single seed, single target client; a data point, not a proof of superiority",
    }
    out_path.write_text(json.dumps(result, indent=2))
    return result


def format_table(result: dict) -> str:
    ref = result["retrained_reference"]
    lines = [
        "| mode | forget acc | retained acc | MIA | GA steps | KD ep | gap: forget | gap: retained |",
        "|---|---|---|---|---|---|---|---|",
    ]
    if ref:
        lines.append(
            f"| retrain (reference) | {ref['forget_client_accuracy']:.4f} | "
            f"{ref['retained_client_accuracy']:.4f} | {ref['mia_success']:.3f} | - | - | - | - |"
        )
    for mode, r in result["modes"].items():
        g = r.get("gap_to_retrain", {})
        lines.append(
            f"| {mode} | {r['forget_client_accuracy']:.4f} | {r['retained_client_accuracy']:.4f} | "
            f"{r['mia_success']:.3f} | {r['ga_steps_completed']} | {r['kd_epochs']} | "
            f"{g.get('forget_accuracy', '-')} | {g.get('retained_accuracy', '-')} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--client-id", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--which", default="all", choices=["all", "core", "baselines"],
        help="all = our 3 modes + established baselines; core = our 3 only; baselines = only GradDiff and retain-only fine-tune (merged into an existing result file)",
    )
    args = p.parse_args(argv)
    result = compare_modes(args.experiment_id, args.client_id, args.device, args.which)
    print(format_table(result))
    print("\nSaved experiments/%s/mode_comparison.json" % args.experiment_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
