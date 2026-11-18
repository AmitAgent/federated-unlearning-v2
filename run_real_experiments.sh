#!/usr/bin/env bash
# Real-data experiments on MNIST, CIFAR-10 and CIFAR-100.
#
# For each dataset this runs the whole pipeline on the REAL torchvision
# data (downloaded into ./data), with a label-skewed split built from the
# dataset's real labels, and the current transformer model:
#   partition -> FedAvg training -> unlearning -> evaluation -> retraining
#   baseline (-> optional method comparison)
# Everything is written to experiments/real-<dataset>/ and then shows up in
# Experiment History as a real, measured, replayable experiment.
#
# usage:  bash run_real_experiments.sh [dataset ...]
#   e.g.  bash run_real_experiments.sh                 # mnist cifar10 cifar100
#         bash run_real_experiments.sh cifar10         # just one
#         ROUNDS=10 bash run_real_experiments.sh mnist # fewer rounds, faster
#
# environment (all optional):
#   DEVICE        cuda | mps | cpu           (default: auto-detect)
#   ROUNDS        FedAvg rounds              (default 20)
#   LOCAL_EPOCHS  local epochs per round     (default 1)
#   NUM_CLIENTS   clients                    (default 10)
#   ALPHA         Dirichlet alpha, smaller = more skewed (default 0.1)
#   SEED          seed                       (default 42)
#   COMPARE       1 = also run the method comparison (slow) (default 0)
#   FORCE         1 = overwrite an existing experiments/real-<dataset>/
#
# Needs torch + torchvision and internet access for the first dataset
# download. Training a transformer on CIFAR-100 on a laptop CPU can take
# hours; a GPU (cuda) or ROUNDS=10 helps.
set -euo pipefail
cd "$(dirname "$0")"

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then DATASETS=(mnist cifar10 cifar100); fi

ROUNDS=${ROUNDS:-20}
LOCAL_EPOCHS=${LOCAL_EPOCHS:-1}
NUM_CLIENTS=${NUM_CLIENTS:-10}
ALPHA=${ALPHA:-0.1}
SEED=${SEED:-42}
COMPARE=${COMPARE:-0}
FORCE=${FORCE:-0}

if [ -z "${DEVICE:-}" ]; then
  DEVICE=$(python - <<'PY'
import torch
if torch.cuda.is_available():
    print("cuda")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY
)
fi
echo "device: $DEVICE | rounds: $ROUNDS | clients: $NUM_CLIENTS | alpha: $ALPHA | seed: $SEED"

run_one() {
  local DS="$1"
  local ID="real-$DS"
  export EXP_ID="$ID" DS DEVICE ROUNDS LOCAL_EPOCHS SEED

  if [ -e "experiments/$ID" ]; then
    if [ "$FORCE" = "1" ]; then
      echo "!! FORCE=1: removing existing experiments/$ID"
      rm -rf -- "experiments/$ID"
    else
      echo "!! experiments/$ID already exists -- skipping (set FORCE=1 to overwrite)"
      return 0
    fi
  fi

  echo
  echo "================ $DS ($ID) ================"

  echo "== 1/5 partition (real labels, Dirichlet alpha=$ALPHA)"
  python -m core.partitioning.cli --dataset "$DS" --num-clients "$NUM_CLIENTS" \
    --strategy non_iid --alpha "$ALPHA" --seed "$SEED" --experiment-id "$ID" \
    --real-labels > /dev/null

  echo "== choose the most label-distinctive client as the target"
  TARGET=$(python - <<'PY'
import os
from core.partitioning.storage import load_partition
from core.unlearning.heterogeneity import client_heterogeneity_scores
p = load_partition(os.environ["EXP_ID"])
assert p.get("label_source") == "real", "partition is not built from real labels"
labels = {c["client_id"]: [int(k) for k, v in c["class_counts"].items() for _ in range(v)]
          for c in p["clients"]}
s = client_heterogeneity_scores(labels, p["num_classes"])
print(max(s, key=s.get))
PY
)
  export TARGET
  echo "target client: $TARGET"

  echo "== 2/5 federated training ($ROUNDS rounds, $LOCAL_EPOCHS local epoch(s))"
  python - <<'PY'
import os
from core.partitioning.storage import load_partition
from core.federated.server import run_federated_training
i = os.environ["EXP_ID"]
run_federated_training(
    i, load_partition(i), "transformer",
    rounds=int(os.environ["ROUNDS"]), local_epochs=int(os.environ["LOCAL_EPOCHS"]),
    seed=int(os.environ["SEED"]), device=os.environ["DEVICE"],
)
PY

  echo "== 3/5 unlearning (gradient ascent + knowledge distillation) on client $TARGET"
  python - <<'PY'
import os
from core.unlearning.unlearning_engine import start_unlearning
start_unlearning(os.environ["EXP_ID"], int(os.environ["TARGET"]), device=os.environ["DEVICE"])
PY

  echo "== 4/5 evaluation (accuracy + membership inference)"
  python -m core.evaluation.cli --experiment-id "$ID" --device "$DEVICE" > /dev/null

  echo "== 5/5 retraining baseline + comparison"
  python -m core.evaluation.retraining_cli --experiment-id "$ID" --device "$DEVICE" > /dev/null

  if [ "$COMPARE" = "1" ]; then
    echo "== method comparison (slow)"
    python -m core.unlearning.compare_modes --experiment-id "$ID" --device "$DEVICE"
  fi

  python - <<'PY'
import json, os
i = os.environ["EXP_ID"]
e = json.load(open(f"experiments/{i}/evaluation_results.json"))
print(f"\n--- {i}: target client {e.get('target_client_id')} ---")
for k in ("before", "after"):
    r = e[k]
    ret = list(r["retained_client_accuracy_per_client"].values())
    print(f"{k:>6}: global {r['global_accuracy']:.4f} | target(forget) {r['forget_client_accuracy']:.4f} "
          f"| retained mean {sum(ret)/len(ret):.4f}")
PY
}

for ds in "${DATASETS[@]}"; do
  case "$ds" in
    mnist|cifar10|cifar100) run_one "$ds" ;;
    *) echo "unknown dataset '$ds' (use mnist, cifar10 or cifar100)"; exit 1 ;;
  esac
done

echo
echo "Done. Open Experiment History in the dashboard: real-<dataset> rows are real, measured runs."
