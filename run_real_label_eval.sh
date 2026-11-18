#!/usr/bin/env bash
# Full real-label experiment: partition -> FL -> unlearn -> retrain -> compare.
# usage: bash run_real_label_eval.sh <dataset> <experiment-id> [gpu-index]
#   e.g. nohup bash run_real_label_eval.sh cifar10 cifar10-real 2 > cifar10-real.log 2>&1 &
set -euo pipefail
DS=${1:?dataset id, e.g. mnist or cifar10}
ID=${2:?experiment id}
export CUDA_VISIBLE_DEVICES=${3:-0}
export EXP_ID="$ID"

echo "== partition (REAL labels)"
python -m core.partitioning.cli --dataset "$DS" --num-clients 10 --strategy non_iid \
  --seed 42 --experiment-id "$ID" --real-labels > /dev/null

echo "== pick the most label-distinctive client (real labels)"
TARGET=$(python - <<'PY'
import os
from core.partitioning.storage import load_partition
from core.unlearning.heterogeneity import client_heterogeneity_scores
p = load_partition(os.environ["EXP_ID"])
assert p.get("label_source") == "real", "partition is not real-label"
labels = {c["client_id"]: [int(k) for k, v in c["class_counts"].items() for _ in range(v)]
          for c in p["clients"]}
s = client_heterogeneity_scores(labels, p["num_classes"])
print(max(s, key=s.get))
PY
)
echo "target client: $TARGET"

echo "== federated training (20 rounds, 1 local epoch)"
python - <<'PY'
import os
from core.partitioning.storage import load_partition
from core.federated.server import run_federated_training
i = os.environ["EXP_ID"]
run_federated_training(i, load_partition(i), "cnn", rounds=20, local_epochs=1, seed=42, device="cuda")
PY

echo "== unlearning once (so the retraining baseline knows the target)"
TARGET="$TARGET" python - <<'PY'
import os
from core.unlearning.unlearning_engine import start_unlearning
start_unlearning(os.environ["EXP_ID"], int(os.environ["TARGET"]), device="cuda")
PY

echo "== retraining baseline"
python -m core.evaluation.retraining_cli --experiment-id "$ID" --device cuda > /dev/null

echo "== comparison (fixed, calibrated, adaptive, GradDiff, retain-only fine-tune)"
python -m core.unlearning.compare_modes --experiment-id "$ID" --device cuda
