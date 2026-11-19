# FedEraser (additional comparison method)

FedEraser (Wang et al., "FedEraser: Enabling Efficient Client-Level Data Removal
from Federated Learning Models", IEEE/ACM IWQoS 2021) is added to the Method
Comparison as a published method that works differently from the GA + KD family:
it does not start from the final model. It replays the recorded training history
without the target client.

It was ported from the separate first project (the `Federated-Unlearning`
repository). That repository is unchanged. This is not the "v1 / v2" of
`docs/algorithm_history.md`, which refers to GA + KD algorithm versions.

## How it works

1. During the original training, record each client's update (trained weights
   minus the global weights it started the round from) for every round.
2. To forget a client, start from the initial global weights W_0 and replay the
   rounds using only the retained clients:
   - fast-forward rounds apply the mean of the retained clients' stored updates;
   - milestone rounds (every `--milestone-interval` rounds, and always the last)
     have each retained client run `--cali-steps` real training steps from the
     current unlearned weights; each tensor of that new update is rescaled to the
     norm of the stored historical update, then the mean is applied.
3. The weights after the last round are the unlearned model.

The target client's data is never read.

## Run it

Needs a finished experiment (partition, trained model) and, for gap-to-retrain
numbers, a retraining baseline (`core.evaluation.retraining_cli`).

```
python -m core.unlearning.federaser_cli --experiment-id <id> --device cuda
```

Options: `--client-id`, `--cali-steps` (default 4), `--cali-epochs` (default 1),
`--milestone-interval` (default 2), `--weighted` (sample-weighted mean instead of
plain mean), `--rerecord`.

The command adds one `federaser` row to `experiments/<id>/mode_comparison.json`,
which the Method Comparison page shows. It does not touch the stored global
model, the stored unlearning files, or any other row. Each run replaces the
previous `federaser` row. Run `compare_modes` first if you also want its rows,
because `compare_modes` rewrites the file; FedEraser then merges into it.

## The recorded history

v2 does not store per-round client updates, so the first run re-runs the
experiment's original FedAvg loop (same partition, seed, model, rounds, local
epochs, learning rate and batch size) with a read-only recorder, and saves
`experiments/<id>/fl_round_log.pt`. Later runs reuse it (`--rerecord` redoes it).

- On CPU the re-run reproduces the stored model exactly. On GPU it can differ
  slightly (non-deterministic kernels); the command warns and records
  `history_matches_stored_model` in the row. FedEraser only uses the recorded
  updates, not the stored model.
- Storage is roughly rounds x clients x model size. The row reports it as
  `history_storage_mb`. In a real deployment the server would record during
  training and the re-run would not exist; the row's `duration_seconds` is the
  unlearning time only, and the recording time is reported separately.

## Differences from the first project

v2's models use LayerNorm only, so the first project's ResNet-specific steps
(frozen BatchNorm, training only `layer4` and the classifier, re-initialising the
forgotten class's classifier weights) are not carried over. Calibration updates
every floating-point parameter, with v2's own client optimizer and, by default,
the learning rate of the original run.

## Status and limits

- The default settings are untuned. A first run on `real-mnist` (10 clients,
  Dirichlet alpha 0.1, 10 rounds) with the defaults gave much lower retained
  accuracy than the retraining reference. Do not read the default row as the
  method's best case; sweep `--cali-steps`, `--milestone-interval` and
  `--weighted` and report every setting tried.
- One seed, one target client, one dataset per run. A row is a measured data
  point, not a proof of superiority.
- No formal unlearning guarantee is claimed.
- Not wired into the dashboard's "run" button or the compute backends; it runs
  from the command line and the dashboard displays the saved row.

Tests: `python -m pytest tests/test_federaser.py -q`.
