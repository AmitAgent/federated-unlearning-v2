# Algorithm history: v1 (baseline) and v2 (candidate)

Purpose: a record of what the adaptive Gradient Ascent + Knowledge Distillation
pipeline did **before** the v2 change, the assumptions behind it, and how to go
back. Nothing in the code falls back automatically; reverting is a manual step.

Code base for v1: git commit `4d0eb6f` ("Dashboard partition: build non-IID
splits from real labels"). v1 results are in `docs/results.md` (single seed,
MNIST, small ViT).

## v1: assumptions and settings

Assumption: a client with a more distinctive label distribution leaves a deeper
footprint, so it needs (a) stricter forgetting and (b) **more** repair
afterwards. The mapping from heterogeneity `h` in [0, 1] is a clamped heuristic,
not derived from theory.

| Setting | v1 value |
|---|---|
| Calibration reference set | generic held-out test samples, indices 1000 to 1500 |
| `KS_TARGET_RANGE` (h=0 to h=1) | 0.15 to 0.05 |
| `STEPS_CAP_SCALE` | 1.0 (ascent step cap = base * (1 + h)) |
| `KD_EPOCHS_SCALE` | 1.0 (distillation epochs = base * (1 + h)) |
| `KD_ALPHA_DROP` | 0.4 (kd alpha = base * (1 - 0.4 h), more weight on teacher) |
| `HETEROGENEITY_SATURATION` | 0.2 (untuned) |

v1 measured result, skewed split (Dirichlet alpha 0.1, forget client 0):
adaptive forget acc 0.8269, retained acc 0.9323, forget gap 0.5210, retained
gap 0.0719 (retrain reference: forget 0.3059, retained 0.8604). It under-forgot.

## v2: what changed and why

Hypotheses (not yet confirmed by an experiment):

1. The calibration reference was a generic test slice, so "looks unseen" was
   judged against the wrong digits for a label-skewed client.
2. The distillation teacher is the original model, which still knows what the
   target client taught it. More distillation epochs and a heavier teacher term
   (v1 behaviour for distinctive clients) may pull that knowledge back.

| Setting | v2 value |
|---|---|
| Calibration reference set (adaptive mode) | held-out test samples matching the target client's class mix (`class_matched_reference_indices`) |
| `KD_EPOCHS_SCALE` | 0.0 (no extra distillation) |
| `KD_ALPHA_RISE` (replaces `KD_ALPHA_DROP`) | 0.4 (kd alpha = base * (1 + 0.4 h), more weight on hard labels) |
| Unchanged | `KS_TARGET_RANGE`, `STEPS_CAP_SCALE`, `HETEROGENEITY_SATURATION` |

Also added (harmless, keep either way): diagnostics in `compare_modes` showing
GA stop reason and forget accuracy original, after ascent, after distillation;
a `--modes` option to re-run only selected modes.

## v2 results

Fill in after running:
`python -m core.unlearning.compare_modes --experiment-id <skewed-id> --modes adaptive`

| mode | forget acc | retained acc | MIA | gap: forget | gap: retained |
|---|---|---|---|---|---|
| adaptive v1 | 0.8269 | 0.9323 | 0.505 | 0.5210 | 0.0719 |
| adaptive v2 | | | | | |

Diagnostics (v2): GA stop reason ______ ; forget acc original ____ , after
ascent ____ , final ____ .

## Decision rule (set before seeing v2 numbers)

Keep v2 if forget gap is clearly lower than v1 (for example below about 0.3) and
retained accuracy stays about 0.9 or higher. Otherwise go back to v1 and report
v2 as a tested negative result. With one seed, differences under about 1 point
are noise.

## How to go back to v1

Restore the v1 algorithm files (diagnostics and `--modes` in
`compare_modes.py` can stay):

```
git checkout 4d0eb6f -- core/unlearning/adaptive.py core/unlearning/unlearning_engine.py tests/test_adaptive.py
```

If the v2 patch was committed as one commit instead, `git revert <that commit>`
undoes everything, or `git apply -R forgetting-fix.patch` if it is still
uncommitted. After reverting, run `python -m pytest tests -q`.
