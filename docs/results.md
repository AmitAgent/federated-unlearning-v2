# Results

> Algorithm versions (v1 baseline, v2 candidate), assumptions and how to revert: see [`algorithm_history.md`](./algorithm_history.md). The numbers below are v1.

All numbers come from `experiments/<id>/mode_comparison.json` produced by
`python -m core.unlearning.compare_modes --which all`.

**Setup (both experiments):** small Vision Transformer, MNIST, 5 clients,
10 FedAvg rounds, 1 local epoch, seed 42. The *retrain* row is FedAvg trained
from scratch without the target client under the same protocol, and is the
reference every gap is measured against. Single seed, no confidence intervals.

Modes:
- **fixed / calibrated / adaptive**: this project's Gradient Ascent + Knowledge
  Distillation pipeline (fixed step budget, calibrated stopping, heterogeneity-adaptive stopping).
- **graddiff, neggrad_plus**: published baselines (NegGrad+ follows Kurmanji et al., 2023).
- **retain_finetune**: fine-tune on retained clients only.

## Experiment 1: IID split, forget client 2

| mode | forget acc | retained acc | MIA | gap: forget | gap: retained |
|---|---|---|---|---|---|
| retrain (reference) | 0.9581 | 0.9622 | 0.507 | - | - |
| fixed | 0.9617 | 0.9653 | 0.512 | 0.0036 | 0.0031 |
| calibrated | 0.9647 | 0.9673 | 0.510 | 0.0066 | 0.0051 |
| adaptive | 0.9649 | 0.9682 | 0.510 | 0.0068 | 0.0060 |
| graddiff | 0.2588 | 0.2631 | 0.515 | 0.6993 | 0.6991 |
| neggrad_plus | 0.9623 | 0.9624 | 0.512 | 0.0042 | 0.0001 |
| retain_finetune | 0.9584 | 0.9599 | 0.512 | 0.0003 | 0.0023 |

Reading: with IID data the target client holds nothing the other clients lack,
so a model that never saw it scores the same on its data. Every method except
GradDiff stays within 0.7 points of retraining, which is inside noise for a
single seed. GradDiff collapses the model. Unlearning took 374 s versus 185 s
for full retraining here, so there is no speedup at this scale.

## Experiment 2: label-skewed split (Dirichlet alpha = 0.1, real labels), forget client 0

Client 0 is dominated by digit 3 (5,526 of 8,505 samples); the other four
clients hold only 605 threes in total.

Original model: global 0.898, forget 0.848, retained 0.899.

| mode | forget acc | retained acc | MIA | gap: forget | gap: retained |
|---|---|---|---|---|---|
| retrain (reference) | 0.3059 | 0.8604 | 0.500 | - | - |
| fixed | 0.7861 | 0.9390 | 0.502 | 0.4802 | 0.0787 |
| calibrated | 0.8369 | 0.9465 | 0.505 | 0.5310 | 0.0861 |
| adaptive | 0.8269 | 0.9323 | 0.505 | 0.5210 | 0.0719 |
| graddiff | 0.0114 | 0.4796 | 0.502 | 0.2945 | 0.3808 |
| neggrad_plus | 0.3005 | 0.9184 | 0.502 | 0.0054 | 0.0580 |
| retain_finetune | 0.7002 | 0.9295 | 0.500 | 0.3942 | 0.0691 |

Reading:
- The skewed split makes forgetting measurable: retraining without client 0
  drops its accuracy from 0.848 to 0.306.
- **NegGrad+ matches the retrained model on forget accuracy** (gap 0.005) while
  keeping retained accuracy at 0.918.
- **The Gradient Ascent + Distillation modes preserve utility but under-forget**:
  forget accuracy only falls from 0.848 to 0.79-0.84, far from the 0.306 target.
  The calibrated and adaptive modes stop gradient ascent after 3 steps, which
  appears too early on this split (not yet verified by an ablation).
- GradDiff overshoots (forget 0.011) and destroys retained accuracy (0.480).
- Retained accuracy of the unlearning modes is above the retrain reference
  because the starting model was still improving at round 10 and distillation
  continues training it. That is extra training, not better unlearning.

## Limitations

- One seed and one dataset; differences under about 1 point should not be read as real.
- The membership-inference attack uses only 200 non-member samples, so MIA values near 0.5 are weak evidence.
- Forget accuracy is measured on the target client's training data.
- The skewed base model reached 0.898 global accuracy, below the IID model's 0.96.
