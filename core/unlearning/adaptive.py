"""Heterogeneity-adaptive unlearning controller (opt-in extension).

Fixed-strength Gradient Ascent + Knowledge Distillation uses the same
settings for every client. This controller asks one question first --
*how distinctive is the target client's data compared with the rest of
the federation?* -- and uses the answer to set how hard to forget and
how hard to repair.

Hypothesis (to be tested, NOT assumed true): a more distinctive client
leaves a deeper, more specialised footprint, so the model must be pushed
closer to "unseen" behaviour (stricter calibration target) and needs more
repair afterwards (more distillation, leaning on the teacher). A client
that looks like everyone else needs little of either.

The mapping is a deliberately simple, monotone, clamped heuristic. It is
NOT derived from theory and carries no forgetting guarantee. Whether it
beats fixed settings is an empirical question answered by
`core.unlearning.compare_modes`, not by this file.

Heterogeneity h is an ABSOLUTE scale: h = min(1, JS / HETEROGENEITY_SATURATION).
An earlier version min-max normalised across clients, which stretches
tiny, noise-level differences into the full [0, 1] range when every
client is nearly IID. The absolute scale gives h ~ 0 for a near-IID
client. HETEROGENEITY_SATURATION is a design constant, not tuned.
"""
from __future__ import annotations

from core.unlearning.heterogeneity import (
    client_heterogeneity_scores,
    dominant_classes,
    label_distribution,
    labels_of,
)

# Ends of each controlled setting: value at h = 0 -> value at h = 1.
KS_TARGET_RANGE = (0.15, 0.05)   # stop ascent when KS distance <= this
STEPS_CAP_SCALE = 1.0            # ascent step cap = base_steps * (1 + h * this)
KD_EPOCHS_SCALE = 1.0            # kd epochs = base * (1 + h * this)
KD_ALPHA_DROP = 0.4              # kd alpha = base * (1 - h * this)
HETEROGENEITY_SATURATION = 0.2   # JS divergence at which h reaches 1 (untuned)


def absolute_heterogeneity(js: float) -> float:
    """Map a JS divergence to h in [0, 1] on a fixed scale."""
    return max(0.0, min(1.0, js / HETEROGENEITY_SATURATION))


def normalise(scores: dict[int, float], client_id: int) -> float:
    """Min-max normalise `client_id`'s score across all clients to [0, 1]."""
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        return 0.5  # every client equally (in)distinctive: stay neutral
    return (scores[client_id] - lo) / (hi - lo)


def adaptive_settings(
    h: float,
    steps: int,
    kd_epochs: int,
    kd_alpha: float,
) -> dict:
    """Map normalised heterogeneity h in [0, 1] to concrete settings."""
    h = max(0.0, min(1.0, h))
    ks_lo, ks_hi = KS_TARGET_RANGE
    return {
        "ks_target": round(ks_lo + (ks_hi - ks_lo) * h, 4),
        "steps": max(1, int(round(steps * (1 + STEPS_CAP_SCALE * h)))),
        "kd_epochs": max(1, int(round(kd_epochs * (1 + KD_EPOCHS_SCALE * h)))),
        "kd_alpha": round(max(0.05, min(1.0, kd_alpha * (1 - KD_ALPHA_DROP * h))), 4),
    }


def plan_adaptive_unlearning(
    dataset,
    clients: list[dict],
    target_client_id: int,
    num_classes: int,
    steps: int,
    kd_epochs: int,
    kd_alpha: float,
) -> dict:
    """Measure heterogeneity for every client and plan the target's settings."""
    client_labels = {c["client_id"]: labels_of(dataset, c["indices"]) for c in clients}
    scores = client_heterogeneity_scores(client_labels, num_classes)
    h = absolute_heterogeneity(scores[target_client_id])
    relative = normalise(scores, target_client_id)
    target_dist = label_distribution(client_labels[target_client_id], num_classes)
    return {
        "target_js_divergence": round(scores[target_client_id], 6),
        "heterogeneity": round(h, 6),
        "relative_rank_heterogeneity": round(relative, 6),
        "per_client_js_divergence": {str(k): round(v, 6) for k, v in scores.items()},
        "target_dominant_classes": dominant_classes(target_dist),
        "base": {"steps": steps, "kd_epochs": kd_epochs, "kd_alpha": kd_alpha},
        "chosen": adaptive_settings(h, steps, kd_epochs, kd_alpha),
    }
