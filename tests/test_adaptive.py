import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from core.unlearning.adaptive import adaptive_settings, normalise
from core.unlearning.heterogeneity import (
    client_heterogeneity_scores,
    dominant_classes,
    js_divergence,
    label_distribution,
)


def test_js_identical_is_zero_and_disjoint_is_one():
    assert js_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert js_divergence([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_js_is_symmetric_and_bounded():
    p, q = [0.7, 0.2, 0.1], [0.1, 0.3, 0.6]
    assert js_divergence(p, q) == pytest.approx(js_divergence(q, p))
    assert 0.0 < js_divergence(p, q) < 1.0


def test_label_distribution_sums_to_one_and_rejects_empty():
    d = label_distribution([0, 0, 1, 2], 3)
    assert d == [0.5, 0.25, 0.25]
    with pytest.raises(ValueError):
        label_distribution([], 3)


def test_scores_rank_the_distinctive_client_highest():
    clients = {
        0: [0, 1, 2, 3] * 25,   # balanced
        1: [0, 1, 2, 3] * 25,   # balanced
        2: [3] * 100,           # only class 3: distinctive
    }
    s = client_heterogeneity_scores(clients, 4)
    assert s[2] > s[0] and s[2] > s[1]
    assert s[0] == pytest.approx(s[1])


def test_scores_need_two_clients():
    with pytest.raises(ValueError):
        client_heterogeneity_scores({0: [0, 1]}, 2)


def test_normalise_min_max_and_neutral_when_all_equal():
    scores = {0: 0.1, 1: 0.3, 2: 0.5}
    assert normalise(scores, 0) == 0.0
    assert normalise(scores, 1) == pytest.approx(0.5)
    assert normalise(scores, 2) == 1.0
    assert normalise({0: 0.2, 1: 0.2}, 0) == 0.5


def test_settings_endpoints_and_monotonicity():
    lo = adaptive_settings(0.0, steps=30, kd_epochs=5, kd_alpha=0.5)
    hi = adaptive_settings(1.0, steps=30, kd_epochs=5, kd_alpha=0.5)
    assert lo == {"ks_target": 0.15, "steps": 30, "kd_epochs": 5, "kd_alpha": 0.5}
    assert hi == {"ks_target": 0.05, "steps": 60, "kd_epochs": 10, "kd_alpha": 0.3}
    mid = adaptive_settings(0.5, steps=30, kd_epochs=5, kd_alpha=0.5)
    assert hi["ks_target"] < mid["ks_target"] < lo["ks_target"]
    assert lo["kd_epochs"] < mid["kd_epochs"] < hi["kd_epochs"]


def test_settings_clamp_out_of_range_h():
    assert adaptive_settings(-3, 30, 5, 0.5) == adaptive_settings(0.0, 30, 5, 0.5)
    assert adaptive_settings(7, 30, 5, 0.5) == adaptive_settings(1.0, 30, 5, 0.5)


def test_dominant_classes():
    assert dominant_classes([0.1, 0.6, 0.05, 0.25], top=2) == [1, 3]


def test_absolute_heterogeneity_is_neutral_for_near_iid_clients():
    from core.unlearning.adaptive import HETEROGENEITY_SATURATION, absolute_heterogeneity

    assert absolute_heterogeneity(0.0005) < 0.01
    assert absolute_heterogeneity(HETEROGENEITY_SATURATION) == 1.0
    assert absolute_heterogeneity(0.9) == 1.0
    assert absolute_heterogeneity(-1) == 0.0
