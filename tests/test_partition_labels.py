import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from core.partitioning.engine import partition_dataset


def _real_like_labels(n=60000, k=10, seed=3):
    # Deliberately NOT index % k, so it differs from the stand-in labels.
    return np.random.default_rng(seed).integers(0, k, size=n)


def test_default_partition_is_marked_stand_in():
    p = partition_dataset("mnist", 5, "non_iid", 1)
    assert p["label_source"] == "stand_in"


def test_real_labels_drive_class_counts_and_are_marked_real():
    labels = _real_like_labels()
    p = partition_dataset("mnist", 5, "non_iid", 1, labels=labels)
    assert p["label_source"] == "real"
    for c in p["clients"]:
        idx = np.array(c["indices"], dtype=int)
        classes, counts = np.unique(labels[idx], return_counts=True)
        assert c["class_counts"] == {int(a): int(b) for a, b in zip(classes, counts)}
    assert sum(c["num_samples"] for c in p["clients"]) == 60000


def test_real_label_skew_differs_from_stand_in_skew():
    a = partition_dataset("mnist", 5, "non_iid", 1)
    b = partition_dataset("mnist", 5, "non_iid", 1, labels=_real_like_labels())
    assert a["clients"][0]["indices"] != b["clients"][0]["indices"]


def test_bad_labels_rejected():
    with pytest.raises(ValueError):
        partition_dataset("mnist", 5, "non_iid", 1, labels=np.zeros(10, dtype=int))
    with pytest.raises(ValueError):
        partition_dataset("mnist", 5, "non_iid", 1, labels=np.full(60000, 99))
