import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import TensorDataset

from core.federated.model import build_model
from core.unlearning.baselines import BASELINES, run_graddiff, run_retain_finetune


def _setup():
    g = torch.Generator().manual_seed(0)
    ds = TensorDataset(torch.randn(120, 1, 8, 8, generator=g), torch.randint(0, 3, (120,), generator=g))
    model = build_model("cnn", channels=1, num_classes=3)
    return ds, list(range(0, 40)), list(range(40, 120)), model.state_dict()


def _changed(a, b):
    return any(not torch.equal(a[k], b[k]) for k in a)


def test_graddiff_runs_and_changes_weights():
    ds, target, retained, state = _setup()
    out = run_graddiff(state, "cnn", 1, 3, ds, target, retained, steps=5)
    assert out["steps_completed"] == 5
    assert _changed(state, out["final_state_dict"])


def test_retain_finetune_runs_and_changes_weights():
    ds, target, retained, state = _setup()
    out = run_retain_finetune(state, "cnn", 1, 3, ds, target, retained, steps=5)
    assert out["steps_completed"] == 5
    assert _changed(state, out["final_state_dict"])


def test_graddiff_is_seed_deterministic():
    ds, target, retained, state = _setup()
    a = run_graddiff(state, "cnn", 1, 3, ds, target, retained, steps=5, seed=1)
    b = run_graddiff(state, "cnn", 1, 3, ds, target, retained, steps=5, seed=1)
    assert not _changed(a["final_state_dict"], b["final_state_dict"])


def test_empty_inputs_rejected():
    ds, target, retained, state = _setup()
    with pytest.raises(ValueError):
        run_graddiff(state, "cnn", 1, 3, ds, [], retained)
    with pytest.raises(ValueError):
        run_retain_finetune(state, "cnn", 1, 3, ds, target, [])


def test_registry_has_both():
    assert set(BASELINES) == {"graddiff", "retain_finetune"}
