import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import TensorDataset

from core.federated.model import build_model
from core.unlearning.baselines import BASELINES, run_graddiff, run_neggrad_plus, run_retain_finetune


def _setup():
    g = torch.Generator().manual_seed(0)
    ds = TensorDataset(torch.randn(120, 1, 8, 8, generator=g), torch.randint(0, 3, (120,), generator=g))
    model = build_model("transformer", channels=1, num_classes=3)
    return ds, list(range(0, 40)), list(range(40, 120)), model.state_dict()


def _changed(a, b):
    return any(not torch.equal(a[k], b[k]) for k in a)


def test_graddiff_runs_and_changes_weights():
    ds, target, retained, state = _setup()
    out = run_graddiff(state, "transformer", 1, 3, ds, target, retained, steps=5)
    assert out["steps_completed"] == 5
    assert _changed(state, out["final_state_dict"])


def test_retain_finetune_runs_and_changes_weights():
    ds, target, retained, state = _setup()
    out = run_retain_finetune(state, "transformer", 1, 3, ds, target, retained, steps=5)
    assert out["steps_completed"] == 5
    assert _changed(state, out["final_state_dict"])


def test_graddiff_is_seed_deterministic():
    ds, target, retained, state = _setup()
    a = run_graddiff(state, "transformer", 1, 3, ds, target, retained, steps=5, seed=1)
    b = run_graddiff(state, "transformer", 1, 3, ds, target, retained, steps=5, seed=1)
    assert not _changed(a["final_state_dict"], b["final_state_dict"])


def test_empty_inputs_rejected():
    ds, target, retained, state = _setup()
    with pytest.raises(ValueError):
        run_graddiff(state, "transformer", 1, 3, ds, [], retained)
    with pytest.raises(ValueError):
        run_retain_finetune(state, "transformer", 1, 3, ds, target, [])


def test_registry_has_all_baselines():
    assert set(BASELINES) == {"graddiff", "neggrad_plus", "retain_finetune"}


def _drift(a, b):
    return sum(float((a[k].float() - b[k].float()).pow(2).sum()) for k in a) ** 0.5


def test_neggrad_plus_runs_and_changes_weights():
    ds, target, retained, state = _setup()
    out = run_neggrad_plus(state, "transformer", 1, 3, ds, target, retained, steps=5)
    assert out["steps_completed"] == 5
    assert _changed(state, out["final_state_dict"])


def test_neggrad_plus_is_seed_deterministic():
    ds, target, retained, state = _setup()
    a = run_neggrad_plus(state, "transformer", 1, 3, ds, target, retained, steps=5, seed=1)
    b = run_neggrad_plus(state, "transformer", 1, 3, ds, target, retained, steps=5, seed=1)
    assert not _changed(a["final_state_dict"], b["final_state_dict"])


def test_neggrad_plus_rejects_bad_inputs():
    ds, target, retained, state = _setup()
    for bad_beta in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            run_neggrad_plus(state, "transformer", 1, 3, ds, target, retained, beta=bad_beta)
    with pytest.raises(ValueError):
        run_neggrad_plus(state, "transformer", 1, 3, ds, [], retained)
    with pytest.raises(ValueError):
        run_neggrad_plus(state, "transformer", 1, 3, ds, target, [])
    with pytest.raises(ValueError):
        run_neggrad_plus(state, "transformer", 1, 3, ds, target, retained, steps=0)


def test_neggrad_plus_smaller_beta_moves_the_model_further():
    # The paper's warning: small beta (strong ascent) destabilises quickly.
    ds, target, retained, state = _setup()
    small = run_neggrad_plus(state, "transformer", 1, 3, ds, target, retained, steps=10, beta=0.5, seed=0)
    large = run_neggrad_plus(state, "transformer", 1, 3, ds, target, retained, steps=10, beta=0.99, seed=0)
    assert _drift(state, small["final_state_dict"]) > _drift(state, large["final_state_dict"])
