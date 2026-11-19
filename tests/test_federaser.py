import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import Dataset, TensorDataset

import core.federated.server as server
from core.unlearning.federaser import (
    RoundLogger,
    history_size_bytes,
    load_history,
    rescale_to_history,
    run_federaser,
    save_history,
)

NUM_CLASSES = 3
N_CLIENTS = 3
PER_CLIENT = 24


class _FakeDataset:
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        n = N_CLIENTS * PER_CLIENT
        self._train = TensorDataset(
            torch.randn(n, 1, 8, 8, generator=g), torch.randint(0, NUM_CLASSES, (n,), generator=g)
        )
        self._test = TensorDataset(
            torch.randn(30, 1, 8, 8, generator=g), torch.randint(0, NUM_CLASSES, (30,), generator=g)
        )

    def get_train(self):
        return self._train

    def get_test(self):
        return self._test


@pytest.fixture
def fake_env(monkeypatch):
    ds = _FakeDataset()
    monkeypatch.setattr(server, "get_dataset", lambda _id: ds)
    monkeypatch.setattr(
        server, "load_manifest",
        lambda: {"fake": {"channels": 1, "num_classes": NUM_CLASSES, "modality": "image"}},
    )
    clients = [
        {"client_id": i, "indices": list(range(i * PER_CLIENT, (i + 1) * PER_CLIENT))}
        for i in range(N_CLIENTS)
    ]
    partition = {"dataset": "fake", "num_clients": N_CLIENTS, "clients": clients}
    return ds, partition


def _train(partition, callback=None, rounds=4):
    return server.fedavg_train(
        experiment_id="t", partition=partition, model_id="transformer",
        rounds=rounds, local_epochs=1, seed=7, lr=0.01, batch_size=8,
        round_callback=callback,
    )


def _same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_recording_hook_does_not_change_training(fake_env):
    """The callback is read-only: identical weights and metrics with and without it."""
    _, partition = fake_env
    hist_a, state_a = _train(partition)
    hist_b, state_b = _train(partition, callback=RoundLogger())
    assert _same(state_a, state_b)
    for ra, rb in zip(hist_a["round_history"], hist_b["round_history"]):
        assert ra["accuracy"] == rb["accuracy"] and ra["loss"] == rb["loss"]


def test_logger_records_replayable_updates(fake_env):
    """W_0 + the (sample-weighted) recorded updates reproduces the final FedAvg model."""
    _, partition = fake_env
    logger = RoundLogger()
    _, final = _train(partition, callback=logger)
    h = logger.to_dict()
    assert len(h["rounds"]) == 4
    state = {k: v.clone() for k, v in h["initial_state"].items()}
    for r in h["rounds"]:
        total = sum(c["num_samples"] for c in r["clients"].values())
        for k in state:
            if state[k].is_floating_point():
                state[k] = state[k] + sum(
                    c["delta"][k] * (c["num_samples"] / total) for c in r["clients"].values()
                )
    for k in final:
        assert torch.allclose(state[k], final[k], atol=1e-5), k


def test_history_round_trips_and_reports_size(fake_env, tmp_path):
    _, partition = fake_env
    logger = RoundLogger()
    _train(partition, callback=logger)
    h = logger.to_dict()
    save_history(h, tmp_path / "log.pt")
    back = load_history(tmp_path / "log.pt")
    assert _same(h["initial_state"], back["initial_state"])
    assert history_size_bytes(back) == history_size_bytes(h) > 0


def _history(partition):
    logger = RoundLogger()
    _train(partition, callback=logger)
    return logger.to_dict()


def _run(ds, partition, history, **kw):
    args = dict(
        history=history, model_id="transformer", channels=1, num_classes=NUM_CLASSES,
        dataset=ds.get_train(), clients=partition["clients"], target_client_id=2,
        lr=0.01, batch_size=8, seed=1,
    )
    args.update(kw)
    return run_federaser(**args)


def test_federaser_never_reads_the_target_clients_data(fake_env):
    ds, partition = fake_env
    history = _history(partition)
    target = set(partition["clients"][2]["indices"])
    seen = set()

    class Spy(Dataset):
        def __len__(self):
            return len(ds.get_train())

        def __getitem__(self, i):
            seen.add(int(i))
            return ds.get_train()[i]

    _run(ds, partition, history, dataset=Spy())
    assert seen, "calibration should have read retained data"
    assert not (seen & target)


def test_federaser_output_shape_and_milestones(fake_env):
    ds, partition = fake_env
    history = _history(partition)  # 4 rounds
    out = _run(ds, partition, history, milestone_interval=2, cali_steps=3)
    assert out["rounds_replayed"] == 4
    assert out["milestone_rounds"] == [2, 4]
    assert out["retained_clients"] == [0, 1]
    # 2 milestones x 2 retained clients x 3 steps
    assert out["steps_completed"] == 12
    init = history["initial_state"]
    assert set(out["final_state_dict"]) == set(init)
    assert not _same(out["final_state_dict"], init)


def test_last_round_is_always_a_milestone(fake_env):
    ds, partition = fake_env
    history = _history(partition)
    out = _run(ds, partition, history, milestone_interval=100, cali_steps=2)
    assert out["milestone_rounds"] == [4]
    assert out["steps_completed"] == 1 * 2 * 2


def test_federaser_is_seed_deterministic(fake_env):
    ds, partition = fake_env
    history = _history(partition)
    a = _run(ds, partition, history, seed=3)
    b = _run(ds, partition, history, seed=3)
    assert _same(a["final_state_dict"], b["final_state_dict"])


def test_differs_from_original_model_and_input_history_untouched(fake_env):
    ds, partition = fake_env
    logger = RoundLogger()
    _, original = _train(partition, callback=logger)
    history = logger.to_dict()
    before = {k: v.clone() for k, v in history["initial_state"].items()}
    out = _run(ds, partition, history)
    assert not _same(out["final_state_dict"], original)
    assert _same(before, history["initial_state"])


def test_rescale_restores_historical_norm_keeps_direction():
    cali = {"w": torch.tensor([3.0, 4.0])}      # norm 5
    hist = {"w": torch.tensor([0.0, 10.0])}     # norm 10
    out = rescale_to_history(cali, hist)["w"]
    assert torch.isclose(out.norm(), torch.tensor(10.0), atol=1e-4)
    assert torch.allclose(out / out.norm(), cali["w"] / cali["w"].norm(), atol=1e-5)


def test_weighted_option_runs(fake_env):
    ds, partition = fake_env
    out = _run(ds, partition, _history(partition), weighted=True)
    assert out["rounds_replayed"] == 4


def test_bad_inputs_rejected(fake_env):
    ds, partition = fake_env
    history = _history(partition)
    with pytest.raises(ValueError):
        _run(ds, partition, history, target_client_id=99)
    with pytest.raises(ValueError):
        _run(ds, partition, history, cali_steps=0)
    with pytest.raises(ValueError):
        _run(ds, partition, history, milestone_interval=0)
    with pytest.raises(ValueError):
        _run(ds, partition, {"initial_state": history["initial_state"], "rounds": []})
    only_target = [partition["clients"][2]]
    with pytest.raises(ValueError):
        _run(ds, partition, history, clients=only_target)
