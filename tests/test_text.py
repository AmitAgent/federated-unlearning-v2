"""Text-data support: tokenizer, AG News loader, text transformer, and the
model/dataset compatibility guard. Everything here runs offline -- the AG
News loader is exercised against a tiny local CSV, never the network.
"""
import json
import os
import subprocess
import sys

import pytest
import torch
from torch.utils.data import TensorDataset

from core.datasets import ag_news
from core.datasets.base import load_manifest
from core.datasets.text import CLS_ID, PAD_ID, SEQ_LEN, VOCAB_SIZE, tokenize
from core.evaluation.accuracy import compute_accuracy
from core.federated.client import local_train
from core.federated.fedavg import federated_average
from core.federated.model import (
    DEFAULT_LR,
    MODEL_IDS,
    build_model,
    check_model_dataset_compatible,
    make_optimizer,
)
from core.federated.server import fedavg_train
from core.unlearning.gradient_ascent import run_gradient_ascent


# ---------------------------------------------------------------- tokenizer
def test_tokenize_shape_cls_first_and_padding():
    ids = tokenize("Wall St. Bears Claw Back Into the Black")
    assert len(ids) == SEQ_LEN
    assert ids[0] == CLS_ID
    n_real = sum(1 for i in ids if i != PAD_ID)
    assert n_real == 1 + 8  # [CLS] + 8 words (the '.' in "St." is dropped)
    assert all(i == PAD_ID for i in ids[n_real:])


def test_tokenize_ids_in_range_and_case_insensitive():
    ids = tokenize("Hello WORLD hello")
    assert all(0 <= i < VOCAB_SIZE for i in ids)
    assert ids[1] == ids[3]  # "Hello" and "hello" share an id
    assert tokenize("Hello World") == tokenize("hello world")


def test_tokenize_truncates_and_handles_empty():
    long_ids = tokenize("word " * 500)
    assert len(long_ids) == SEQ_LEN and long_ids[0] == CLS_ID
    assert all(i != PAD_ID for i in long_ids)
    empty = tokenize("")
    assert empty[0] == CLS_ID and all(i == PAD_ID for i in empty[1:])


def test_tokenize_is_stable_across_processes():
    """Checkpoints are reloaded in other processes, so ids must not depend
    on Python's per-process hash salt."""
    code = "from core.datasets.text import tokenize; print(tokenize('federated unlearning of news'))"
    outputs = set()
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        outputs.add(out.stdout.strip())
    assert len(outputs) == 1


# ------------------------------------------------------------------ manifest
def test_manifest_text_entry_matches_tokenizer_constants():
    meta = load_manifest()["ag_news"]
    assert meta["modality"] == "text"
    assert meta["seq_len"] == SEQ_LEN
    assert meta["vocab_size"] == VOCAB_SIZE
    assert meta["num_classes"] == 4


def test_image_datasets_are_marked_image():
    manifest = load_manifest()
    for dataset_id in ("mnist", "cifar10", "cifar100"):
        assert manifest[dataset_id]["modality"] == "image"


# -------------------------------------------------------------------- loader
def _write_csv(directory, split, rows):
    directory.mkdir(parents=True, exist_ok=True)
    import csv

    with open(directory / f"{split}.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def test_load_split_reads_csv_and_caches(tmp_path):
    rows = [
        ["3", "Stocks rally", "Markets rose sharply today."],
        ["2", "Cup final", "The home team won in extra time."],
        ["1", "Talks resume", "Diplomats met for a third day."],
        ["4", "New chip", "Researchers unveiled a faster processor."],
    ]
    _write_csv(tmp_path, "train", rows)

    ds = ag_news.load_split("train", directory=tmp_path)
    assert len(ds) == 4
    tokens, label = ds[0]
    assert tokens.shape == (SEQ_LEN,) and tokens.dtype == torch.long
    assert tokens[0].item() == CLS_ID
    assert [int(ds[i][1]) for i in range(4)] == [2, 1, 0, 3]  # 1..4 -> 0..3

    cache_files = list(tmp_path.glob("train_tokens_*.pt"))
    assert len(cache_files) == 1
    (tmp_path / "train.csv").unlink()  # second load must come from the cache alone
    again = ag_news.load_split("train", directory=tmp_path)
    assert torch.equal(again.tokens, ds.tokens) and torch.equal(again.labels, ds.labels)


# --------------------------------------------------------------------- model
def test_text_transformer_forward_shape_and_finite_on_cls_only_row():
    model = build_model("text_transformer", None, 4)
    batch = torch.zeros(3, SEQ_LEN, dtype=torch.long)
    batch[:, 0] = CLS_ID  # empty text: only [CLS], rest padding
    batch[1, 1:6] = torch.tensor([5, 77, 901, 15000, 29999])
    model.eval()
    out = model(batch)
    assert out.shape == (3, 4)
    assert torch.isfinite(out).all()


def test_text_transformer_ignores_channels_argument():
    a = build_model("text_transformer", None, 4)
    b = build_model("text_transformer", 3, 4)
    assert set(a.state_dict()) == set(b.state_dict())


def test_padding_positions_do_not_affect_output():
    """Padding is masked, so the output depends only on the real tokens."""
    torch.manual_seed(0)
    model = build_model("text_transformer", None, 4).eval()
    x = torch.zeros(1, SEQ_LEN, dtype=torch.long)
    x[0, :5] = torch.tensor([CLS_ID, 11, 22, 33, 44])
    with torch.no_grad():
        full = model(x)
        # Same real tokens, but shift padding into a shorter-than-seq_len input.
        short = model(x[:, :5])
    assert torch.allclose(full, short, atol=1e-5)


def test_model_ids_cover_both_models():
    assert set(MODEL_IDS) == {"transformer", "text_transformer"}
    with pytest.raises(ValueError):
        build_model("cnn", 1, 10)


# ----------------------------------------------------------- compat + optim
def test_compatibility_check():
    manifest = load_manifest()
    check_model_dataset_compatible("text_transformer", manifest["ag_news"])
    check_model_dataset_compatible("transformer", manifest["mnist"])
    check_model_dataset_compatible("transformer", {"id": "legacy"})  # no modality -> image
    with pytest.raises(ValueError, match="text data"):
        check_model_dataset_compatible("transformer", manifest["ag_news"])
    with pytest.raises(ValueError, match="image data"):
        check_model_dataset_compatible("text_transformer", manifest["mnist"])


def test_fedavg_train_rejects_mismatched_model_before_loading_data():
    partition = {"dataset": "ag_news", "clients": []}
    with pytest.raises(ValueError, match="text data"):
        fedavg_train("e", partition, "transformer", rounds=1, local_epochs=1, seed=0)


def test_make_optimizer_choice_and_default_lrs():
    params = list(build_model("text_transformer", None, 4).parameters())
    assert isinstance(make_optimizer("text_transformer", params, 0.003), torch.optim.AdamW)
    assert isinstance(make_optimizer("transformer", params, 0.01), torch.optim.SGD)
    assert DEFAULT_LR == {"transformer": 0.01, "text_transformer": 0.003}


# ------------------------------------------------------ pipeline smoke tests
def _token_dataset(n=96, num_classes=4, seed=0):
    """Learnable toy task: the class is encoded by one marker token id."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, num_classes, (n,), generator=g)
    tokens = torch.zeros(n, SEQ_LEN, dtype=torch.long)
    tokens[:, 0] = CLS_ID
    tokens[:, 1:9] = torch.randint(100, 5000, (n, 8), generator=g)
    tokens[:, 4] = 6000 + labels  # marker token
    return TensorDataset(tokens, labels)


def test_local_train_learns_with_text_transformer():
    torch.manual_seed(0)
    dataset = _token_dataset()
    indices = list(range(len(dataset)))
    model = build_model("text_transformer", None, 4)
    result = local_train(
        model.state_dict(), "text_transformer", None, 4, dataset, indices,
        epochs=15, lr=0.003, batch_size=16,
    )
    trained = build_model("text_transformer", None, 4)
    trained.load_state_dict(result["state_dict"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=64)
    assert compute_accuracy(trained, loader, "cpu") > 0.9


def test_fedavg_over_text_models():
    torch.manual_seed(0)
    dataset = _token_dataset()
    states, sizes = [], []
    for part in (list(range(0, 48)), list(range(48, 96))):
        model = build_model("text_transformer", None, 4)
        r = local_train(model.state_dict(), "text_transformer", None, 4, dataset, part, 1, lr=0.003, batch_size=16)
        states.append(r["state_dict"])
        sizes.append(r["num_samples"])
    merged = federated_average(states, sizes)
    assert set(merged) == set(states[0])
    build_model("text_transformer", None, 4).load_state_dict(merged)


def test_gradient_ascent_raises_loss_on_text_model():
    torch.manual_seed(0)
    dataset = _token_dataset(n=64)
    model = build_model("text_transformer", None, 4)
    trained = local_train(
        model.state_dict(), "text_transformer", None, 4, dataset, list(range(64)),
        epochs=10, lr=0.003, batch_size=16,
    )
    result = run_gradient_ascent(
        global_state=trained["state_dict"],
        model_id="text_transformer",
        channels=None,
        num_classes=4,
        dataset=dataset,
        target_indices=list(range(32)),
        steps=15,
        lr=0.05,
        batch_size=8,
        plateau_patience=15,
    )
    assert result["final_loss"] > result["initial_loss"]
