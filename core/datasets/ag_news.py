"""AG News dataset loader -- 4-class news-topic text classification.

Classes: 0 World, 1 Sports, 2 Business, 3 Sci/Tech.
120,000 training and 7,600 test articles (title + description).

The CSV files are downloaded once with the standard library (no extra
dependencies), tokenized with the shared hashing tokenizer
(``core.datasets.text``), and cached as tensors under ``./data/ag_news``
so every later CLI process loads in about a second.
"""
from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

from core.datasets.base import Dataset
from core.datasets.text import SEQ_LEN, VOCAB_SIZE, tokenize

DATA_DIR = Path("./data/ag_news")

#: Mirrors tried in order. The CSVs have no header; columns are
#: (class index 1-4, title, description).
_BASE_URLS = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv",
    "https://raw.githubusercontent.com/mhjabreel/CharCNN/master/data/ag_news_csv",
)


def _download_csv(split: str, directory: Path) -> Path:
    target = directory / f"{split}.csv"
    if target.exists() and target.stat().st_size > 0:
        return target
    directory.mkdir(parents=True, exist_ok=True)
    errors = []
    for base in _BASE_URLS:
        url = f"{base}/{split}.csv"
        try:
            tmp = target.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(target)
            return target
        except Exception as exc:  # noqa: BLE001 -- try the next mirror
            errors.append(f"{url}: {exc}")
    raise RuntimeError(
        "Could not download AG News. Download train.csv and test.csv manually "
        f"into {directory.resolve()} and rerun.\n" + "\n".join(errors)
    )


def _read_csv(path: Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            labels.append(int(row[0]) - 1)  # 1..4 -> 0..3
            texts.append(f"{row[1]} {row[2]}")
    return texts, labels


class TokenizedTextDataset:
    """Map-style dataset of (token_ids[seq_len] LongTensor, label int)."""

    def __init__(self, tokens, labels):
        self.tokens = tokens  # (N, SEQ_LEN) long
        self.labels = labels  # (N,) long

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, index: int):
        return self.tokens[index], int(self.labels[index])


def load_split(split: str, directory: Path | None = None) -> TokenizedTextDataset:
    import torch

    directory = Path(directory) if directory is not None else DATA_DIR
    cache = directory / f"{split}_tokens_v{VOCAB_SIZE}_{SEQ_LEN}.pt"
    if cache.exists():
        saved = torch.load(cache)
        return TokenizedTextDataset(saved["tokens"], saved["labels"])

    texts, labels = _read_csv(_download_csv(split, directory))
    tokens = torch.tensor([tokenize(t) for t in texts], dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"tokens": tokens, "labels": labels_t}, cache)
    return TokenizedTextDataset(tokens, labels_t)


class AGNewsDataset(Dataset):
    id = "ag_news"

    def get_train(self):
        return load_split("train")

    def get_test(self):
        return load_split("test")
