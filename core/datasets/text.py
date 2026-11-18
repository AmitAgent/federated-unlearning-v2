"""Shared text-preprocessing constants and tokenizer (text datasets).

Pure Python -- no torch import -- so `core.federated.model` can read the
constants and manifest-only callers stay lightweight.

Tokenization is a deterministic *hashing* tokenizer: each lowercase word is
hashed with CRC32 into a fixed-size vocabulary. Compared with a learned
vocabulary this means:

* no vocabulary file has to be built, saved, or shared between federated
  clients (every client tokenizes identically by construction);
* tokenization is stable across processes and machines (Python's built-in
  ``hash()`` is salted per process, so it is deliberately NOT used);
* rare words may collide into the same bucket -- an acceptable trade-off
  for a course-project-scale model.

Token ids: 0 = padding, 1 = [CLS] (always the first token), 2..VOCAB_SIZE-1
= hashed words. Because [CLS] is always present, no sequence is ever fully
padding, which keeps attention well-defined.
"""
from __future__ import annotations

import re
import zlib
from functools import lru_cache

#: Size of the hashed vocabulary (including the pad and [CLS] ids).
VOCAB_SIZE = 30000
#: Every text is truncated / padded to exactly this many tokens.
SEQ_LEN = 64

PAD_ID = 0
CLS_ID = 1
_FIRST_WORD_ID = 2

_WORD_RE = re.compile(r"[a-z0-9']+")


@lru_cache(maxsize=200_000)
def _word_id(word: str) -> int:
    return zlib.crc32(word.encode("utf-8")) % (VOCAB_SIZE - _FIRST_WORD_ID) + _FIRST_WORD_ID


def tokenize(text: str, seq_len: int = SEQ_LEN) -> list[int]:
    """Text -> exactly ``seq_len`` token ids: [CLS], words..., then padding."""
    ids = [CLS_ID]
    for word in _WORD_RE.findall(text.lower()):
        if len(ids) >= seq_len:
            break
        ids.append(_word_id(word))
    ids.extend([PAD_ID] * (seq_len - len(ids)))
    return ids
