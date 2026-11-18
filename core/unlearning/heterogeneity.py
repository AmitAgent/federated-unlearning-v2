"""Client heterogeneity measurement (pure Python, no torch).

How different is the target client's label distribution from the rest
of the federation? We use the Jensen-Shannon divergence (base 2, so it
lies in [0, 1]) between the target client's class distribution and the
pooled class distribution of every other client.

0 = the target looks like everybody else; 1 = completely disjoint classes.

This only measures *label* heterogeneity. It says nothing about feature
shift, and it is a proxy for "how distinctive is this client's
influence", not a proven measure of it.
"""
from __future__ import annotations

import math
from typing import Sequence


def labels_of(dataset, indices: Sequence[int]) -> list[int]:
    """Class labels for `indices`, using `.targets` when the dataset has it
    (fast) and falling back to indexing the dataset (slow, but general)."""
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return [int(targets[i]) for i in indices]
    return [int(dataset[i][1]) for i in indices]


def label_distribution(labels: Sequence[int], num_classes: int) -> list[float]:
    counts = [0] * num_classes
    for y in labels:
        counts[int(y)] += 1
    total = sum(counts)
    if total == 0:
        raise ValueError("cannot build a label distribution from zero samples")
    return [c / total for c in counts]


def _kl(p: Sequence[float], q: Sequence[float]) -> float:
    return sum(pi * math.log2(pi / qi) for pi, qi in zip(p, q) if pi > 0)


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence, base 2, in [0, 1]."""
    if len(p) != len(q):
        raise ValueError("distributions must have the same length")
    m = [(a + b) / 2 for a, b in zip(p, q)]
    return max(0.0, min(1.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m)))


def client_heterogeneity_scores(
    client_labels: dict[int, Sequence[int]], num_classes: int
) -> dict[int, float]:
    """For every client: JS divergence between its label distribution and
    the pooled label distribution of all *other* clients."""
    if len(client_labels) < 2:
        raise ValueError("need at least two clients to measure heterogeneity")
    counts = {
        cid: [0] * num_classes for cid in client_labels
    }
    for cid, labels in client_labels.items():
        for y in labels:
            counts[cid][int(y)] += 1
    total = [sum(counts[cid][k] for cid in counts) for k in range(num_classes)]

    scores: dict[int, float] = {}
    for cid, own in counts.items():
        rest = [total[k] - own[k] for k in range(num_classes)]
        if sum(own) == 0 or sum(rest) == 0:
            raise ValueError(f"client {cid} or the remaining clients have no samples")
        p = [c / sum(own) for c in own]
        q = [c / sum(rest) for c in rest]
        scores[cid] = js_divergence(p, q)
    return scores


def dominant_classes(dist: Sequence[float], top: int = 3) -> list[int]:
    """The `top` classes holding the most of a distribution's mass."""
    return sorted(range(len(dist)), key=lambda k: dist[k], reverse=True)[:top]
