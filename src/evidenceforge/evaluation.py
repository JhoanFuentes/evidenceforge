"""Retrieval metrics and a paired query-level bootstrap, using only the stdlib.

Each eligible query has equal weight. Documents absent from qrels count as
nonrelevant; this convention does not assert that incomplete judgments are truth.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any


def _cutoffs(ks: tuple[int, ...]) -> tuple[int, ...]:
    if not ks or any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError("ks must contain positive integer cutoffs")
    return tuple(sorted(set(ks)))


def _deduplicate(ranking: list[str]) -> list[str]:
    if not isinstance(ranking, list) or any(not isinstance(doc_id, str) for doc_id in ranking):
        raise ValueError("Each ranking must be a list of document ID strings")
    return list(dict.fromkeys(ranking))


def _gain(grade: int, maximum: int) -> float:
    """Exponential gain scaled by 2**maximum to avoid large-grade overflow."""
    if grade <= 0:
        return 0.0
    # Exponents below -1074 underflow in binary64 and can be clamped before
    # converting unusually large Python integers to floats.
    exponent = grade - maximum
    first = 0.0 if exponent < -1074 else 2.0 ** exponent
    offset = 0.0 if maximum > 1074 else 2.0 ** (-maximum)
    return first - offset


def evaluate_run(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, Any]:
    """Return macro aggregate and per-query retrieval metrics.

    Recall is computed at each supplied cutoff; MRR and NDCG use the largest
    cutoff. NDCG uses (2**relevance - 1) gain and log2(rank + 1) discount.
    Rankings are deduplicated before truncation. Missing ranked queries receive
    zero, never disappear. Queries with no positive judgments are explicitly
    listed as excluded because recall and NDCG have no positive denominator.
    """
    cutoffs = _cutoffs(ks)
    if not isinstance(rankings, dict) or not isinstance(qrels, dict):
        raise ValueError("rankings and qrels must be dictionaries keyed by query ID")
    normalized = {}
    for query_id, ranking in rankings.items():
        if not isinstance(query_id, str):
            raise ValueError("Query IDs must be strings")
        normalized[query_id] = _deduplicate(ranking)

    per_query: dict[str, dict[str, float]] = {}
    excluded = []
    missing = []
    largest = max(cutoffs)
    for query_id, judgments in qrels.items():
        if not isinstance(query_id, str) or not isinstance(judgments, dict):
            raise ValueError("qrels must map query ID strings to judgment dictionaries")
        if any(not isinstance(doc_id, str) or type(grade) is not int for doc_id, grade in judgments.items()):
            raise ValueError("Judgments must map document ID strings to integer relevance grades")
        relevant = {doc_id: grade for doc_id, grade in judgments.items() if grade > 0}
        if not relevant:
            excluded.append(query_id)
            continue
        if query_id not in normalized:
            missing.append(query_id)
        ranking = normalized.get(query_id, [])
        metrics = {
            f"recall@{k}": sum(doc_id in relevant for doc_id in ranking[:k]) / len(relevant)
            for k in cutoffs
        }
        top = ranking[:largest]
        metrics[f"mrr@{largest}"] = next(
            (1.0 / rank for rank, doc_id in enumerate(top, 1) if doc_id in relevant), 0.0
        )
        maximum = max(relevant.values())
        dcg = sum(
            _gain(relevant.get(doc_id, 0), maximum) / math.log2(rank + 1)
            for rank, doc_id in enumerate(top, 1)
        )
        ideal = sum(
            _gain(grade, maximum) / math.log2(rank + 1)
            for rank, grade in enumerate(sorted(relevant.values(), reverse=True)[:largest], 1)
        )
        metrics[f"ndcg@{largest}"] = dcg / ideal
        per_query[query_id] = metrics

    if not per_query:
        raise ValueError("Evaluation requires at least one query with a positive relevance judgment")
    names = next(iter(per_query.values())).keys()
    return {
        "aggregate": {name: statistics.fmean(metrics[name] for metrics in per_query.values()) for name in names},
        "per_query": per_query,
        "evaluated_query_count": len(per_query),
        "excluded_query_ids": sorted(excluded),
        "missing_query_ids": sorted(missing),
        "unjudged_ranking_query_ids": sorted(set(normalized) - set(qrels)),
    }


def _quantile(ordered: list[float], probability: float) -> float:
    """Linear interpolation between order statistics (R/NumPy type 7)."""
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def compare_runs(
    per_query_a: dict[str, dict[str, float]],
    per_query_b: dict[str, dict[str, float]],
    metric: str = "ndcg@10",
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap of mean query delta B-A with a percentile 95% interval.

    The unit resampled is the query, preserving the A/B pairing. The interval is
    conditional on this dataset and run; it does not include corpus, model, or
    relevance-judgment uncertainty. No multiple-comparison correction is applied.
    """
    if not isinstance(per_query_a, dict) or not isinstance(per_query_b, dict):
        raise ValueError("Per-query scores must be dictionaries")
    if not per_query_a or set(per_query_a) != set(per_query_b):
        raise ValueError("Paired comparison requires the same nonempty query IDs")
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    query_ids = sorted(per_query_a)
    deltas = []
    for query_id in query_ids:
        pair = []
        for run in (per_query_a, per_query_b):
            if metric not in run[query_id]:
                raise ValueError(f"Missing metric {metric!r} for query {query_id!r}")
            value = run[query_id][metric]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError("Metric values must be finite numbers")
            pair.append(float(value))
        deltas.append(pair[1] - pair[0])
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(samples))
    tolerance = 1e-12
    return {
        "metric": metric,
        "mean_delta": statistics.fmean(deltas),
        "ci95": [_quantile(means, 0.025), _quantile(means, 0.975)],
        "wins": sum(delta > tolerance for delta in deltas),
        "ties": sum(abs(delta) <= tolerance for delta in deltas),
        "losses": sum(delta < -tolerance for delta in deltas),
        "query_count": n,
        "samples": samples,
        "seed": seed,
        "method": "paired query bootstrap; percentile 95% interval; linear interpolation",
        "delta_direction": "B - A",
    }
