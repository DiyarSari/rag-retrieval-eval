"""Ranking metrics, matching the definitions used by BEIR / pytrec_eval.

A *run* maps each query id to its ranked list of document ids.
*qrels* map each query id to {doc id: graded relevance > 0}.
"""

from __future__ import annotations

import math
from statistics import mean

Run = dict[str, list[str]]
Qrels = dict[str, dict[str, int]]


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    dcg = sum(relevant.get(doc, 0) / math.log2(rank + 2) for rank, doc in enumerate(ranked[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def mrr_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    for rank, doc in enumerate(ranked[:k]):
        if doc in relevant:
            return 1.0 / (rank + 1)
    return 0.0


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant.keys()) / len(relevant)


def evaluate(run: Run, qrels: Qrels) -> dict[str, float]:
    """Macro-average each metric over the queries that have judgements."""
    qids = [q for q in qrels if qrels[q]]
    return {
        "nDCG@10": mean(ndcg_at_k(run.get(q, []), qrels[q], 10) for q in qids),
        "MRR@10": mean(mrr_at_k(run.get(q, []), qrels[q], 10) for q in qids),
        "Recall@10": mean(recall_at_k(run.get(q, []), qrels[q], 10) for q in qids),
        "Recall@100": mean(recall_at_k(run.get(q, []), qrels[q], 100) for q in qids),
    }
