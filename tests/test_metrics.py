import math

import pytest

from rageval.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a": 2, "b": 1}, 10) == pytest.approx(1.0)


def test_ndcg_hand_computed_graded():
    # relevant: a=2 at rank 3, b=1 at rank 1
    dcg = 1 / math.log2(2) + 2 / math.log2(4)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["b", "x", "a"], {"a": 2, "b": 1}, 10) == pytest.approx(dcg / idcg)


def test_ndcg_respects_cutoff():
    assert ndcg_at_k(["x", "y", "a"], {"a": 1}, 2) == 0.0


def test_mrr_uses_first_relevant_hit():
    assert mrr_at_k(["x", "a", "b"], {"a": 1, "b": 1}, 10) == pytest.approx(0.5)
    assert mrr_at_k(["x", "y"], {"a": 1}, 10) == 0.0


def test_recall_counts_unique_relevant_docs():
    assert recall_at_k(["a", "x", "c"], {"a": 1, "b": 1, "c": 1, "d": 1}, 10) == pytest.approx(0.5)


def test_evaluate_macro_averages_and_treats_missing_runs_as_empty():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    scores = evaluate({"q1": ["a"]}, qrels)
    assert scores["MRR@10"] == pytest.approx(0.5)
    assert scores["Recall@100"] == pytest.approx(0.5)
