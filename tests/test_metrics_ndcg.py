"""Hand-computed checks for nDCG@k (linear gain, log2(rank + 1) discount).

log2(2) = 1, log2(3) = 1.5849625007, log2(4) = 2, 1 / log2(3) = 0.6309297536.
"""

import pytest

from rag_engine.evaluation import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

GRADES = {"a": 2, "b": 1, "c": 0}


def test_ideal_order_scores_one() -> None:
    assert ndcg_at_k(GRADES, ["a", "b", "c"], 3) == pytest.approx(1.0)


def test_reversed_order_hand_computed() -> None:
    # DCG = 0/1 + 1/log2(3) + 2/log2(4) = 0.6309297536 + 1.0 = 1.6309297536
    # IDCG = 2/1 + 1/log2(3) = 2.6309297536 -> 1.6309297536 / 2.6309297536
    assert ndcg_at_k(GRADES, ["c", "b", "a"], 3) == pytest.approx(0.6199062, abs=1e-7)


def test_cutoff_uses_ideal_of_same_depth() -> None:
    # k=1: DCG = 1/1, IDCG = 2/1
    assert ndcg_at_k(GRADES, ["b", "a"], 1) == pytest.approx(0.5)


def test_repeated_ids_earn_gain_once() -> None:
    # DCG = 2/1 + 0 (repeat) + 1/log2(4) = 2.5 ; IDCG = 2.6309297536
    assert ndcg_at_k(GRADES, ["a", "a", "b"], 3) == pytest.approx(0.9502344, abs=1e-7)


def test_unjudged_items_count_as_zero_and_no_positive_grades_gives_zero() -> None:
    # DCG = 0 + 2/log2(3) = 1.2618595 ; IDCG = 2 (only one positive judgment)
    assert ndcg_at_k({"a": 2}, ["x", "a"], 2) == pytest.approx(0.6309298, abs=1e-7)
    assert ndcg_at_k({"a": 0}, ["a"], 1) == 0.0
    assert ndcg_at_k({}, [], 3) == 0.0


@pytest.mark.parametrize("grades", [{"a": -1}, {"a": float("nan")}, {"a": True}])
def test_invalid_grades_rejected(grades) -> None:
    with pytest.raises(ValueError):
        ndcg_at_k(grades, ["a"], 1)


@pytest.mark.parametrize("k", [0, -1, True])
def test_invalid_k_rejected(k) -> None:
    with pytest.raises(ValueError):
        ndcg_at_k(GRADES, ["a"], k)


def test_existing_metrics_hand_computed_on_same_ranking() -> None:
    relevant = {"a", "b"}
    ranking = ["c", "b", "x", "a"]
    assert recall_at_k(relevant, ranking, 2) == pytest.approx(0.5)  # b of {a, b}
    assert precision_at_k(relevant, ranking, 4) == pytest.approx(0.5)  # b, a of 4
    assert reciprocal_rank(relevant, ranking) == pytest.approx(0.5)  # first hit at rank 2
