"""Fail CI if synthetic offline retrieval regresses."""

from scripts.benchmark_retrieval import benchmark


def test_offline_retrieval_regression() -> None:
    report = benchmark()
    assert report["cases"] == 6
    assert report["hit_rate_at_3"] == 1.0
    assert report["mrr"] >= 0.75
    assert all(row["latency_ms"] >= 0 for row in report["rows"])
