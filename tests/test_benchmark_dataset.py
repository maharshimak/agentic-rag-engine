import copy
import json
from pathlib import Path

import pytest

from rag_engine.benchmark import (
    dataset_fingerprint,
    load_dataset,
    main,
    run_benchmark,
    validate_dataset,
)

DATASET_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "retrieval_v1.json"
# Pinned so judgments cannot change silently; update deliberately with a reviewed diff.
DATASET_SHA256 = "5688b37a2f38885f64e88610b8a4b0b4886c0081c7290e77591845be0ac4ee4f"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATASET_PATH)


@pytest.fixture(scope="module")
def report(dataset):
    return run_benchmark(dataset, k=5)


def test_dataset_fingerprint_is_pinned_and_formatting_independent(dataset) -> None:
    assert dataset_fingerprint(dataset) == DATASET_SHA256
    reformatted = json.loads(json.dumps(dataset, indent=None))
    assert dataset_fingerprint(reformatted) == DATASET_SHA256


def test_report_links_dataset_and_index(report) -> None:
    assert report["dataset_sha256"] == DATASET_SHA256
    assert len(report["index_fingerprint"]) == 64
    assert report["engine"]["embedding"]["kind"] == "sha256_token_hash"
    assert report["aggregate"]["answerable"]["queries"] == 12
    assert report["aggregate"]["unanswerable"]["queries"] == 2


def test_benchmark_invariants_hold(report) -> None:
    invariants = report["invariants"]
    assert invariants["rankings_stable_across_insertion_order"] is True
    assert invariants["index_fingerprint_stable_across_insertion_order"] is True
    assert invariants["deleted_document_leaks"] == []
    assert invariants["citations_outside_retrieved_set"] == []
    assert invariants["unsupported_answer_citations"] == 0


def test_metrics_are_bounded_and_repeatable(dataset, report) -> None:
    for row in report["queries"]:
        for name in ("recall_at_k", "precision_at_k", "reciprocal_rank", "ndcg_at_k"):
            if name in row:
                assert 0.0 <= row[name] <= 1.0
    again = run_benchmark(dataset, k=5)
    assert again["queries"] == report["queries"]
    assert again["aggregate"] == report["aggregate"]


def test_cli_writes_report_and_exits_zero(tmp_path) -> None:
    output = tmp_path / "report.json"
    assert main([str(DATASET_PATH), "--k", "3", "--output", str(output)]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["k"] == 3
    assert written["dataset_sha256"] == DATASET_SHA256


@pytest.mark.parametrize(
    "mutate, error",
    [
        (lambda d: d["queries"][0]["relevance"].update({"missing": 1}), "unknown document"),
        (lambda d: d["queries"][0]["relevance"].update({"backup-policy": -1}), "invalid grade"),
        (lambda d: d["documents"].append(dict(d["documents"][0])), "unique"),
        (lambda d: d["queries"].append(dict(d["queries"][0])), "query ids"),
        (lambda d: d["queries"][0]["relevance"].update({"retired-runbook": 2}), "deleted"),
        (lambda d: d.update({"delete_after_ingest": ["nope"]}), "delete_after_ingest"),
    ],
)
def test_invalid_datasets_rejected(dataset, mutate, error) -> None:
    broken = copy.deepcopy(dataset)
    mutate(broken)
    with pytest.raises(ValueError, match=error):
        validate_dataset(broken)
