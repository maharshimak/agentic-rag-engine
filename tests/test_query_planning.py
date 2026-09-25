import pytest

from rag_engine.planning import OpenAICompatibleQueryPlanner


def test_model_query_planner_preserves_original_and_deduplicates():
    planned = OpenAICompatibleQueryPlanner.parse(
        "Compare vector and lexical retrieval",
        '{"queries":["Compare vector and lexical retrieval","vector retrieval","lexical retrieval"]}',
        max_queries=3,
    )
    assert planned == [
        "Compare vector and lexical retrieval",
        "vector retrieval",
        "lexical retrieval",
    ]


def test_model_query_planner_rejects_non_text_queries():
    with pytest.raises(ValueError):
        OpenAICompatibleQueryPlanner.parse("query", '{"queries":[42]}')
