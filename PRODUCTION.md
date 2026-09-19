# Production engineering

The RAG engine now exposes query-level retrieval diagnostics through `rag_engine.diagnostics` rather than relying only on a final answer score.

## Metrics

`evaluate_retrieval` reports precision@k, recall@k, reciprocal rank, citation coverage and duplicate-result redundancy. These metrics make hybrid-retrieval regressions visible before they become generation failures.

## Release gates

- Evaluate representative queries with explicit relevance labels.
- Track retrieval and generation metrics separately.
- Fail releases when citation coverage or recall materially regresses.
- Preserve query IDs and corpus fingerprints with evaluation results.
- Run the existing package, lint, test, wheel and container checks before publication.

The diagnostics are deterministic and dependency-free, so they can run in CI or offline evaluation jobs.
