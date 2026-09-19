# Security

Offline embeddings use token hashing, not a trained semantic model. The reranker and query planner are deterministic heuristics. Indexes live in memory and are rebuilt during ingestion. Input is already-extracted text; PDF/OCR/HTML ingestion is not implemented. Citations identify supplied context but do not establish answer faithfulness. The small synthetic retrieval test is a regression fixture, not a general retrieval benchmark. Remote adapters require an independently hosted compatible service and are not validated against live providers by offline CI.

Use synthetic or explicitly authorized public data. Do not commit API keys, databases, model credentials, patient records or private employer material. Remote model adapters transmit supplied text to the configured endpoint; choose the provider deliberately.

For a suspected vulnerability, use GitHub private vulnerability reporting if enabled. Otherwise contact the maintainer privately through the [portfolio](https://maharshipatel-portfolio.vercel.app/). Do not put secrets or exploit payloads containing private data in a public issue. No response SLA is promised.
