# EvidenceForge contributor instructions

- Keep source changes small and independently reviewable. Python 3.12+.
- Core/demo/verifier must work without inference dependencies or network.
- Tests: `python -m unittest discover -s tests -v`; lint: `ruff check .`.
- Never tune on the included test results and present the new run as untouched test evidence.
- Preserve complete per-query rankings, provenance and explicit limitations for every published run.
- Keep source documents, weights, keys and local caches out of Git.
- Retrieval relevance, quote integrity, semantic support and truth are separate claims.
- Do not silently change the pinned model, query prefix, tokenizer or index format.
- No live API inference during default tests. Use mocks for adapter contract checks.
