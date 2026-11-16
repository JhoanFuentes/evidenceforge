# Experiment protocol

EvidenceForge evaluates **retrieval of relevant scientific abstracts** using BEIR SciFact. It compares a lexical baseline, a local dense retriever, and fusion. Answer generation is a separate, optional path and is not part of the retrieval score.

This document specifies how to interpret and reproduce an experiment. It contains no measured results. Use the generated run artifacts for actual scores, timings, configuration, and completeness. The [README](../README.md) is the source of truth for available commands.

## Question and systems

The experiment asks: **on a fixed public query set, does dense retrieval or rank fusion improve evidence ranking over BM25, and at what measured latency?** A negative result is useful evidence, not a reason to change the test protocol.

| System | Role | Main tradeoff |
| --- | --- | --- |
| BM25 | Lexical baseline over the same corpus. | Sensitive to tokenization and vocabulary overlap; no neural inference is needed. |
| Dense | BGE embeddings through FastEmbed, with exact vector search. | Adds model download, indexing, inference, and truncation choices. |
| RRF | Combine lexical and dense rankings by reciprocal rank. | Avoids mixing incomparable raw scores, but depends on candidate depth and the fusion constant. |

Cross-encoder reranking is an **extension**, not a result implied by this document. If added, report the candidate pool, model, per-stage timing, and a separate evaluation. Do not label a fused ranking as reranked unless a reranker actually ran.

## Freeze before test

BEIR SciFact contains 5,183 abstracts and has train/test query partitions. The 300-query BEIR test partition corresponds to the original SciFact public development set; it is not the original hidden leaderboard test. See [DATA_AND_MODELS.md](DATA_AND_MODELS.md) for provenance.

1. Use the 809 BEIR training queries for implementation checks or parameter exploration. If a development subset is needed, derive it deterministically from train and save its IDs.
2. Freeze a configuration before examining test results: tokenizer, BM25 parameters, title/body composition, embedding model and artifact revision, query prefix, truncation, normalization, retrieval depth, RRF constant, and evaluation cutoffs.
3. Keep the full corpus identical across systems. Indexing the shared corpus is expected; adding test qrels or expected answers to the retriever is leakage.
4. Evaluate only the query IDs belonging to the selected qrels partition. The shared queries file contains more queries than the 300 test queries.
5. Record any change after examining test. Such a run becomes exploratory rather than a fresh confirmatory comparison. Use a new evaluation set for another confirmatory claim.

Fixed defaults without parameter search are an acceptable first experiment. Do not describe them as optimized. Testing on a public benchmark also does not establish that the pretrained models have never encountered its contents.

## Metrics and uncertainty

For a query, a document is relevant when its qrels relevance value is positive. Each ranked document ID must occur at most once. Documents absent from qrels receive no relevance credit under this protocol; this is not proof that a human would consider them irrelevant.

| Metric | Definition used for interpretation |
| --- | --- |
| Recall@5, Recall@10 | Number of relevant documents retrieved within the cutoff divided by the query's total number of judged relevant documents. |
| MRR@10 | Reciprocal of the rank of the first relevant document if it appears within ten results; otherwise zero. |
| nDCG@10 | Discounted gain of the top ten ranking divided by its ideal ranking. Record the gain/discount definition; for binary SciFact labels, gain is one for relevant documents and zero otherwise. |

Aggregate using the arithmetic mean over evaluated queries. Report the query count and per-query values so failures and regressions can be inspected. Do not silently drop queries that retrieve nothing or encounter an error. Zero-result queries must receive zero retrieval metrics; infrastructure errors must remain visible and make completeness explicit.

Bootstrap uncertainty should resample query IDs with replacement, preserving all methods' scores for each sampled query. A **paired bootstrap of per-query differences** is the appropriate comparison between two methods; separate confidence intervals for their means are not a substitute. Save the random seed, number of resamples, interval level, and interval construction used by the implementation.

These intervals describe variation across the sampled queries. They do not capture all uncertainty from model pretraining, incomplete judgments, repeated tuning, corpus selection, or correlated claims. Avoid using a small positive difference as evidence of universal superiority. If many comparisons are explored, disclose that exploration rather than presenting one selected interval as an untouched test.

## Reproducibility record

The run and index artifacts should make the following reconstructible:

- Dataset URL, observed archive/file hashes, split, and ordered query/document IDs.
- Code revision or source hash, interpreter, dependency versions, operating system, and hardware description.
- Actual ONNX model repository/revision/file hash, tokenizer files, pooling, normalization, prefix, and truncation choices.
- BM25 and RRF configuration, candidate depth, evaluation cutoffs, tie-breaking, and bootstrap settings.
- Retrieval output IDs/scores, per-query metrics, errors, and timing scope.

Record the actual derivative model used by FastEmbed, not only its convenient public alias. A model name or mutable `main` branch is insufficient for strict reproduction. Index reuse requires matching the corpus, document order, and embedding configuration; a mismatch should cause a rebuild or explicit rejection.

## Timing and resource interpretation

Separate one-time download/model loading/index construction from query execution. Report whether query embeddings were cached or generated in the timed path. For RRF, include both retrieval branches and fusion when reporting end-to-end query latency; time only fusion when explicitly labeling it fusion overhead.

Record CPU provider, thread count, batch size, warmup procedure, and query ordering. Avoid comparing warm dense timings against cold BM25 timings without labeling the difference. Record percentiles with sample size and percentile convention. Local inference avoids a paid inference API but still uses compute, storage, and initial network downloads.

## Retrieval and answer quality are different experiments

The retrieval benchmark does not call the optional answer provider. Its labels identify evidence-bearing abstracts, including abstracts that contradict a claim. Retrieval success must not be presented as claim verification, factual answer accuracy, or medical advice.

The optional answer path can enforce quote integrity: a cited document belongs to the supplied evidence pack, and a quotation occurs in that document. That does **not** establish that the quotation supports the generated conclusion, preserves negation, reflects scientific consensus, or is sufficient to answer the question. Live answer generation requires its own executed evaluation and should not be implied by offline retrieval results.

## Next experiments — not built by this document

- Candidate-depth ablation and cross-encoder reranking with full latency accounting.
- A second dataset/domain to investigate whether observed differences transfer.
- Counterfactual grounding tests: remove supporting evidence, swap a relevant abstract for a contradictory one, and check whether answers appropriately change or abstain.
- Multi-hop retrieval with hop-level evidence traces and answer-level evaluation.
- Contextual retrieval: compare enriched document representations against original text while accounting for enrichment cost and avoiding query-label leakage.
- A human-reviewed semantic support rubric to evaluate claims and citations independently of substring matching.

Each extension needs a hypothesis, a control, a held-out evaluation plan, and actual measurements before it becomes a portfolio claim.
