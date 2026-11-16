# Two queries behind the average

The measured run has 300 queries. These are deliberately selected illustrations
of the largest negative and positive RRF-versus-BM25 nDCG@10 differences, not a
representative sample or an additional independent evaluation. Full results remain
available in the [report](../examples/scifact-v1/report.html).

## Regression: query 1363

The sole judged relevant document is **8290953**. BM25 ranks it first, producing
nDCG@10 = **1.0**. Dense and RRF both miss it in their top ten, producing **0.0**.
It is absent from the dense top 100; fusion places it at rank **17**.
The lexical branch found a judged relevant document, but combining the candidate
rankings did not preserve its top-ten position. This exposes a specific failure
of unweighted rank fusion; it does not establish why the scientific document is
relevant or whether it supports the claim.

```bash
python -m evidenceforge download
python -m evidenceforge case --query-id 1363 --output runs/case-1363.json
```

Inspect the original claim and documents locally. Compare vocabulary overlap and
what the dense branch retrieved. Treat any semantic explanation as a hypothesis
to test, not as something the ranking metrics prove.

## Recovery: query 1140

The sole judged relevant document is **12009265**. BM25 ranks it ninth
(nDCG@10 = **0.3010**); dense and RRF rank it first (**1.0**). The dense retriever
surfaces this evidence earlier. Fusion preserves that improvement for this query.

```bash
python -m evidenceforge case --query-id 1140 --output runs/case-1140.json
```

These commands inspect saved rankings and the checksum-verified local corpus.
They do not re-run inference or send documents to a provider.

## Decision and next test

Across all 300 queries, dense leads nDCG@10 at **0.7127**, versus **0.6634** for
BM25 and **0.7021** for RRF. Dense has higher latency than BM25; fusion adds work.
RRF's paired difference against dense has an interval crossing zero. For an
nDCG@10-oriented prototype, start with dense and retain the baseline for monitoring
regressions. The higher RRF Recall@5 (0.7712 vs dense 0.7590) gives a concrete reason
to reconsider the choice if the product's operating objective is top-five recall.

A next experiment could vary candidate depth or introduce a reranker, selecting
parameters only on train. Once these test results have been examined, further
experiments on this same test set are exploratory. A new confirmatory claim needs
a genuinely new evaluation plan/set. Relevant evidence may support or contradict
a claim; retrieval scores do not grade generated answers.
