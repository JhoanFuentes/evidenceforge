# Technical review guide

EvidenceForge is a personal Applied AI engineering project developed with AI
assistance. The reviewable contribution is an empirical retrieval comparison with
preserved rankings and explicit limits on what citations establish.

[Interactive experiment](https://jhoanfuentes.github.io/evidenceforge/) ·
[Source repository](https://github.com/JhoanFuentes/evidenceforge) ·
[Current CI runs](https://github.com/JhoanFuentes/evidenceforge/actions/workflows/ci.yml)

## A 60-second review

1. Read the [measured results](../README.md#measured-result): actual local CPU
   embedding inference over 5,183 documents and 300 public SciFact queries.
2. Inspect a regression in the [case study](CASE_STUDY.md), then follow its query
   ID in the report. Aggregate improvement does not mean every query improved.
3. Verify the saved artifact with
   `python -m evidenceforge verify examples/scifact-v1/results.json`.
4. Inspect the [experiment protocol](EXPERIMENT.md) and
   [data/model provenance](DATA_AND_MODELS.md) before generalizing the result.

## Decisions and evidence

| Engineering decision | Evidence to inspect | Remaining limitation |
| --- | --- | --- |
| Keep lexical retrieval as a baseline | BM25 nDCG@10 0.6634; dense 0.7127; paired difference +0.0493 with 95% interval [+0.0128, +0.0850] | One public benchmark does not establish cross-domain effectiveness. |
| Evaluate fusion instead of assuming it helps | RRF nDCG@10 0.7021; RRF-minus-dense interval [−0.0377, +0.0164] crosses zero | RRF has the highest Recall@5, 0.7712; the preferred method depends on the objective. |
| Preserve individual outcomes | Full ranked IDs, per-query scores, wins/ties/losses; dense regressed against BM25 on 52 queries | Relevance judgments are incomplete and the public data may be present in model training. |
| Separate query cost from setup | Median query latency: 26.59 ms BM25, 50.63 ms dense, 78.00 ms RRF; corpus encoding reported separately | Local serial CPU measurements are not deployed-service SLOs. |
| Make artifacts independently checkable | Verifier recomputes metrics and intervals without model inference; hashes identify configuration and scientific modules | Integrity checks do not authenticate the machine or author that produced rankings. |
| Keep quote integrity distinct from semantic support | Offline demo rejects fabricated quotations while allowing a real quote attached to an unsupported conclusion | No answer-quality or semantic-entailment benchmark was run. |

## Questions worth discussing

- Why use exact cosine search rather than introduce ANN approximation on this corpus?
- What changes if Recall@5 is the objective instead of nDCG@10?
- How would an index mismatch or malformed provider response become observable?
- Which comparison remains meaningful after seeing this public test result, and
  which next experiment should be labeled exploratory?
- What additional evidence is required before choosing a retriever for a new domain?

The local validation record includes 105 passing tests with the dense extra.
Inspect the hosted CI run for the commit being reviewed instead of treating
that local count as proof of a remote workflow result.

## Relationship to HarnessLab

[HarnessLab](https://github.com/JhoanFuentes/harnesslab) measures runtime contracts
under scripted agent decisions: persistence, recovery, approvals and budgets.
EvidenceForge measures retrieval behavior with real local embedding inference.
Their separation is deliberate: reliable execution, useful retrieval, valid
quotations and true answers are different properties and need different evidence.
