# Data, models, and attribution

Sources checked: **2026-11-16**. This is an inventory of source declarations and reproducibility choices, not a claim that these are the newest models or that this project is a state-of-the-art benchmark.

## SciFact through BEIR

EvidenceForge uses the BEIR conversion of SciFact, a dataset for scientific claim evidence retrieval. The original task also includes stance and rationale annotations; a retrieval qrel alone does not distinguish supporting from contradicting evidence.

| Property | Value / source |
| --- | --- |
| Official BEIR download | [UKP SciFact archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip), linked by the [BEIR repository](https://github.com/beir-cellar/beir). |
| Published archive checksum | MD5 `5f7d1de60b170fc8027bb7898e2efca1`, from the [BEIR dataset table](https://github.com/beir-cellar/beir/wiki/Datasets-available). MD5 is an upstream identity check, not a modern authenticity guarantee. |
| Corpus | 5,183 documents. The shared queries file has 1,109 entries. [BEIR dataset card](https://huggingface.co/datasets/BeIR/scifact). |
| Train | 809 queries and 919 relevance judgments. |
| Test | 300 queries and 339 relevance judgments. This is the original SciFact public dev set, repackaged as BEIR test. [ir_datasets adapter documentation](https://ir-datasets.com/beir.html#beir/scifact/test). |
| Original schema | Claims can have `SUPPORT` or `CONTRADICT` evidence annotations. [AllenAI data documentation](https://github.com/allenai/scifact/blob/master/doc/data.md). |

No official SHA-256 was identified during source review. The downloader should compute and record the SHA-256 of the bytes it actually receives. Label that value as locally observed; do not invent an upstream-published SHA-256. Check archive contents and extraction paths before unpacking.

Use train for development and keep test untouched until configuration is frozen. Do not call BEIR test the hidden SciFact leaderboard test or claim it is unseen by the embedding model. See [EXPERIMENT.md](EXPERIMENT.md).

## Licensing and distribution boundary

The [original SciFact license file](https://github.com/allenai/scifact/blob/master/LICENSE.md) distinguishes components:

- Claims and evidence annotations: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Corpus abstracts from S2ORC: [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/).
- Original SciFact code: Apache 2.0.

The Hugging Face BEIR cards label their distribution `cc-by-sa-4.0`, which differs from the original component declarations. This project does not resolve that discrepancy by relabeling the data under its own code license. Preserve the source notices and links; downloaded datasets, indexes, and model weights should stay outside the Git source distribution.

Attribution is required by the cited data licenses. CC BY asks that modifications be indicated. ODC-By addresses database rights and does not necessarily grant every right in individual contents. Before redistributing a corpus or substantial derived collection, inspect the applicable source terms rather than assuming that the repository's software license covers it. A download script is a distribution choice, not a replacement for those terms.

Suggested attribution for reports: **“Retrieval evaluation using the BEIR conversion of SciFact, originally by Wadden et al. (2020), with abstracts drawn from Semantic Scholar S2ORC. BEIR by Thakur et al. (2021). Dataset and model licenses are separate from the EvidenceForge code license.”** Include the source links alongside it. Retain document IDs in results so source records remain traceable.

## Dense model

The selected alias is [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5): an English embedding model with 384 dimensions and a 512-token input limit, declared MIT by its authors. Its size makes local experiments practical; its age and benchmark exposure mean it must not be described as a new frontier model.

FastEmbed's registry currently resolves that alias to [Qdrant/bge-small-en-v1.5-onnx-Q](https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q), a quantized ONNX derivative, also declared MIT. The [registry source](https://github.com/qdrant/fastembed/blob/main/fastembed/text/onnx_embedding.py) specifies `model_optimized.onnx`. Record the actual resolved artifact and its hash. Results from this path are not automatically identical to the original full-precision weights.

For short-query retrieval, the BGE card recommends a query instruction; passages do not need it. Fix the prefix policy before evaluation and record it. Do not silently alternate `embed` and query-specific preprocessing. Document title/body composition, truncation, pooling, and normalization; these can affect retrieval even when the model name is unchanged.

[FastEmbed](https://qdrant.github.io/fastembed/) uses ONNX Runtime. Model download needs network access initially; inference can run locally afterward. Record installed versions, CPU execution provider, threads, and cache location. “No paid inference API” does not mean zero compute cost or no initial download.

## Optional reranker research

Reranking is not promised as an implemented feature. If added, the current [FastEmbed cross-encoder registry](https://github.com/qdrant/fastembed/blob/main/fastembed/rerank/cross_encoder/onnx_text_cross_encoder.py) includes `Xenova/ms-marco-MiniLM-L-6-v2` with an Apache 2.0 declaration. Verify the resolved revision and license again before packaging or publishing results.

Do not assume that every model in the same library shares a license. The [Qdrant reranking tutorial](https://qdrant.tech/documentation/fastembed/fastembed-rerankers/) uses Jina Reranker v2 under CC-BY-NC-4.0. Model selection, redistribution, and commercial use need to respect each model's own terms.

## Primary references

- Wadden et al., **Fact or Fiction: Verifying Scientific Claims**, EMNLP 2020. [Paper](https://aclanthology.org/2020.emnlp-main.609/) · [Code/data](https://github.com/allenai/scifact).
- Thakur et al., **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models**, NeurIPS 2021 Datasets and Benchmarks. [Paper](https://openreview.net/forum?id=wCu6T5xFjeJ) · [Repository](https://github.com/beir-cellar/beir).
- Cormack, Clarke, and Büttcher, **Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods**, SIGIR 2009. [Author-hosted paper](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf) · [DOI](https://doi.org/10.1145/1571941.1572114). Its fixed `k=60` is a historical baseline, not a guarantee for this corpus.
- BAAI model card, Qdrant derivative model card, and FastEmbed implementation/documentation linked above; accessed 2026-11-16. Their mutable online contents do not replace pinned artifact identities in an experiment manifest.
