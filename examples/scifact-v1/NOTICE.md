# Benchmark artifact attribution

These artifacts contain generated retrieval rankings/scores and relevance judgments
from **SciFact**, transformed into the **BEIR** format. They do not contain corpus
abstracts, original query text, or model weights.

- SciFact: David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen,
  Arman Cohan, Hannaneh Hajishirzi. *Fact or Fiction: Verifying Scientific Claims*,
  EMNLP 2020. https://aclanthology.org/2020.emnlp-main.609/
- SciFact source and component-specific terms:
  https://github.com/allenai/scifact and
  https://github.com/allenai/scifact/blob/master/LICENSE.md
- Upstream claims/annotations license: CC BY 4.0,
  https://creativecommons.org/licenses/by/4.0/
- BEIR: Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna
  Gurevych. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information
  Retrieval Models*, NeurIPS 2021. https://openreview.net/forum?id=wCu6T5xFjeJ
- BEIR source: https://github.com/beir-cellar/beir

Changes: relevance TSVs were converted to a JSON mapping for recomputation;
EvidenceForge generated ranking IDs, aggregate/per-query metrics, paired bootstrap
intervals and presentation. The repo does not imply endorsement by dataset authors.
Upstream licensing differs from some redistribution metadata; see
[data and model notes](../../docs/DATA_AND_MODELS.md) for that distinction.
