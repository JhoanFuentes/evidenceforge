"""Audit the bundled experiment without downloading data or model weights."""

import json
from pathlib import Path

from evidenceforge.experiment import verify_result
from evidenceforge.io import sha256_file

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "examples" / "scifact-v1" / "results.json"
result = json.loads(ARTIFACT.read_text(encoding="utf-8"))
checks = verify_result(ARTIFACT)
scientific_modules = ["dataset.py", "embeddings.py", "evaluation.py", "experiment.py", "io.py", "retrieval.py"]
for name in scientific_modules:
    actual = sha256_file(ROOT / "src" / "evidenceforge" / name)
    if actual != result["frozen_config"]["source_files"][name]:
        raise SystemExit(f"Scientific code differs from the included run: {name}")
if result["dataset"]["query_count"] != 300 or result["dataset"]["corpus_count"] != 5183:
    raise SystemExit("Unexpected dataset size")
if any(s["excluded_query_ids"] or s["missing_query_ids"] for s in result["systems"].values()):
    raise SystemExit("Included run has excluded/missing queries")
print(json.dumps({**checks, "scientific_source_hashes": "match", "complete": True}, indent=2))
