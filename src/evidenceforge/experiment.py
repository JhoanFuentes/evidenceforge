"""Frozen retrieval protocol, full rankings and paired query-level comparisons."""

import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .dataset import Dataset
from .embeddings import Encoder, MODEL_NAME, MODEL_REPO, MODEL_REVISION, QUERY_PREFIX, corpus_vectors
from .evaluation import compare_runs, evaluate_run
from .io import fingerprint, sha256_file, write_json
from .retrieval import BM25, TOKENIZER_VERSION, ExactDense, reciprocal_rank_fusion


def source_manifest() -> dict:
    return {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def validate_config(config: dict) -> None:
    if config.get("systems") != ["bm25", "dense", "rrf"]:
        raise ValueError("Protocol v1 compares bm25, dense and rrf in that order")
    if config.get("split") not in {"train", "test"}:
        raise ValueError("Invalid split")
    if config.get("model_name") != MODEL_NAME or config.get("model_repository") != MODEL_REPO:
        raise ValueError("Unsupported model")
    if config.get("model_revision") != MODEL_REVISION or config.get("query_prefix") != QUERY_PREFIX:
        raise ValueError("Model revision/query prefix does not match this implementation")
    for name in ("candidates", "rrf_constant", "threads", "batch_size", "bootstrap_samples"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["candidates"] < 10:
        raise ValueError("At least 10 candidates required for metrics at 10")
    if type(config.get("seed")) is not int:
        raise ValueError("seed must be an integer")
    if not 0 < config.get("bm25_k1", 0) <= 100 or not 0 <= config.get("bm25_b", -1) <= 1:
        raise ValueError("Invalid BM25 parameters")


def _timing(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"search_seconds": sum(values) / 1000, "p50_ms": statistics.median(values),
            "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "definition": "warm, serial query encoding + retrieval; excludes dataset/index/model load"}


def benchmark(dataset: Dataset, config: dict, model_dir: Path, index: Path, output: Path) -> dict:
    validate_config(config)
    if dataset.manifest["split"] != config["split"]:
        raise ValueError("Dataset split differs from frozen configuration")
    if not dataset.queries:
        raise ValueError("No queries to evaluate")
    output.mkdir(parents=True, exist_ok=True)
    frozen = {"config": config, "sha256": fingerprint(config),
              "frozen_at": datetime.now(timezone.utc).isoformat(), "source_files": source_manifest()}
    freeze_path = output / "frozen-config.json"
    if freeze_path.exists():
        raise ValueError("Output already contains a frozen run. Use a new --output directory.")
    write_json(freeze_path, frozen)  # BEFORE loading test labels into metrics or running inference.
    started = time.perf_counter()
    encoder = Encoder(model_dir, threads=config["threads"], batch_size=config["batch_size"])
    vectors, index_manifest = corpus_vectors(dataset.documents, encoder, index)
    dense = ExactDense([d.id for d in dataset.documents], vectors)
    sparse = BM25(dataset.documents, config["bm25_k1"], config["bm25_b"])
    setup_seconds = time.perf_counter() - started
    # One fixed warmup without a benchmark query or relevance judgment.
    dense.search(encoder.encode(["scientific evidence"], query=True)[0])
    sparse.search("scientific evidence")
    rankings = {name: {} for name in config["systems"]}
    timings = {name: [] for name in config["systems"]}
    for i, (qid, query) in enumerate(dataset.queries.items(), 1):
        begin = time.perf_counter()
        sparse_hits = sparse.search(query, config["candidates"])
        sparse_ms = (time.perf_counter() - begin) * 1000
        begin = time.perf_counter()
        dense_hits = dense.search(encoder.encode([query], query=True)[0], config["candidates"])
        dense_ms = (time.perf_counter() - begin) * 1000
        begin = time.perf_counter()
        fused = reciprocal_rank_fusion([sparse_hits, dense_hits], config["candidates"],
                                       config["rrf_constant"])
        fusion_ms = (time.perf_counter() - begin) * 1000
        for name, hits, duration in (("bm25", sparse_hits, sparse_ms),
                                     ("dense", dense_hits, dense_ms),
                                     ("rrf", fused, sparse_ms + dense_ms + fusion_ms)):
            rankings[name][qid] = [hit.document_id for hit in hits]
            timings[name].append(duration)
        if i % 50 == 0:
            print(f"Evaluated {i}/{len(dataset.queries)} queries", file=sys.stderr, flush=True)
    systems = {}
    for name, ranking in rankings.items():
        systems[name] = {**evaluate_run(ranking, dataset.qrels),
                         "timing": _timing(timings[name]), "rankings": ranking}
    comparisons = []
    for baseline, candidate in (("bm25", "dense"), ("bm25", "rrf"), ("dense", "rrf")):
        delta = compare_runs(systems[baseline]["per_query"], systems[candidate]["per_query"],
                             metric="ndcg@10", samples=config["bootstrap_samples"],
                             seed=config["seed"])
        comparisons.append({"baseline": baseline, "candidate": candidate, **delta})
    result = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset.manifest, "config": config, "frozen_config": frozen,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                        "encoder": encoder.identity()},
        "index": index_manifest, "setup_seconds": setup_seconds,
        "tokenizer": TOKENIZER_VERSION,
        "systems": systems, "comparisons": comparisons,
        "qrels": dataset.qrels,
        "limitations": ["Retrieval relevance only; answer generation was not evaluated.",
                        "Public benchmark; pretraining contamination is unknown.",
                        "BEIR test is original SciFact public dev; it is not the hidden test.",
                        "RRF latency is the sum of measured serial component work, not a separate request.",
                        "BM25 has no stemming/stopwords; results are this implementation, not official BEIR baselines."],
    }
    # Query text and abstracts are intentionally excluded from portable benchmark artifacts.
    write_json(output / "results.json", result)
    return result


def verify_result(path: Path) -> dict:
    """Recompute all published scores and intervals from committed rankings + qrels."""
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema_version") != 1:
        raise ValueError("Unsupported results schema")
    validate_config(result["config"])
    if result["frozen_config"]["config"] != result["config"]:
        raise ValueError("Frozen configuration differs from results")
    if result["frozen_config"]["sha256"] != fingerprint(result["config"]):
        raise ValueError("Config fingerprint mismatch")
    if set(result["systems"]) != set(result["config"]["systems"]) or not result["qrels"]:
        raise ValueError("Missing systems or relevance judgments")
    if (result["dataset"]["split"] != result["config"]["split"]
            or result["dataset"]["query_count"] != len(result["qrels"])):
        raise ValueError("Dataset metadata does not match the evaluated split and query count")
    for name, system in result["systems"].items():
        if set(system["rankings"]) != set(result["qrels"]):
            raise ValueError("Rankings must cover exactly the evaluated query IDs")
        checked = evaluate_run(system["rankings"], result["qrels"])
        if any(system.get(key) != value for key, value in checked.items()):
            raise ValueError(f"Metrics do not match rankings: {name}")
    expected_pairs = {("bm25", "dense"), ("bm25", "rrf"), ("dense", "rrf")}
    pairs = [(c["baseline"], c["candidate"]) for c in result["comparisons"]]
    if len(pairs) != 3 or set(pairs) != expected_pairs:
        raise ValueError("Missing or duplicate system comparisons")
    for comparison in result["comparisons"]:
        if comparison["metric"] != "ndcg@10":
            raise ValueError("Unexpected comparison metric")
        checked = compare_runs(result["systems"][comparison["baseline"]]["per_query"],
                               result["systems"][comparison["candidate"]]["per_query"],
                               metric=comparison["metric"],
                               samples=result["config"]["bootstrap_samples"], seed=result["config"]["seed"])
        if any(comparison.get(key) != value for key, value in checked.items()):
            raise ValueError("Comparison cannot be reproduced")
    return {"valid": True, "systems": len(result["systems"]),
            "queries": len(result["qrels"]), "checks": "metrics, paired bootstrap, frozen config"}
