"""Verifier contracts using synthetic artifacts, without models or downloads.

These fixtures test whether published scores can be reconstructed. They are not
retrieval benchmark results and must never be reported as model-quality evidence.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from evidenceforge.embeddings import MODEL_NAME, MODEL_REPO, MODEL_REVISION, QUERY_PREFIX
from evidenceforge.evaluation import compare_runs, evaluate_run
from evidenceforge.experiment import validate_config, verify_result
from evidenceforge.io import fingerprint


def valid_config() -> dict:
    return {
        "name": "synthetic-verifier-fixture", "split": "test", "systems": ["bm25", "dense", "rrf"],
        "bm25_k1": 1.2, "bm25_b": 0.75, "candidates": 10, "rrf_constant": 60,
        "query_prefix": QUERY_PREFIX, "model_name": MODEL_NAME, "model_repository": MODEL_REPO,
        "model_revision": MODEL_REVISION, "threads": 1, "batch_size": 1,
        "bootstrap_samples": 50, "seed": 42, "tuning": "Synthetic artifact verifier test only",
    }


def synthetic_artifact() -> dict:
    config = valid_config()
    qrels = {"q1": {"a": 1}, "q2": {"b": 1, "c": 1}, "q3": {"c": 1}}
    rankings = {
        "bm25": {"q1": ["a", "b", "c"], "q2": ["a", "b", "c"], "q3": ["b", "c", "a"]},
        "dense": {"q1": ["b", "a", "c"], "q2": ["b", "c", "a"], "q3": ["c", "a", "b"]},
        "rrf": {"q1": ["a", "b", "c"], "q2": ["b", "a", "c"], "q3": ["c", "b", "a"]},
    }
    systems = {
        name: {**evaluate_run(ranking, qrels), "rankings": ranking,
               "timing": {"search_seconds": 0.003, "p50_ms": 1.0, "p95_ms": 1.0}}
        for name, ranking in rankings.items()
    }
    comparisons = []
    for baseline, candidate in (("bm25", "dense"), ("bm25", "rrf"), ("dense", "rrf")):
        comparison = compare_runs(systems[baseline]["per_query"], systems[candidate]["per_query"],
                                  samples=config["bootstrap_samples"], seed=config["seed"])
        comparisons.append({"baseline": baseline, "candidate": candidate, **comparison})
    return {
        "schema_version": 1, "generated_at": "2026-11-16T00:00:00Z", "config": config,
        "frozen_config": {"config": deepcopy(config), "sha256": fingerprint(config),
                          "frozen_at": "2026-11-15T23:59:00Z", "source_files": {}},
        "dataset": {"name": "Synthetic verifier fixture", "split": "test", "corpus_count": 3,
                    "query_count": 3, "sha256": "a" * 64, "doc_order_sha256": fingerprint(["a", "b", "c"]),
                    "corpus_sha256": "b" * 64},
        "environment": {"python": "synthetic", "platform": "unit-test"},
        "index": {"note": "No index built for this fixture"}, "setup_seconds": 0.0,
        "qrels": qrels, "systems": systems, "comparisons": comparisons,
        "limitations": ["Synthetic verifier fixture; no model inference"],
    }


class ConfigurationTests(unittest.TestCase):
    def test_valid_configuration(self) -> None:
        validate_config(valid_config())

    def test_seed_must_be_an_explicit_integer(self) -> None:
        for value in (None, "42", 0.5, True):
            config = valid_config()
            config["seed"] = value
            with self.subTest(seed=value), self.assertRaises(ValueError):
                validate_config(config)

    def test_missing_required_values_are_rejected(self) -> None:
        for key in ("seed", "systems", "split", "candidates", "bootstrap_samples", "model_revision"):
            config = valid_config()
            del config[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_config(config)

    def test_unknown_split_or_insufficient_candidates_is_rejected(self) -> None:
        config = valid_config()
        config["split"] = "hidden-test"
        with self.assertRaises(ValueError):
            validate_config(config)
        for value in (0, 9, "100", True):
            config = valid_config()
            config["candidates"] = value
            with self.subTest(candidates=value), self.assertRaises(ValueError):
                validate_config(config)


class ArtifactVerificationTests(unittest.TestCase):
    def verify(self, artifact: dict) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            return verify_result(path)

    def test_complete_artifact_regrades_all_systems_and_comparisons(self) -> None:
        verified = self.verify(synthetic_artifact())
        self.assertTrue(verified["valid"])
        self.assertEqual(verified["systems"], 3)
        self.assertEqual(verified["queries"], 3)

    def test_empty_or_unsupported_artifact_is_rejected(self) -> None:
        for artifact in ({}, {"schema_version": 1}, {**synthetic_artifact(), "schema_version": 999}):
            with self.subTest(keys=list(artifact)), self.assertRaises((ValueError, KeyError)):
                self.verify(artifact)

    def test_config_requires_valid_values_and_frozen_content_equivalence(self) -> None:
        artifact = synthetic_artifact()
        artifact["frozen_config"]["config"]["rrf_constant"] = 10
        with self.assertRaises(ValueError):
            self.verify(artifact)
        artifact = synthetic_artifact()
        artifact["frozen_config"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.verify(artifact)
        # Rehashing an invalid protocol does not make that protocol valid.
        artifact = synthetic_artifact()
        artifact["config"]["candidates"] = 1
        artifact["frozen_config"]["config"] = deepcopy(artifact["config"])
        artifact["frozen_config"]["sha256"] = fingerprint(artifact["config"])
        with self.assertRaises(ValueError):
            self.verify(artifact)

    def test_missing_or_extra_systems_are_rejected(self) -> None:
        for change in ("empty", "missing", "extra"):
            artifact = synthetic_artifact()
            if change == "empty":
                artifact["systems"] = {}
            elif change == "missing":
                del artifact["systems"]["rrf"]
            else:
                artifact["systems"]["unregistered"] = deepcopy(artifact["systems"]["rrf"])
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                self.verify(artifact)

    def test_rankings_must_cover_exact_judged_query_set(self) -> None:
        for change in ("missing", "extra"):
            artifact = synthetic_artifact()
            if change == "missing":
                del artifact["systems"]["bm25"]["rankings"]["q3"]
            else:
                artifact["systems"]["bm25"]["rankings"]["unjudged-query"] = ["a"]
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.verify(artifact)
        artifact = synthetic_artifact()
        artifact["qrels"] = {}
        with self.assertRaises(ValueError):
            self.verify(artifact)

    def test_metric_or_ranking_tampering_is_rejected(self) -> None:
        for change in ("aggregate", "per-query", "ranking", "qrels"):
            artifact = synthetic_artifact()
            if change == "aggregate":
                artifact["systems"]["bm25"]["aggregate"]["ndcg@10"] += 0.1
            elif change == "per-query":
                artifact["systems"]["bm25"]["per_query"]["q1"]["ndcg@10"] = 0
            elif change == "ranking":
                artifact["systems"]["bm25"]["rankings"]["q1"].reverse()
            else:
                artifact["qrels"]["q1"] = {"b": 1}
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.verify(artifact)

    def test_evaluation_metadata_is_checked_not_only_aggregate_scores(self) -> None:
        tampering = {
            "evaluated_query_count": 99,
            "excluded_query_ids": ["q1"],
            "missing_query_ids": ["q2"],
            "unjudged_ranking_query_ids": ["unexpected"],
        }
        for key, value in tampering.items():
            artifact = synthetic_artifact()
            artifact["systems"]["dense"][key] = value
            with self.subTest(metadata=key), self.assertRaises(ValueError):
                self.verify(artifact)

    def test_dataset_split_and_query_count_match_the_run(self) -> None:
        for key, value in (("split", "train"), ("query_count", 100)):
            artifact = synthetic_artifact()
            artifact["dataset"][key] = value
            with self.subTest(metadata=key), self.assertRaises(ValueError):
                self.verify(artifact)

    def test_paired_comparison_values_and_full_pair_set_are_verified(self) -> None:
        for change in ("empty", "missing", "duplicate", "interval", "wins", "seed", "metric"):
            artifact = synthetic_artifact()
            if change == "empty":
                artifact["comparisons"] = []
            elif change == "missing":
                artifact["comparisons"].pop()
            elif change == "duplicate":
                artifact["comparisons"][2] = deepcopy(artifact["comparisons"][0])
            elif change == "interval":
                artifact["comparisons"][0]["ci95"] = [1, 1]
            elif change == "wins":
                artifact["comparisons"][0]["wins"] = 100
            elif change == "seed":
                artifact["comparisons"][0]["seed"] = 999
            else:
                artifact["comparisons"][0]["metric"] = "recall@10"
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.verify(artifact)

    def test_timings_are_reported_not_reconstructed_from_rankings(self) -> None:
        artifact = synthetic_artifact()
        artifact["systems"]["bm25"]["timing"] = {
            "search_seconds": 12.0, "p50_ms": 4000.0, "p95_ms": 4000.0,
        }
        self.assertTrue(self.verify(artifact)["valid"])


if __name__ == "__main__":
    unittest.main()
