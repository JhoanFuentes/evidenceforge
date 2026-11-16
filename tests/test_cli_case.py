"""Inspect saved rankings against a matching local corpus without inference."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evidenceforge.cli import parser, run
from evidenceforge.dataset import Dataset, Document
from evidenceforge.io import write_json


class CaseInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / "results.json"
        self.output = self.root / "case.json"
        self.dataset = Dataset(
            [Document("a", "Trial A", "LOCAL-ONLY abstract A"), Document("b", "Trial B", "LOCAL-ONLY abstract B")],
            {"q1": "Original query text"}, {"q1": {"a": 1}},
            {"split": "test", "corpus_sha256": "matching-corpus-hash"},
        )
        self.artifact = {
            "dataset": {"split": "test", "corpus_sha256": "matching-corpus-hash"},
            "qrels": {"q1": {"a": 1}},
            "systems": {
                "bm25": {"rankings": {"q1": ["a", "b"]}, "per_query": {"q1": {"ndcg@10": 1.0}}},
                "dense": {"rankings": {"q1": ["b", "a"]}, "per_query": {"q1": {"ndcg@10": 0.63}}},
            },
        }
        self.save_artifact()
        self.args = parser().parse_args(["case", "--query-id", "q1", "--results", str(self.results),
                                        "--data", str(self.root / "data"), "--output", str(self.output)])

    def save_artifact(self):
        write_json(self.results, self.artifact)

    def inspect(self):
        with patch("evidenceforge.experiment.verify_result", return_value={"valid": True}) as verify:
            with patch("evidenceforge.cli.load_dataset", return_value=self.dataset) as load:
                result = run(self.args)
                verify.assert_called_once_with(self.results)
                load.assert_called_once_with(self.args.data, "test")
        return result

    def test_saved_order_original_text_and_judgment_flags_are_preserved(self):
        with patch("evidenceforge.embeddings.Encoder", side_effect=AssertionError("No model allowed")), \
                patch("evidenceforge.providers.OpenAIResponses", side_effect=AssertionError("No API allowed")):
            summary = self.inspect()
        saved = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(saved["query"], "Original query text")
        self.assertEqual([d["document_id"] for d in saved["systems"]["dense"]["documents"]], ["b", "a"])
        self.assertFalse(saved["systems"]["dense"]["documents"][0]["is_judged_relevant"])
        self.assertTrue(saved["systems"]["dense"]["documents"][1]["is_judged_relevant"])
        self.assertEqual(saved["systems"]["bm25"]["documents"][0]["text"], "LOCAL-ONLY abstract A")
        self.assertEqual(saved["systems"]["dense"]["scores"], {"ndcg@10": 0.63})
        self.assertNotIn("LOCAL-ONLY", json.dumps(summary))
        self.assertNotIn("Original query text", json.dumps(summary))
        self.assertIn("Judgments can be incomplete", saved["notice"])

    def test_top_k_is_applied_without_reranking(self):
        self.args.top_k = 1
        self.inspect()
        saved = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(len(saved["systems"]["dense"]["documents"]), 1)
        self.assertEqual(saved["systems"]["dense"]["documents"][0]["document_id"], "b")

    def test_bad_top_k_fails_before_reading_dataset(self):
        for value in (0, 101, True, 1.5):
            self.args.top_k = value
            with self.subTest(value=value), patch("evidenceforge.cli.load_dataset") as load:
                with self.assertRaises(ValueError):
                    run(self.args)
                load.assert_not_called()

    def test_invalid_result_stops_before_local_dataset_is_loaded(self):
        with patch("evidenceforge.experiment.verify_result", side_effect=ValueError("Metric mismatch")), \
                patch("evidenceforge.cli.load_dataset") as load:
            with self.assertRaises(ValueError):
                run(self.args)
            load.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_corpus_mismatch_and_unknown_query_fail_without_output(self):
        self.dataset.manifest["corpus_sha256"] = "wrong-corpus"
        with self.assertRaisesRegex(ValueError, "corpus differs"):
            self.inspect()
        self.dataset.manifest["corpus_sha256"] = "matching-corpus-hash"
        self.args.query_id = "missing"
        with self.assertRaisesRegex(ValueError, "Unknown query"):
            self.inspect()
        self.assertFalse(self.output.exists())

    def test_unknown_document_ids_fail_even_below_requested_top_k(self):
        self.args.top_k = 1
        self.artifact["systems"]["dense"]["rankings"]["q1"].append("missing-document")
        self.save_artifact()
        with self.assertRaisesRegex(ValueError, "unknown document ID"):
            self.inspect()
        self.assertFalse(self.output.exists())

    def test_missing_system_query_or_changed_judgments_fail(self):
        original = deepcopy(self.artifact)
        del self.artifact["systems"]["dense"]["per_query"]["q1"]
        self.save_artifact()
        with self.assertRaisesRegex(ValueError, "missing the requested query"):
            self.inspect()
        self.artifact = original
        self.save_artifact()
        self.dataset.qrels["q1"] = {"b": 1}
        with self.assertRaisesRegex(ValueError, "judgments differ"):
            self.inspect()


if __name__ == "__main__":
    unittest.main()
