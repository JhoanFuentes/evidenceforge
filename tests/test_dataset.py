"""Local BEIR-format fixtures; downloading SciFact is never part of unit tests."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from evidenceforge.dataset import DATA_FILES, load_dataset, prepare_scifact
from evidenceforge.io import sha256_file, write_json


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "dataset"
        self.path.mkdir()
        self.rows = [{"_id": "b", "title": "Beta", "text": "Second document"},
                     {"_id": "a", "title": "Alpha", "text": "First document"}]
        self.queries = [{"_id": "q1", "text": "Test query"}, {"_id": "q2", "text": "Train query"}]
        self.write_fixture()

    def write_fixture(self, qrels=None):
        (self.path / "qrels").mkdir(exist_ok=True)
        for filename, rows in (("corpus.jsonl", self.rows), ("queries.jsonl", self.queries)):
            (self.path / filename).write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        header = "query-id\tcorpus-id\tscore\n"
        (self.path / "qrels" / "test.tsv").write_text(header + (qrels or "q1\ta\t1\n"), encoding="utf-8")
        (self.path / "qrels" / "train.tsv").write_text(header + "q2\tb\t2\n", encoding="utf-8")
        write_json(self.path / "manifest.json", {"name": "tiny-fixture", "files": {
            name: sha256_file(self.path / name) for name in DATA_FILES}})

    def test_load_sorts_documents_and_limits_queries_to_selected_split(self):
        test = load_dataset(self.path)
        train = load_dataset(self.path, "train")
        self.assertEqual([document.id for document in test.documents], ["a", "b"])
        self.assertEqual(test.queries, {"q1": "Test query"})
        self.assertEqual(train.queries, {"q2": "Train query"})
        self.assertEqual(train.qrels, {"q2": {"b": 2}})
        self.assertEqual(test.manifest["corpus_count"], 2)
        self.assertEqual(test.manifest["doc_order_sha256"], train.manifest["doc_order_sha256"])

    def test_file_changes_are_detected_before_dataset_use(self):
        with (self.path / "corpus.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Dataset file changed"):
            load_dataset(self.path)

    def test_duplicate_document_and_query_ids_are_rejected(self):
        self.rows.append(dict(self.rows[0]))
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate document"):
            load_dataset(self.path)
        self.rows.pop()
        self.queries.append(dict(self.queries[0]))
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate query"):
            load_dataset(self.path)

    def test_duplicate_and_unknown_qrels_are_rejected(self):
        for qrels, reason in (("q1\ta\t1\nq1\ta\t2\n", "Duplicate relevance"),
                              ("missing\ta\t1\n", "unknown query or document"),
                              ("q1\tmissing\t1\n", "unknown query or document")):
            self.write_fixture(qrels)
            with self.subTest(qrels=qrels), self.assertRaisesRegex(ValueError, reason):
                load_dataset(self.path)

    def test_invalid_split_fails_explicitly(self):
        with self.assertRaises(ValueError):
            load_dataset(self.path, "validation")

    def test_wrong_archive_checksum_is_rejected_without_extraction_or_network(self):
        archive = Path(self.temp.name) / "bad.zip"
        archive.write_bytes(b"not the pinned dataset")
        target = Path(self.temp.name) / "bad-output"
        with patch("evidenceforge.dataset.download") as network:
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                prepare_scifact(target, archive)
            network.assert_not_called()
        self.assertFalse((target / "corpus.jsonl").exists())

    def test_prepare_extracts_only_allowlisted_members(self):
        archive = Path(self.temp.name) / "fixture.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for name in DATA_FILES:
                bundle.writestr("scifact/" + name, (self.path / name).read_bytes())
            bundle.writestr("../outside.txt", "untrusted archive member")
            bundle.writestr("scifact/unrequested.txt", "not a dataset input")
        target = Path(self.temp.name) / "prepared"
        with patch("evidenceforge.dataset.DATA_SHA256", sha256_file(archive)), patch("evidenceforge.dataset.download") as network:
            manifest = prepare_scifact(target, archive)
            network.assert_not_called()
        self.assertEqual(set(manifest["files"]), set(DATA_FILES))
        self.assertFalse((Path(self.temp.name) / "outside.txt").exists())
        self.assertFalse((target / "unrequested.txt").exists())
        self.assertEqual(load_dataset(target).queries, {"q1": "Test query"})


if __name__ == "__main__":
    unittest.main()
