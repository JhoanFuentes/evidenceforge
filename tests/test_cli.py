import json
import tempfile
import unittest
from pathlib import Path

from evidenceforge.cli import demo, read_pack
from evidenceforge.io import fingerprint, write_json


class DemoTests(unittest.TestCase):
    def test_offline_boundary_demo(self):
        with tempfile.TemporaryDirectory() as directory:
            result = demo(Path(directory))
            self.assertEqual(result["top_document"], "runbook")
            self.assertTrue(result["exact_quote_valid"])
            self.assertFalse(result["fabricated_quote_valid"])
            self.assertTrue(result["unsupported_conclusion_quote_valid"])
            _, documents = read_pack(Path(directory) / "evidence.json")
            self.assertGreater(len(documents), 0)

    def test_pack_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            demo(Path(directory))
            pack = json.loads(path.read_text())
            pack["documents"][0]["text"] = "Tampered"
            write_json(path, pack)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                read_pack(path)

    def test_recomputed_fingerprint_does_not_replace_schema_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            pack = {"schema_version": 1, "query": "test", "documents": [
                {"id": "1", "title": "", "text": "x"},
                {"id": "1", "title": "", "text": "y"}]}
            pack["fingerprint"] = fingerprint(pack)
            write_json(path, pack)
            with self.assertRaisesRegex(ValueError, "Invalid evidence documents"):
                read_pack(path)


if __name__ == "__main__":
    unittest.main()
