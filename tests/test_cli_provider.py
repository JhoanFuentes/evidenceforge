"""The CLI preserves safe accounting metadata on both successful and failed API calls."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evidenceforge.cli import build_pack, main, run
from evidenceforge.dataset import Document
from evidenceforge.evidence import Answer, Citation, validate_answer
from evidenceforge.io import write_json
from evidenceforge.providers import ProviderError
from evidenceforge.retrieval import Hit


class CLIProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pack_path = self.root / "evidence.json"
        self.output = self.root / "answer.json"
        self.pack = build_pack("How many adults?", [Document("trial", "Study", "The trial enrolled 48 adults.")],
                               [Hit("trial", 1.0)], "bm25")
        write_json(self.pack_path, self.pack)
        self.args = SimpleNamespace(command="answer", pack=self.pack_path, output=self.output,
                                    model="test-model-2026-11-01", max_output_tokens=1200)
        self.environment = patch.dict(os.environ, {"OPENAI_API_KEY": "test-private-key-not-for-output"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_success_manifest_is_completed_and_retains_usage(self):
        def generate(provider, query, documents):
            self.assertEqual(query, "How many adults?")
            answer = Answer("answer", "The trial enrolled 48 adults.",
                            [Citation("trial", "The trial enrolled 48 adults.")])
            provider.last_usage = {"input_tokens": 100, "output_tokens": 25, "total_tokens": 125}
            provider.response_id = "resp_success"
            provider.last_validation = validate_answer(answer, documents)
            return answer

        with patch("evidenceforge.providers.OpenAIResponses.generate", autospec=True, side_effect=generate):
            returned = run(self.args)
        saved = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(saved, returned)
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["usage"]["total_tokens"], 125)
        self.assertEqual(saved["response_id"], "resp_success")
        self.assertEqual(saved["evidence_fingerprint"], self.pack["fingerprint"])
        self.assertTrue(saved["validation"]["valid"])
        self.assertNotIn("error_code", saved)
        self.assertNotIn("test-private-key", self.output.read_text(encoding="utf-8"))

    def test_incomplete_response_preserves_accounting_and_reraises(self):
        def generate(provider, query, documents):
            provider.last_usage = {"input_tokens": 100, "output_tokens": 1200, "total_tokens": 1300}
            provider.response_id = "resp_incomplete"
            error = ProviderError("response_incomplete")
            error.raw_body = "untrusted-server-body-not-for-output"
            raise error

        with patch("evidenceforge.providers.OpenAIResponses.generate", autospec=True, side_effect=generate):
            with self.assertRaises(ProviderError):
                run(self.args)
        text = self.output.read_text(encoding="utf-8")
        saved = json.loads(text)
        self.assertEqual(saved["status"], "error")
        self.assertEqual(saved["error_code"], "response_incomplete")
        self.assertEqual(saved["usage"]["total_tokens"], 1300)
        self.assertEqual(saved["response_id"], "resp_incomplete")
        self.assertEqual(saved["model"], self.args.model)
        self.assertEqual(saved["evidence_fingerprint"], self.pack["fingerprint"])
        self.assertIsNone(saved["validation"])
        self.assertNotIn("answer", saved)
        self.assertNotIn("test-private-key", text)
        self.assertNotIn("untrusted-server-body", text)

    def test_transport_failure_keeps_unknown_usage_and_nonzero_exit(self):
        with patch("evidenceforge.providers.OpenAIResponses.generate", side_effect=ProviderError("transport_error_usage_unknown")):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(["answer", "--pack", str(self.pack_path), "--model", self.args.model,
                             "--output", str(self.output)])
        self.assertEqual(code, 1)
        self.assertIn("transport_error_usage_unknown", stderr.getvalue())
        saved = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "error")
        self.assertIsNone(saved["usage"])
        self.assertIsNone(saved["response_id"])
        self.assertIsNone(saved["validation"])

    def test_citation_failure_persists_safe_validation_without_invalid_answer(self):
        def generate(provider, query, documents):
            provider.last_usage = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
            provider.response_id = "resp_citation_failure"
            provider.last_validation = {"valid": False, "errors": ["citation[0]:quote_not_exact_substring"],
                                        "checks": {"quote_match": False, "entailment": "not_checked"}}
            raise ProviderError("answer_citation_contract_failed")

        with patch("evidenceforge.providers.OpenAIResponses.generate", autospec=True, side_effect=generate):
            with self.assertRaises(ProviderError):
                run(self.args)
        saved = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertFalse(saved["validation"]["valid"])
        self.assertEqual(saved["usage"]["total_tokens"], 120)
        self.assertNotIn("answer", saved)


if __name__ == "__main__":
    unittest.main()
