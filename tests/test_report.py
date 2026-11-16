from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest

from evidenceforge.report import write_report


def fixture() -> dict:
    system = {
        "aggregate": {"recall@5": 0.5, "recall@10": 1.0, "mrr@10": 0.5, "ndcg@10": 0.63093},
        "per_query": {"q1": {"recall@5": 0.5, "recall@10": 1.0, "mrr@10": 0.5, "ndcg@10": 0.63093}},
        "timing": {"search_seconds": 0.01, "p50_ms": 10, "p95_ms": 10},
        "rankings": {"q1": ["doc1", "doc2"]},
    }
    return {
        "schema_version": 1,
        "dataset": {"name": "Example relevance benchmark", "split": "test", "corpus_count": 25,
                    "query_count": 1, "sha256": "a" * 64},
        "config": {"top_k": 10}, "environment": {"python": "3.11"},
        "systems": {"baseline": system, "candidate": system},
        "comparisons": [{"baseline": "baseline", "candidate": "candidate", "metric": "ndcg@10",
                         "mean_delta": 0, "ci95": [0, 0], "wins": 0, "ties": 1, "losses": 0}],
        "generated_at": "2026-11-16T00:00:00Z",
    }


class Inspector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict]] = []
        self.data_chunks: list[str] = []
        self.in_data = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attributes = dict(attrs)
        self.tags.append((tag, attributes))
        if tag == "script" and attributes.get("id") == "report-data":
            self.in_data = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self.in_data = False

    def handle_data(self, data: str) -> None:
        if self.in_data:
            self.data_chunks.append(data)


class ReportTests(unittest.TestCase):
    def render(self, result: dict) -> tuple[str, Inspector]:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "report.html"
            write_report(result, output)
            html = output.read_text(encoding="utf-8")
        parser = Inspector()
        parser.feed(html)
        return html, parser

    def test_data_is_embedded_losslessly_and_report_has_no_remote_assets(self) -> None:
        result = fixture()
        html, parsed = self.render(result)
        self.assertEqual(json.loads("".join(parsed.data_chunks)), result)
        self.assertEqual(len([tag for tag, _ in parsed.tags if tag == "script"]), 2)
        self.assertFalse(any(attrs.get("src", "").startswith(("https:", "http:")) for _, attrs in parsed.tags))
        self.assertFalse(any(tag == "link" and attrs.get("rel") == "stylesheet" for tag, attrs in parsed.tags))
        self.assertIn("are not evaluated in this report", html)
        self.assertIn('id="negative-only"', html)
        self.assertIn('id="theme-toggle"', html)

    def test_untrusted_names_queries_and_rankings_cannot_inject_html(self) -> None:
        attack = '</script><img src=x onerror="alert(1)"><script>alert(2)</script>&\u2028'
        result = fixture()
        result["dataset"]["name"] = attack
        result["queries"] = {"q1": attack}
        system = result["systems"].pop("candidate")
        system["rankings"]["q1"].append(attack)
        result["systems"][attack] = system
        result["comparisons"][0]["candidate"] = attack
        result["config"]["note"] = attack
        result["limitations"] = [attack, "This public split is not a hidden test."]
        html, parsed = self.render(result)
        self.assertNotIn(attack, html)
        self.assertEqual(json.loads("".join(parsed.data_chunks)), result)
        self.assertFalse(any(tag == "img" for tag, _ in parsed.tags))
        self.assertFalse(any("onerror" in attrs for _, attrs in parsed.tags))
        self.assertEqual(len([tag for tag, _ in parsed.tags if tag == "script"]), 2)
        self.assertNotIn(".innerHTML", html)
        self.assertIn("\\u003c/script\\u003e", html)
        self.assertIn("<li>This public split is not a hidden test.</li>", html)

    def test_template_like_user_strings_are_literal(self) -> None:
        result = fixture()
        result["systems"]["@@CONFIG@@"] = result["systems"].pop("candidate")
        result["dataset"]["name"] = "@@JSON@@"
        html, parsed = self.render(result)
        self.assertIn("<h3>@@CONFIG@@</h3>", html)
        self.assertIn('class="dataset-name">@@JSON@@</div>', html)
        self.assertEqual(json.loads("".join(parsed.data_chunks)), result)

    def test_absent_query_text_stays_absent(self) -> None:
        result = fixture()
        _, parsed = self.render(result)
        embedded = json.loads("".join(parsed.data_chunks))
        self.assertNotIn("queries", embedded)
        self.assertEqual(embedded["systems"]["baseline"]["rankings"]["q1"], ["doc1", "doc2"])

    def test_empty_systems_are_rejected(self) -> None:
        result = fixture()
        result["systems"] = {}
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            write_report(result, Path(directory) / "report.html")


if __name__ == "__main__":
    unittest.main()
