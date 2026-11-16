"""Citation provenance checks are deliberately weaker than semantic grading."""

import unittest

from evidenceforge.evidence import Answer, Citation, EvidenceDocument, parse_answer, validate_answer


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            EvidenceDocument("paper-a", "Trial A", "The study enrolled 48 adults. Follow-up lasted six weeks."),
            EvidenceDocument("paper-b", "Trial B", "No mortality benefit was established."),
        ]

    def answer(self, **updates):
        value = {"status": "answer", "answer": "Trial A enrolled 48 adults.",
                 "citations": [Citation("paper-a", "The study enrolled 48 adults.")]}
        value.update(updates)
        return Answer(**value)

    def test_exact_quote_in_retrieved_document_passes(self):
        result = validate_answer(self.answer(), self.documents)
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["checks"]["quote_match"])

    def test_document_id_must_be_in_this_retrieval_set(self):
        result = validate_answer(self.answer(citations=[Citation("not-retrieved", "The study enrolled 48 adults.")]), self.documents)
        self.assertFalse(result["valid"])
        self.assertFalse(result["checks"]["citation_documents"])
        self.assertIn("citation[0]:document_not_retrieved", result["errors"])

    def test_quote_must_belong_to_the_cited_document(self):
        answer = self.answer(citations=[Citation("paper-b", "The study enrolled 48 adults.")])
        result = validate_answer(answer, self.documents)
        self.assertFalse(result["valid"])
        self.assertTrue(result["checks"]["citation_documents"])
        self.assertFalse(result["checks"]["quote_match"])

    def test_paraphrase_case_change_and_empty_quote_are_rejected(self):
        for quote in ("The study had 48 participants.", "the study enrolled 48 adults.", "", " \n "):
            with self.subTest(quote=quote):
                result = validate_answer(self.answer(citations=[Citation("paper-a", quote)]), self.documents)
                self.assertFalse(result["valid"])

    def test_unicode_quotes_are_exact_without_normalization(self):
        documents = [EvidenceDocument("unicode", "Muestra", "La dosis fue 10 μg; café.")]
        exact = Answer("answer", "La dosis fue 10 μg.", [Citation("unicode", "10 μg")])
        altered = Answer("answer", "La dosis fue 10 μg.", [Citation("unicode", "cafe\u0301")])
        self.assertTrue(validate_answer(exact, documents)["valid"])
        self.assertFalse(validate_answer(altered, documents)["valid"])

    def test_factual_answer_requires_text_and_at_least_one_citation(self):
        for answer in (self.answer(answer=""), self.answer(answer=" \n"), self.answer(citations=[])):
            with self.subTest(answer=answer):
                self.assertFalse(validate_answer(answer, self.documents)["valid"])

    def test_abstention_requires_empty_prose_and_citations(self):
        self.assertTrue(validate_answer(Answer("abstain", "", []), self.documents)["valid"])
        self.assertTrue(validate_answer(Answer("abstain", "", []), [])["valid"])
        invalid = [Answer("abstain", "I think the answer is 48.", []),
                   Answer("abstain", " ", []), Answer("abstain", "", [Citation("paper-a", "48")])]
        for answer in invalid:
            with self.subTest(answer=answer):
                self.assertFalse(validate_answer(answer, self.documents)["valid"])

    def test_duplicate_document_ids_fail_instead_of_picking_a_source(self):
        duplicate = self.documents + [EvidenceDocument("paper-a", "Conflict", "Different source.")]
        result = validate_answer(self.answer(), duplicate)
        self.assertFalse(result["valid"])
        self.assertIn("document[2]:duplicate_id", result["errors"])

    def test_invalid_retrieval_envelopes_fail_closed(self):
        candidates = [None, {}, ["not a document"], [EvidenceDocument("", "T", "text")],
                      [EvidenceDocument("a", 4, "text")]]
        for documents in candidates:
            with self.subTest(documents=documents):
                result = validate_answer(self.answer(), documents)
                self.assertFalse(result["valid"])

    def test_wire_parser_requires_exact_keys_and_types(self):
        valid = self.answer().to_dict()
        invalid = [None, [], {}, {**valid, "extra": "field"}, {**valid, "status": "unknown"},
                   {**valid, "status": []}, {**valid, "answer": 48}, {**valid, "citations": "paper-a"},
                   {**valid, "citations": [{"document_id": "paper-a", "quote": "48", "extra": True}]},
                   {**valid, "citations": [{"document_id": 1, "quote": "48"}]}]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_answer(payload)

    def test_wire_parser_roundtrips_without_coercion(self):
        answer = self.answer()
        self.assertEqual(parse_answer(answer.to_dict()), answer)
        self.assertEqual(self.documents[0].to_dict()["id"], "paper-a")
        self.assertEqual(answer.citations[0].to_dict(), {"document_id": "paper-a", "quote": "The study enrolled 48 adults."})

    def test_malformed_typed_envelopes_also_fail_validation(self):
        invalid = [Answer("unsupported", "x", []), Answer("answer", 1, []),
                   Answer("answer", "x", [{"document_id": "paper-a", "quote": "48"}]),
                   {"status": "answer", "answer": "x", "citations": []}]
        for answer in invalid:
            with self.subTest(answer=answer):
                self.assertFalse(validate_answer(answer, self.documents)["valid"])

    def test_quote_match_is_not_an_entailment_or_truth_claim(self):
        # Intentionally contradictory prose illustrates the precise boundary.
        # An LLM/human claim-level grader must reject this in a separate eval.
        contradictory = self.answer(answer="Trial A enrolled 9,000 children.")
        result = validate_answer(contradictory, self.documents)
        self.assertTrue(result["valid"])
        for check in ("entailment", "source_veracity", "causal_grounding"):
            self.assertEqual(result["checks"][check], "not_checked")

    def test_multiple_citations_all_need_to_pass(self):
        answer = self.answer(citations=[Citation("paper-a", "48 adults"), Citation("paper-b", "Invented claim")])
        result = validate_answer(answer, self.documents)
        self.assertFalse(result["valid"])
        self.assertIn("citation[1]:quote_not_exact_substring", result["errors"])


if __name__ == "__main__":
    unittest.main()
