"""Ranking semantics and numerical boundaries; no external corpus required."""

import math
import unittest

from evidenceforge.dataset import Document
from evidenceforge.retrieval import BM25, ExactDense, Hit, reciprocal_rank_fusion, tokenize

try:
    import numpy as np
except ImportError:
    np = None


class SparseRetrievalTests(unittest.TestCase):
    def test_unicode_tokenization_is_case_insensitive_and_splits_underscores(self):
        self.assertEqual(tokenize("Café T_CELL 48μg — DNA"), ["café", "t", "cell", "48μg", "dna"])

    def test_single_document_bm25_matches_known_score(self):
        index = BM25([Document("one", "", "alpha alpha")])
        hit = index.search("alpha")[0]
        self.assertAlmostEqual(hit.score, math.log(4 / 3) * 1.375)

    def test_retrieval_uses_title_and_returns_no_unmatched_documents(self):
        index = BM25([Document("a", "Mitochondria", "energy production"), Document("b", "Heart", "blood flow")])
        self.assertEqual([h.document_id for h in index.search("mitochondria")], ["a"])
        self.assertEqual(index.search("nonexistent-term"), [])
        self.assertEqual(index.search(""), [])

    def test_bm25_length_normalization_rewards_concise_matching_document(self):
        index = BM25([Document("short", "", "alpha"), Document("long", "", "alpha " + "noise " * 50)])
        self.assertEqual(index.search("alpha")[0].document_id, "short")
        self.assertTrue(all(h.score > 0 for h in index.search("alpha")))

    def test_ties_are_document_id_order_independent_of_corpus_order(self):
        index = BM25([Document("z", "", "same"), Document("a", "", "same")])
        self.assertEqual([h.document_id for h in index.search("same")], ["a", "z"])

    def test_duplicate_ids_and_nonfinite_bm25_parameters_are_rejected(self):
        document = Document("a", "", "text")
        with self.assertRaises(ValueError):
            BM25([document, document])
        for kwargs in ({"k1": float("nan")}, {"k1": float("inf")}, {"b": float("nan")}, {"k1": 0}, {"b": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BM25([document], **kwargs)

    def test_rrf_uses_rank_not_incomparable_raw_scores(self):
        first = [[Hit("a", 10000), Hit("b", 1)], [Hit("b", -100), Hit("c", -200)]]
        second = [[Hit("a", 0.1), Hit("b", 0.09)], [Hit("b", 1e9), Hit("c", 1e8)]]
        fused = reciprocal_rank_fusion(first)
        self.assertEqual(fused, reciprocal_rank_fusion(second))
        self.assertEqual(fused[0].document_id, "b")
        self.assertAlmostEqual(fused[0].score, 1 / 62 + 1 / 61)

    def test_rrf_duplicates_do_not_inflate_votes_or_shift_unique_ranks(self):
        unique = [[Hit("a", 1), Hit("b", 0.5)]]
        duplicate = [[Hit("a", 1), Hit("a", 0.9), Hit("b", 0.5)]]
        self.assertEqual(reciprocal_rank_fusion(unique), reciprocal_rank_fusion(duplicate))

    def test_rrf_ties_limits_and_empty_rankings(self):
        rankings = [[Hit("z", 1)], [Hit("a", 1)]]
        self.assertEqual([h.document_id for h in reciprocal_rank_fusion(rankings, k=1)], ["a"])
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion(rankings, constant=0)


@unittest.skipIf(np is None, "NumPy is an optional dense-retrieval dependency")
class DenseRetrievalTests(unittest.TestCase):
    def test_cosine_ranking_normalizes_documents_and_query(self):
        index = ExactDense(["opposite", "orthogonal", "aligned"], [[-1, 0], [0, 12], [8, 0]])
        hits = index.search([5, 0])
        self.assertEqual([h.document_id for h in hits], ["aligned", "orthogonal", "opposite"])
        self.assertEqual([round(h.score, 6) for h in hits], [1, 0, -1])

    def test_tied_dense_results_use_document_id(self):
        index = ExactDense(["z", "a"], [[1, 1], [2, 2]])
        self.assertEqual([h.document_id for h in index.search([1, 1])], ["a", "z"])

    def test_invalid_vectors_and_shapes_are_rejected(self):
        for matrix in ([[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]], [1, 2], [[1, 0], [0, 1]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                ExactDense(["a"], matrix)
        index = ExactDense(["a"], [[1, 0]])
        for query in ([0, 0], [1], [float("nan"), 1], [[1, 0]]):
            with self.subTest(query=query), self.assertRaises(ValueError):
                index.search(query)

    def test_large_finite_vectors_retain_their_direction(self):
        index = ExactDense(["a", "b"], [[1e38, 0], [0, 1e38]])
        hits = index.search([1e38, 0])
        self.assertEqual(hits[0].document_id, "a")
        self.assertAlmostEqual(hits[0].score, 1.0)
        self.assertTrue(all(math.isfinite(hit.score) for hit in hits))


if __name__ == "__main__":
    unittest.main()
