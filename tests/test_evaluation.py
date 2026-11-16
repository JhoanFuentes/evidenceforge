from __future__ import annotations

import math
import unittest

from evidenceforge.evaluation import compare_runs, evaluate_run


class RetrievalMetricsTests(unittest.TestCase):
    def test_multiple_gold_documents_define_recall_denominator(self) -> None:
        result = evaluate_run({"q": ["a", "noise", "b"]}, {"q": {"a": 1, "b": 1, "c": 1}}, ks=(2, 3))
        metrics = result["per_query"]["q"]
        self.assertAlmostEqual(metrics["recall@2"], 1 / 3)
        self.assertAlmostEqual(metrics["recall@3"], 2 / 3)
        self.assertEqual(metrics["mrr@3"], 1)
        ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(metrics["ndcg@3"], (1 + 1 / math.log2(4)) / ideal)

    def test_duplicate_rankings_are_removed_before_cutoff(self) -> None:
        qrels = {"q": {"a": 1, "b": 1}}
        clean = evaluate_run({"q": ["a", "b"]}, qrels, ks=(2,))
        duplicate = evaluate_run({"q": ["a", "a", "b", "b"]}, qrels, ks=(2,))
        self.assertEqual(duplicate["aggregate"], clean["aggregate"])
        self.assertEqual(duplicate["aggregate"]["recall@2"], 1)

    def test_missing_queries_are_zero_and_remain_in_macro_average(self) -> None:
        result = evaluate_run({"seen": ["a"]}, {"seen": {"a": 1}, "missing": {"b": 1}})
        self.assertEqual(result["missing_query_ids"], ["missing"])
        self.assertEqual(result["evaluated_query_count"], 2)
        self.assertEqual(result["per_query"]["missing"]["ndcg@10"], 0)
        self.assertEqual(result["aggregate"]["ndcg@10"], 0.5)
        self.assertEqual(result["aggregate"]["recall@10"], 0.5)

    def test_graded_ndcg_penalizes_reversing_strong_and_weak_evidence(self) -> None:
        qrels = {"q": {"strong": 3, "weak": 1}}
        metrics = evaluate_run({"q": ["weak", "strong"]}, qrels)["aggregate"]
        expected = (1 + 7 / math.log2(3)) / (7 + 1 / math.log2(3))
        self.assertAlmostEqual(metrics["ndcg@10"], expected)
        self.assertLess(metrics["ndcg@10"], 1)
        self.assertEqual(metrics["recall@10"], 1)

    def test_unjudged_documents_are_nonrelevant_at_their_rank(self) -> None:
        metrics = evaluate_run({"q": ["unjudged", "gold"]}, {"q": {"gold": 1}})["aggregate"]
        self.assertEqual(metrics["mrr@10"], 0.5)
        self.assertAlmostEqual(metrics["ndcg@10"], 1 / math.log2(3))

    def test_no_positive_qrels_are_excluded_explicitly(self) -> None:
        result = evaluate_run(
            {"eligible": ["a"], "no-labels": ["x"], "unjudged-query": ["a"]},
            {"eligible": {"a": 1}, "no-labels": {}, "only-zero": {"x": 0, "y": -1}},
        )
        self.assertEqual(result["excluded_query_ids"], ["no-labels", "only-zero"])
        self.assertEqual(result["unjudged_ranking_query_ids"], ["unjudged-query"])
        self.assertEqual(result["aggregate"]["ndcg@10"], 1)
        self.assertEqual(result["evaluated_query_count"], 1)

    def test_empty_evaluation_is_not_a_perfect_score(self) -> None:
        for qrels in ({}, {"q": {}}, {"q": {"a": 0}}):
            with self.subTest(qrels=qrels), self.assertRaises(ValueError):
                evaluate_run({}, qrels)

    def test_large_grades_do_not_overflow_ndcg(self) -> None:
        result = evaluate_run({"q": ["strong", "weak"]}, {"q": {"strong": 10000, "weak": 1}})
        self.assertEqual(result["aggregate"]["ndcg@10"], 1)

    def test_invalid_cutoffs_rankings_and_labels_fail_clearly(self) -> None:
        for ks in ((), (0,), (-1,), (True,), (2.5,)):
            with self.subTest(ks=ks), self.assertRaises(ValueError):
                evaluate_run({"q": ["a"]}, {"q": {"a": 1}}, ks=ks)
        with self.assertRaises(ValueError):
            evaluate_run({"q": "a"}, {"q": {"a": 1}})
        with self.assertRaises(ValueError):
            evaluate_run({"q": ["a"]}, {"q": {"a": True}})


class BootstrapTests(unittest.TestCase):
    def test_identical_runs_have_zero_interval(self) -> None:
        scores = {"a": {"ndcg@10": 0.4}, "b": {"ndcg@10": 0.9}}
        result = compare_runs(scores, scores, samples=100)
        self.assertEqual(result["mean_delta"], 0)
        self.assertEqual(result["ci95"], [0, 0])
        self.assertEqual((result["wins"], result["ties"], result["losses"]), (0, 2, 0))

    def test_pairing_preserves_constant_query_improvement(self) -> None:
        baseline = {"a": {"ndcg@10": 0.1}, "b": {"ndcg@10": 0.7}}
        candidate = {"a": {"ndcg@10": 0.3}, "b": {"ndcg@10": 0.9}}
        result = compare_runs(baseline, candidate, samples=100)
        self.assertAlmostEqual(result["mean_delta"], 0.2)
        for endpoint in result["ci95"]:
            self.assertAlmostEqual(endpoint, 0.2)
        self.assertEqual(result["wins"], 2)

    def test_seed_reproducible_despite_input_dictionary_order(self) -> None:
        baseline = {"c": {"ndcg@10": 0.4}, "a": {"ndcg@10": 0.7}, "b": {"ndcg@10": 0.3}}
        candidate = {"a": {"ndcg@10": 0.2}, "b": {"ndcg@10": 0.3}, "c": {"ndcg@10": 0.9}}
        first = compare_runs(baseline, candidate, samples=200, seed=9)
        reordered = compare_runs(dict(reversed(list(baseline.items()))), candidate, samples=200, seed=9)
        self.assertEqual(first, reordered)
        self.assertEqual((first["wins"], first["ties"], first["losses"]), (1, 1, 1))
        self.assertLess(first["ci95"][0], 0)
        self.assertGreater(first["ci95"][1], 0)

    def test_mismatched_or_empty_query_sets_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compare_runs({}, {})
        with self.assertRaises(ValueError):
            compare_runs({"a": {"ndcg@10": 1}}, {"b": {"ndcg@10": 1}})

    def test_missing_and_nonfinite_metrics_are_rejected(self) -> None:
        valid = {"a": {"ndcg@10": 1}}
        for invalid in ({"a": {}}, {"a": {"ndcg@10": float("nan")}}, {"a": {"ndcg@10": True}}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                compare_runs(valid, invalid)
        with self.assertRaises(ValueError):
            compare_runs(valid, valid, samples=0)


if __name__ == "__main__":
    unittest.main()
