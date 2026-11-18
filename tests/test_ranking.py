import math
import unittest
import tempfile
from pathlib import Path
from ranking_lab import Rating, split, train_model, rank, evaluate, experiment, load_data, paired_bootstrap, stratify_users


class RankingTests(unittest.TestCase):
    def test_timestamp_groups_do_not_leak(self):
        rows = [Rating(1, t, 5, t) for t in range(1, 19)]
        rows += [Rating(2, 19, 5, 17), Rating(2, 20, 5, 19)]
        train, val, test, cutoffs = split(rows)
        self.assertEqual(cutoffs, (17, 18))
        self.assertTrue(all(r.timestamp < 17 for r in train))
        self.assertEqual({r.timestamp for r in val}, {17})
        self.assertEqual({r.timestamp for r in test}, {18, 19})
        self.assertEqual(sum(r.timestamp == 17 for r in val), 2)

    def test_seen_exclusion_ties_and_relevance(self):
        rows = [Rating(1, 1, 1, 1), Rating(2, 2, 5, 1), Rating(3, 3, 5, 1)]
        model = train_model(rows)
        self.assertEqual(rank(model, 1, {1, 2, 3, 4}, 3), [2, 3, 4])
        result = evaluate(model, [Rating(1, 3, 5, 2), Rating(1, 4, 4, 2)], {1, 2, 3, 4}, 2)
        self.assertEqual(result['hit_rate'], 1)
        self.assertAlmostEqual(result['ndcg'], (1 / math.log2(3)) / (1 + 1 / math.log2(3)))
        self.assertEqual(result['catalog_coverage'], 0.5)

    def test_item_cf_and_zero_positive_users(self):
        rows = [Rating(1, 1, 5, 1), Rating(2, 1, 5, 1), Rating(2, 3, 5, 1)]
        model = train_model(rows, 'item_cf')
        self.assertEqual(rank(model, 1, {1, 2, 3}, 2), [3, 2])
        result = evaluate(model, [Rating(1, 3, 4, 2), Rating(2, 2, 2, 2)], {1, 2, 3}, 1)
        self.assertEqual(result['eligible_users'], 1)
        self.assertEqual(result['excluded_no_novel_positive'], 1)
        self.assertEqual(result['hit_rate'], 1)

    def test_prior_targets_and_cold_user_are_explicit(self):
        model = train_model([Rating(1, 1, 5, 1), Rating(2, 2, 5, 1)])
        targets = [Rating(1, 1, 5, 2), Rating(3, 3, 5, 2)]
        result = evaluate(model, targets, {1, 2}, 1)
        self.assertEqual(result['excluded_no_novel_positive'], 1)
        self.assertEqual(result['cold_users'], 1)
        self.assertEqual(result['unavailable_positives'], 1)
        self.assertEqual(result['ndcg'], 0)

    def test_loader_rejects_duplicate_interactions(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'u.item').write_text('1|one\n', encoding='latin-1')
            (p / 'u.data').write_text('1 1 5 1\n1 1 4 2\n', encoding='ascii')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load_data(p)

    def test_bootstrap_pairs_ids_and_is_reproducible(self):
        baseline = [{'user_id': 1, 'hit': 0, 'ndcg': 0}, {'user_id': 2, 'hit': 1, 'ndcg': .5}]
        candidate = [{'user_id': 2, 'hit': 1, 'ndcg': .5}, {'user_id': 1, 'hit': 1, 'ndcg': 1}]
        result = paired_bootstrap(baseline, candidate, repetitions=100)
        self.assertEqual(result, paired_bootstrap(baseline, list(reversed(candidate)), repetitions=100))
        self.assertEqual(result['metrics']['hit']['mean_difference'], .5)
        self.assertEqual(result['metrics']['ndcg']['mean_difference'], .5)
        self.assertEqual(result['metrics']['hit']['ci_low'], 0)
        self.assertEqual(result['metrics']['hit']['ci_high'], 1)
        with self.assertRaises(ValueError):
            paired_bootstrap(baseline, candidate[:1])
        with self.assertRaises(ValueError):
            paired_bootstrap(baseline, [candidate[0], candidate[0]])

    def test_strata_have_user_averages_and_no_coverage(self):
        result = {'per_user': [
            {'cold_user': True, 'hit': 1, 'ndcg': .5, 'target_positives': 2, 'unavailable_positives': 1},
            {'cold_user': True, 'hit': 0, 'ndcg': 0, 'target_positives': 1, 'unavailable_positives': 0}]}
        groups = stratify_users(result)
        self.assertEqual(groups['cold']['users'], 2)
        self.assertEqual(groups['cold']['hit_rate'], .5)
        self.assertEqual(groups['cold']['ndcg'], .25)
        self.assertNotIn('catalog_coverage', groups['cold'])
        self.assertIsNone(groups['warm']['ndcg'])

    def test_validation_selection_and_refit(self):
        rows = [Rating(1, t, 5, t) for t in range(1, 21)]
        output = experiment(rows, set(range(1, 21)), k=2)
        self.assertEqual(output['protocol']['train_ratings'], 16)
        self.assertEqual(len(output['validation']), 4)
        self.assertEqual(output['test_selected']['eligible_users'], 1)
        self.assertEqual(output['test_selected']['hit_rate'], 0)
        self.assertEqual(output['test_selected']['unavailable_positives'], 2)



if __name__ == '__main__':
    unittest.main()
