import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ranking_lab import Rating, rank, ranked_details, train_model
from ranking_extra import rolling_windows, rolling_experiment


class RollingTests(unittest.TestCase):
    def test_global_cutoffs_exclude_other_users_future_and_keep_ties(self):
        rows = [Rating(1, i, 5, i) for i in range(1, 21)]
        rows += [Rating(2, 100, 5, 13), Rating(3, 101, 5, 19)]
        windows = list(rolling_windows(rows))
        for train, targets, start, end in windows:
            self.assertTrue(all(r.timestamp < start for r in train))
            self.assertTrue(all(r.timestamp >= start and (end is None or r.timestamp < end) for r in targets))
            self.assertFalse(set(train) & set(targets))
        self.assertEqual(sum(len(w[1]) for w in windows), len({r for w in windows for r in w[1]}))

    def test_cold_and_warm_zero_signal_use_popularity_and_exclude_seen(self):
        rows = [Rating(1, 1, 1, 1), Rating(2, 4, 5, 1), Rating(3, 4, 5, 1), Rating(4, 2, 5, 1)]
        model = train_model(rows, 'item_cf_fallback')
        self.assertEqual(rank(model, 999, {1, 2, 4}, 3), [4, 2, 1])
        self.assertEqual(rank(model, 1, {1, 2, 4}, 2), [4, 2])
        self.assertTrue(all(r['source'] == 'popularity' for r in ranked_details(model, 999, {1, 2, 4}, 3)))

    def test_cf_signal_precedes_popularity_and_popularity_fills(self):
        rows = [Rating(1, 1, 5, 1), Rating(2, 1, 5, 1), Rating(2, 3, 5, 1),
                Rating(3, 2, 5, 1), Rating(4, 2, 5, 1), Rating(5, 2, 5, 1)]
        model = train_model(rows, 'item_cf_fallback')
        result = ranked_details(model, 1, {1, 2, 3, 4}, 3)
        self.assertEqual([r['item_id'] for r in result], [3, 2, 4])
        self.assertEqual([r['source'] for r in result], ['item_cf', 'popularity', 'popularity'])

    def test_rolling_report_compares_same_eligible_users(self):
        rows = [Rating(i, 1, 5, i) for i in range(1, 21)]
        report = rolling_experiment(rows, k=1)
        self.assertEqual(len(report['windows']), 4)
        for window in report['windows']:
            self.assertEqual(window['popularity']['eligible_users'], window['item_cf_fallback']['eligible_users'])
            self.assertEqual(window['paired_difference_fallback_minus_popularity']['metrics']['hit']['mean_difference'], 0)
            self.assertEqual(window['users_with_only_popularity'], window['item_cf_fallback']['eligible_users'])

    def test_recommend_cli_uses_cutoff_and_titles(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'u.item').write_text('1|First\n2|Second\n3|Later\n', encoding='latin-1')
            (p / 'u.data').write_text('1 1 5 1\n2 2 5 2\n2 3 5 10\n', encoding='ascii')
            command = [sys.executable, 'ranking_extra.py', 'recommend', '1', '--data', d, '--as-of', '5']
            proc = subprocess.run(command, text=True, capture_output=True, check=True)
            output = json.loads(proc.stdout)
            self.assertEqual(output['history_count'], 1)
            self.assertEqual([r['title'] for r in output['recommendations']], ['Second'])
            self.assertEqual(output['recommendations'][0]['source'], 'popularity')


if __name__ == '__main__':
    unittest.main()
