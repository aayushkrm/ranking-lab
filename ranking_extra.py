"""Exploratory rolling-window evaluation and local recommendation inspection."""
import argparse
import hashlib
import json
from pathlib import Path

from ranking_lab import load_data, train_model, ranked_details, evaluate, paired_bootstrap, stratify_users


def rolling_windows(rows, boundaries=(.6, .7, .8, .9, 1.0)):
    """Yield expanding train histories and disjoint target windows, grouped by timestamp."""
    if len(boundaries) < 2 or boundaries[-1] != 1 or any(not 0 < x <= 1 for x in boundaries) or any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError('boundaries must increase from above zero to one')
    timestamps = sorted(row.timestamp for row in rows)
    if not timestamps:
        raise ValueError('no ratings')
    cutoffs = [timestamps[int(len(timestamps) * p)] for p in boundaries[:-1]] + [None]
    if any(a >= b for a, b in zip(cutoffs[:-2], cutoffs[1:-1])):
        raise ValueError('not enough distinct timestamps at window boundaries')
    for start, end in zip(cutoffs, cutoffs[1:]):
        training = [row for row in rows if row.timestamp < start]
        targets = [row for row in rows if row.timestamp >= start and (end is None or row.timestamp < end)]
        if not training or not targets:
            raise ValueError('empty training or evaluation window')
        yield training, targets, start, end


def rolling_experiment(rows, k=10, shrink=10):
    """Compare fixed models; these windows are retrospective and exploratory."""
    reports = []
    for training, targets, start, end in rolling_windows(rows):
        catalog = {r.item for r in training}
        baseline = evaluate(train_model(training), targets, catalog, k)
        model = train_model(training, 'item_cf_fallback', shrink)
        candidate = evaluate(model, targets, catalog, k)
        # Fallback is measured over eligible users, the same population as metrics.
        fallback_users = sum(all(r['source'] == 'popularity' for r in ranked_details(model, x['user_id'], catalog, k)) for x in candidate['per_user'])
        def compact(metrics):
            return {key: value for key, value in metrics.items() if key != 'per_user'}
        reports.append({'start_inclusive': start, 'end_exclusive': end,
                        'train_ratings': len(training), 'target_ratings': len(targets),
                        'catalog_size': len(catalog), 'popularity': compact(baseline),
                        'item_cf_fallback': compact(candidate), 'users_with_only_popularity': fallback_users,
                        'slices': {'popularity': stratify_users(baseline), 'item_cf_fallback': stratify_users(candidate)},
                        'paired_difference_fallback_minus_popularity': paired_bootstrap(baseline['per_user'], candidate['per_user'])})
    return {'protocol': {'boundaries_by_event_fraction': [.6, .7, .8, .9, 1.0],
                         'fixed_shrink': shrink, 'k': k, 'exploratory': True,
                         'note': 'All four windows are retrospective, not independent final tests. Earlier test results have been inspected.'}, 'windows': reports}


def titles_from_file(path):
    titles = {}
    with open(path, encoding='latin-1') as f:
        for line in f:
            fields = line.rstrip('\n').split('|')
            titles[int(fields[0])] = fields[1]
    return titles


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    rolling = sub.add_parser('rolling', help='Compare fixed methods across exploratory time windows')
    rolling.add_argument('--data', default='data/ml-100k')
    rolling.add_argument('--output', default='rolling-results.json')
    rolling.add_argument('--k', type=int, default=10)
    demo = sub.add_parser('recommend', help='Inspect one user and recommendations using past ratings')
    demo.add_argument('user_id', type=int)
    demo.add_argument('--data', default='data/ml-100k')
    demo.add_argument('--as-of', type=int, help='Unix cutoff: use only ratings strictly before this time')
    demo.add_argument('--k', type=int, default=10)
    args = parser.parse_args(argv)
    try:
        if args.k < 1:
            raise ValueError('k must be positive')
        rows, _ = load_data(args.data)
        if args.command == 'rolling':
            report = rolling_experiment(rows, args.k)
            report['protocol']['ratings_sha256'] = hashlib.sha256((Path(args.data) / 'u.data').read_bytes()).hexdigest()
            Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            print(json.dumps(report, indent=2, allow_nan=False))
        else:
            if args.user_id < 1 or (args.as_of is not None and args.as_of < 1):
                raise ValueError('user ID and cutoff must be positive')
            history = [r for r in rows if args.as_of is None or r.timestamp < args.as_of]
            if not history:
                raise ValueError('no ratings before cutoff')
            titles = titles_from_file(Path(args.data) / 'u.item')
            model = train_model(history, 'item_cf_fallback', 10)
            user_history = sorted((r for r in history if r.user == args.user_id), key=lambda r: (-r.timestamp, r.item))
            recommendations = ranked_details(model, args.user_id, {r.item for r in history}, args.k)
            print(json.dumps({'user_id': args.user_id, 'cutoff_exclusive': args.as_of,
                              'scope': 'all available historical ratings' if args.as_of is None else 'ratings strictly before cutoff',
                              'history_count': len(user_history),
                              'recent_history': [{'title': titles[r.item], 'rating': r.rating, 'timestamp': r.timestamp} for r in user_history[:10]],
                              'recommendations': [{**r, 'title': titles[r['item_id']]} for r in recommendations]}, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
