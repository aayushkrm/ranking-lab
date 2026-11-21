"""Offline MovieLens ranking with timestamp-group holdouts and full-catalog scoring."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import sys
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

URL = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
MD5 = '0e33842e24a9c977be4e0107933c0723'


@dataclass(frozen=True)
class Rating:
    user: int
    item: int
    rating: int
    timestamp: int


def download(destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f'destination exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        tmp = Path(tmp)
        archive = tmp / 'download.zip'
        digest = hashlib.md5()  # GroupLens published checksum; detects accidental changes, not authentication.
        with urlopen(URL, timeout=60) as response, archive.open('wb') as out:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 50 * 1024 * 1024:
                    raise ValueError('archive exceeds 50 MiB')
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != MD5:
            raise ValueError('MovieLens archive checksum mismatch')
        stage = tmp / 'stage'
        stage.mkdir()
        with ZipFile(archive) as z:
            for name in ('u.data', 'u.item', 'README'):
                info = z.getinfo('ml-100k/' + name)
                if info.is_dir() or info.file_size > 20 * 1024 * 1024:
                    raise ValueError(f'invalid archive member: {name}')
                with z.open(info) as source, (stage / name).open('wb') as out:
                    while chunk := source.read(1024 * 1024):
                        out.write(chunk)
        stage.rename(destination)
    return destination


def load_data(directory):
    directory = Path(directory)
    items = set()
    with (directory / 'u.item').open(encoding='latin-1') as f:
        for line in f:
            item = int(line.split('|', 1)[0])
            if item in items:
                raise ValueError(f'duplicate catalog item {item}')
            items.add(item)
    rows, seen = [], set()
    with (directory / 'u.data').open(encoding='ascii') as f:
        for n, line in enumerate(f, 1):
            try:
                user, item, rating, timestamp = map(int, line.split())
            except ValueError as exc:
                raise ValueError(f'u.data line {n}: expected four integers') from exc
            if user <= 0 or item not in items or not 1 <= rating <= 5 or timestamp <= 0 or (user, item) in seen:
                raise ValueError(f'u.data line {n}: invalid or duplicate interaction')
            seen.add((user, item))
            rows.append(Rating(user, item, rating, timestamp))
    if not rows or not items:
        raise ValueError('empty input')
    return rows, items


def split(rows):
    """Global 80/10/10 event-time cutoffs; all identical timestamps stay together."""
    if len(rows) < 3:
        raise ValueError('at least three interactions are required')
    times = sorted(r.timestamp for r in rows)
    val_start, test_start = times[int(len(times) * 0.8)], times[int(len(times) * 0.9)]
    if val_start >= test_start:
        raise ValueError('not enough distinct timestamps around the split cutoffs')
    train = [r for r in rows if r.timestamp < val_start]
    validation = [r for r in rows if val_start <= r.timestamp < test_start]
    test = [r for r in rows if r.timestamp >= test_start]
    if not train or not validation or not test:
        raise ValueError('empty split')
    return train, validation, test, (val_start, test_start)


def train_model(rows, mode='popularity', shrink=0):
    """Binary positive interactions. Cosine item-item similarity with co-count shrinkage."""
    if mode not in ('popularity', 'item_cf', 'item_cf_fallback') or shrink < 0:
        raise ValueError('invalid model')
    users = defaultdict(set)
    seen = defaultdict(set)
    for r in rows:
        seen[r.user].add(r.item)
        if r.rating >= 4:
            users[r.user].add(r.item)
    counts = Counter(i for history in users.values() for i in history)
    neighbors = defaultdict(list)
    if mode in ('item_cf', 'item_cf_fallback'):
        pairs = Counter()
        for history in users.values():
            ordered = sorted(history)
            for a_index, a in enumerate(ordered):
                for b in ordered[a_index + 1:]:
                    pairs[a, b] += 1
        for (a, b), count in pairs.items():
            similarity = count / math.sqrt(counts[a] * counts[b]) * count / (count + shrink)
            neighbors[a].append((b, similarity))
            neighbors[b].append((a, similarity))
    return {'mode': mode, 'seen': seen, 'users': users, 'counts': counts, 'neighbors': neighbors}


def ranked_details(model, user, catalog, k):
    """Rank unseen candidates; fallback uses popularity only when CF has no score."""
    if k < 1:
        raise ValueError('k must be positive')
    scores = defaultdict(float)
    if model['mode'] in ('item_cf', 'item_cf_fallback'):
        for prior in sorted(model['users'].get(user, ())):
            for item, similarity in model['neighbors'].get(prior, ()):
                scores[item] += similarity
    else:
        scores.update(model['counts'])
    seen = model['seen'].get(user, set())
    candidates = [item for item in catalog if item not in seen]
    if model['mode'] == 'item_cf_fallback':
        # CF evidence has priority; popularity fills all zero-signal positions.
        candidates.sort(key=lambda item: (-bool(scores[item]), -scores[item],
                                           -model['counts'].get(item, 0), item))
    else:
        candidates.sort(key=lambda item: (-scores[item], item))
    return [{'item_id': item, 'score': scores[item],
             'source': ('item_cf' if scores[item] else 'popularity')
             if model['mode'] == 'item_cf_fallback' else model['mode'],
             'prior_positive_count': model['counts'].get(item, 0)}
            for item in candidates[:k]]


def rank(model, user, catalog, k):
    return [r['item_id'] for r in ranked_details(model, user, catalog, k)]


def evaluate(model, targets, catalog, k=10):
    if not catalog:
        raise ValueError('empty candidate catalog')
    grouped = defaultdict(list)
    for row in targets:
        grouped[row.user].append(row)
    per_user, recommendations = [], set()
    for user in sorted(grouped):
        relevant = {r.item for r in grouped[user] if r.rating >= 4 and r.item not in model['seen'].get(user, set())}
        if not relevant:
            continue
        ranked = rank(model, user, catalog, k)
        recommendations.update(ranked)
        hit = float(any(i in relevant for i in ranked))
        dcg = sum(1 / math.log2(position + 2) for position, item in enumerate(ranked) if item in relevant)
        ideal = sum(1 / math.log2(position + 2) for position in range(min(k, len(relevant))))
        per_user.append({'user_id': user, 'target_positives': len(relevant),
                         'unavailable_positives': len(relevant - catalog),
                         'cold_user': user not in model['seen'], 'hit': hit, 'ndcg': dcg / ideal})
    if not per_user:
        raise ValueError('no users with positive targets in this split')
    return {'eligible_users': len(per_user), 'excluded_no_novel_positive': len(grouped) - len(per_user),
            'cold_users': sum(x['cold_user'] for x in per_user),
            'unavailable_positives': sum(x['unavailable_positives'] for x in per_user),
            'target_positives': sum(x['target_positives'] for x in per_user),
            'hit_rate': sum(x['hit'] for x in per_user) / len(per_user),
            'ndcg': sum(x['ndcg'] for x in per_user) / len(per_user),
            'catalog_coverage': len(recommendations) / len(catalog), 'per_user': per_user}


def stratify_users(result):
    """Summarize user-averaged metrics; catalog coverage is not additive by group."""
    groups = {}
    for name, is_cold in (('cold', True), ('warm', False)):
        rows = [r for r in result['per_user'] if r['cold_user'] == is_cold]
        groups[name] = {'users': len(rows),
                        'hit_rate': sum(r['hit'] for r in rows) / len(rows) if rows else None,
                        'ndcg': sum(r['ndcg'] for r in rows) / len(rows) if rows else None,
                        'target_positives': sum(r['target_positives'] for r in rows),
                        'unavailable_positives': sum(r['unavailable_positives'] for r in rows)}
    return groups


def paired_bootstrap(baseline, candidate, *, seed=42, repetitions=2000):
    """95% percentile bootstrap of paired per-user metric differences (candidate - baseline)."""
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError('repetitions must be a positive integer')
    def index(rows):
        output = {}
        for row in rows:
            uid = row['user_id']
            if uid in output:
                raise ValueError('duplicate user in paired comparison')
            output[uid] = row
        return output
    old, new = index(baseline), index(candidate)
    if not old or old.keys() != new.keys():
        raise ValueError('paired comparisons require identical nonempty user sets')
    ids = sorted(old)
    differences = {metric: [new[u][metric] - old[u][metric] for u in ids] for metric in ('hit', 'ndcg')}
    rng = random.Random(seed)
    samples = {metric: [] for metric in differences}
    for _ in range(repetitions):
        draw = [rng.randrange(len(ids)) for _ in ids]
        for metric in samples:
            samples[metric].append(sum(differences[metric][i] for i in draw) / len(ids))
    def percentile(values, p):
        values = sorted(values)
        position = p * (len(values) - 1)
        low, high = math.floor(position), math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)
    return {'users': len(ids), 'seed': seed, 'repetitions': repetitions, 'confidence_level': .95,
            'metrics': {metric: {'mean_difference': sum(differences[metric]) / len(ids),
                                 'ci_low': percentile(samples[metric], .025),
                                 'ci_high': percentile(samples[metric], .975)} for metric in differences}}


def experiment(rows, catalog, k=10):
    train, val, test, cutoffs = split(rows)
    val_catalog = {r.item for r in train}
    test_catalog = {r.item for r in train + val}
    settings = [('popularity', 0), ('item_cf', 0), ('item_cf', 10), ('item_cf', 100)]
    trials = []
    for mode, shrink in settings:
        metrics = evaluate(train_model(train, mode, shrink), val, val_catalog, k)
        trials.append({'mode': mode, 'shrink': shrink, 'metrics': metrics})
    comparisons = [{'mode': trial['mode'], 'shrink': trial['shrink'],
                    'difference_from_popularity': paired_bootstrap(trials[0]['metrics']['per_user'], trial['metrics']['per_user'])}
                   for trial in trials[1:]]
    selected = max(range(len(trials)), key=lambda i: trials[i]['metrics']['ndcg'])
    setting = trials[selected]
    baseline = evaluate(train_model(train + val), test, test_catalog, k)
    selected_test = baseline if setting['mode'] == 'popularity' else evaluate(train_model(train + val, setting['mode'], setting['shrink']), test, test_catalog, k)
    return {'protocol': {'k': k, 'positive_rating_min': 4, 'selection_metric': 'validation_ndcg',
                         'tie_break': 'first setting in declared grid', 'metadata_catalog_size': len(catalog),
                         'validation_catalog_size': len(val_catalog), 'test_catalog_size': len(test_catalog),
                         'validation_start_timestamp': cutoffs[0], 'test_start_timestamp': cutoffs[1],
                         'train_ratings': len(train), 'validation_ratings': len(val), 'test_ratings': len(test)},
            'validation': trials, 'validation_paired_bootstrap': comparisons,
            'test_by_user_history': {'popularity': stratify_users(baseline), 'selected': stratify_users(selected_test)},
            'selected': {'mode': setting['mode'], 'shrink': setting['shrink']},
            'test_popularity': baseline, 'test_selected': selected_test}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    d = sub.add_parser('download'); d.add_argument('--data', default='data/ml-100k')
    run = sub.add_parser('run'); run.add_argument('--data', default='data/ml-100k'); run.add_argument('--k', type=int, default=10); run.add_argument('--output', default='results.json')
    args = parser.parse_args(argv)
    try:
        if args.command == 'download':
            print(f'Downloaded verified data to {download(args.data)}')
        else:
            if args.k < 1:
                raise ValueError('k must be positive')
            rows, catalog = load_data(args.data)
            result = experiment(rows, catalog, args.k)
            Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            print(json.dumps({'selected': result['selected'], 'test_popularity': {k:v for k,v in result['test_popularity'].items() if k != 'per_user'}, 'test_selected': {k:v for k,v in result['test_selected'].items() if k != 'per_user'}}, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
