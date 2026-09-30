"""
Cache a per-language sample of FineWeb-2 (plus FineWeb for English) under `corpus/fineweb/`.

The tuned classifier is trained on these documents, so `build.py` reads the cache and
never the network: streaming 175 configs is slow and flaky, and a build must be
repeatable. Each FineWeb-2 config is aligned to a fastText label with
`align_to_fasttext`; configs that fastText cannot name are skipped. Test rows come from
FineWeb-2's own `test` split, and for English from documents whose id hashes into a
held-out tenth.

Usage:
    uv run python -m lumi_language_id.fetch_fineweb [--train 300] [--test 100]
"""
import argparse
import hashlib
import itertools
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, list_repo_tree

from lumi_language_id import align_to_fasttext

FINEWEB2 = 'HuggingFaceFW/fineweb-2'
FINEWEB = 'HuggingFaceFW/fineweb'
# FineWeb (v1) is English-only; it goes through the same aligner as every FineWeb-2 config.
FINEWEB_CONFIG = 'eng_Latn'
OUT_DIR = Path(__file__).parent.parent / 'corpus' / 'fineweb'


def fineweb2_configs():
    """Map each fastText label to the FineWeb-2 configs (and their splits) that carry it."""
    splits = defaultdict(set)
    for entry in list_repo_tree(FINEWEB2, repo_type='dataset', path_in_repo='data', recursive=True):
        parts = entry.path.split('/')
        if len(parts) >= 3 and not parts[1].endswith('_removed'):
            splits[parts[1]].add(parts[2])
    by_label = defaultdict(list)
    for config, config_splits in sorted(splits.items()):
        label = align_to_fasttext(*config.split('_'))
        if label != 'und':
            by_label[label].append((config, config_splits))
    return by_label


def fetch_sizes(by_label):
    """
    Write `sizes.json`: bytes of training data per fastText label on the Hub.

    `build.py` weights each language by its size, tempered, so the classifier is
    calibrated for text as it occurs rather than for a uniform mix of 176 languages.
    """
    api = HfApi()
    by_config = defaultdict(int)
    for entry in api.list_repo_tree(FINEWEB2, repo_type='dataset', path_in_repo='data', recursive=True):
        parts = entry.path.split('/')
        if len(parts) == 4 and parts[2] == 'train' and getattr(entry, 'size', None):
            by_config[parts[1]] += entry.size
    sizes = {label: sum(by_config[c] for c, _ in configs) for label, configs in by_label.items()}
    sizes[align_to_fasttext(*FINEWEB_CONFIG.split('_'))] = sum(
        entry.size for entry in api.list_repo_tree(FINEWEB, repo_type='dataset', path_in_repo='data', recursive=True)
        if getattr(entry, 'size', None))
    (OUT_DIR / 'sizes.json').write_text(json.dumps(sizes, indent=1), encoding='utf-8')
    return sizes


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def fetch_label(label, configs, n_train, n_test):
    """Write `<label>.train.jsonl` and `<label>.test.jsonl`, splitting the budget over configs."""
    counts = {}
    for split, n in (('train', n_train), ('test', n_test)):
        path = OUT_DIR / f'{label}.{split}.jsonl'
        if path.exists():
            counts[split] = sum(1 for _ in open(path, encoding='utf-8'))
            continue
        usable = [c for c, s in configs if split in s]
        per_config = -(-n // len(usable)) if usable else 0
        rows = []
        for config in usable:
            # Read the parquet files with their own schema: some configs (diq_Latn,
            # eml_Latn) carry an extra `wordlist_ratio` column the repo's declared
            # features lack, and casting to those features fails.
            ds = load_dataset('parquet', split='train', streaming=True,
                              data_files=f'hf://datasets/{FINEWEB2}/data/{config}/{split}/*.parquet')
            rows += [{'text': r['text'], 'label': label, 'source': config}
                     for r in itertools.islice(ds, per_config)]
        write_rows(path, rows[:n])
        counts[split] = len(rows[:n])
    return label, counts


def fetch_english(n_train, n_test):
    """English is not in FineWeb-2; take it from FineWeb, holding out by id hash."""
    label = align_to_fasttext(*FINEWEB_CONFIG.split('_'))
    if (OUT_DIR / f'{label}.train.jsonl').exists() and (OUT_DIR / f'{label}.test.jsonl').exists():
        return label, {'train': n_train, 'test': n_test}
    rows = {'train': [], 'test': []}
    for r in load_dataset(FINEWEB, name='sample-10BT', split='train', streaming=True):
        split = 'test' if int(hashlib.sha1(r['id'].encode()).hexdigest(), 16) % 10 == 0 else 'train'
        if len(rows[split]) < (n_train if split == 'train' else n_test):
            rows[split].append({'text': r['text'], 'label': label, 'source': FINEWEB_CONFIG})
        if len(rows['train']) >= n_train and len(rows['test']) >= n_test:
            break
    for split, split_rows in rows.items():
        write_rows(OUT_DIR / f'{label}.{split}.jsonl', split_rows)
    return label, {s: len(v) for s, v in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--train', type=int, default=300, help='documents per label for training')
    parser.add_argument('--test', type=int, default=100, help='documents per label for testing')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_label = fineweb2_configs()
    print(f'{len(by_label)} fastText labels from FineWeb-2, plus en from FineWeb', flush=True)
    if not (OUT_DIR / 'sizes.json').exists():
        print(f'sizes for {len(fetch_sizes(by_label))} labels', flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(fetch_label, label, configs, args.train, args.test)
                for label, configs in by_label.items()]
        jobs.append(pool.submit(fetch_english, args.train, args.test))
        for job in jobs:
            label, counts = job.result()
            print(f'{label:5} train {counts["train"]:4}  test {counts["test"]:4}', flush=True)


def exit_without_arrow_teardown():
    """
    End the process without running C++ static destructors.

    After a full run, pyarrow 25.0.1's global thread pool deadlocks in its own static
    destructor: C `exit()` destroys the pool, `ThreadPool::Shutdown` waits on a condition
    variable for its workers, and every worker is idle waiting on a condition variable
    for work. Sampled on macOS: 1 thread in `__cxa_finalize_ranges` ->
    `ThreadPool::Shutdown`, 8 workers in `condition_variable::wait`, 0% CPU, forever.
    Every file is written and closed before this point, so skipping teardown loses
    nothing. Small runs never grow the pool and exit normally either way.

    Fixed upstream: the same full run exits cleanly on pyarrow 26.0.0.dev323, which
    contains apache/arrow#48137's fix (a hang in `ThreadPool::Shutdown`). Remove this
    once pyarrow 26 is the minimum.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
    exit_without_arrow_teardown()
