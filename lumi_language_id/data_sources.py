"""
Functions that read training/test data from corpus files.
"""
import csv
import json
import math
import random
from pathlib import Path

import ftfy

from lumi_language_id import corpus_file, align_language_to_fasttext


def short_window(text, rng, min_chars=10, max_chars=500):
    """
    A window of log-uniform length starting at a random offset.

    Web documents are long, and the classifier must also be calibrated on short text
    (tweets, headlines, single sentences). The offset is random rather than 0 because
    many pages open with navigation boilerplate.
    """
    length = int(math.exp(rng.uniform(math.log(min_chars), math.log(max_chars))))
    if len(text) <= length:
        return text
    start = rng.randrange(len(text) - length)
    return text[start:start + length]


def fineweb_gen(split, seed=0):
    """
    Yield (text, label) from the FineWeb cache that `fetch_fineweb` writes: each
    document once in full and once as a short window, so both regimes are covered.
    """
    rng = random.Random(seed)
    for path in sorted(Path(corpus_file('fineweb')).glob(f'*.{split}.jsonl')):
        with open(path, encoding='utf-8') as rows:
            for line in rows:
                row = json.loads(line)
                text = ftfy.fix_text(row['text'])
                yield (text, row['label'])
                yield (short_window(text, rng), row['label'])


def twitter_gen():
    with open(
        corpus_file('TweetLID_corpusV2/tweetlid-test-tweets.tsv')
    ) as twitter_file:
        reader = csv.reader(twitter_file, delimiter='\t')
        for row in reader:
            # If there are multiple possibilities separated by +, take the first one
            label = row[2].split('+')[0]
            text = ''.join(x for x in row[3] if x.isprintable()).replace('\n', ' ')
            text = ftfy.fix_text(text)
            fixed_label = align_language_to_fasttext(label)
            if fixed_label != 'und':
                yield (text, fixed_label)


def wiki_gen():
    with open(corpus_file('WiLI/x_test.txt')) as texts_file:
        with open(corpus_file('WiLI/y_test.txt')) as labels_file:
            for text, label in zip(texts_file, labels_file):
                label = label.rstrip()
                text = ''.join(x for x in text if x.isprintable()).replace('\n', ' ')
                text = ftfy.fix_text(text)
                fixed_label = align_language_to_fasttext(label)
                if fixed_label != 'und':
                    yield (text, fixed_label)


def tatoeba_gen():
    with open(corpus_file('Tatoeba/tatoeba_short_text.txt')) as tatoeba_file:
        for line in tatoeba_file:
            line = line.split('\t')
            text = line[1].replace('\n', ' ')
            text = ftfy.fix_text(text)
            label = line[0]
            fixed_label = align_language_to_fasttext(label)
            if fixed_label != 'und':
                yield (text, fixed_label)
