# Retraining the tuned classifier

The package ships two models in `lumi_language_id/data/`:

- `lid.176.ftz` -- fastText's language identifier, used as-is;
- `tuned.npz` -- a small classifier that estimates whether fastText's answer is
  right, from four features of the text (log length, fastText's confidence in bits,
  log space count, log Han-character count).

Only `tuned.npz` is retrained. It needs a labelled sample of text, which is built
from FineWeb-2 (175 fastText languages) and FineWeb (English).

## 1. Install the training extras

```sh
uv sync --extra train
```

## 2. Build the training data

```sh
uv run python -m lumi_language_id.fetch_fineweb
```

This streams the Hub and writes a cache under `corpus/fineweb/` (git-ignored,
about 350 MB):

- `<label>.train.jsonl` / `<label>.test.jsonl` -- up to 300 training and 100 test
  documents per fastText label. Test documents come from FineWeb-2's own `test`
  split; English test documents are a held-out tenth of FineWeb, by id hash.
- `sizes.json` -- bytes of text per label on the Hub, used to weight languages.

Each FineWeb-2 config (`cmn_Hani`, `srp_Latn`, ...) is mapped to a fastText label
with `align_to_fasttext`. Configs fastText cannot name, such as romanized Hindi or
Cherokee, are skipped rather than attached to a neighbouring language.

Options:

```sh
uv run python -m lumi_language_id.fetch_fineweb --train 300 --test 100 --workers 8
```

Known issue: a full run has been seen to print its last label line and then not
exit. Every file is complete at that point -- each is written only after all its rows
are collected -- so stop the process. A small run exits normally, and the cause is not
yet known.

The run resumes: a label whose files already exist is skipped, so rerunning after
a network failure fetches only what is missing. Delete `corpus/fineweb/` to start
over. Some rare languages have fewer test documents than requested, because
FineWeb-2's test split holds fewer.

More training documents do not help. Measured on this data, going from 37 to 300
documents per language changed the weighted log loss by 0.003 (0.2570 to 0.2537),
against run-to-run noise of 0.001. More *test* documents do sharpen the
per-language readout, so raise `--test` if you need finer measurement.

## 3. Retrain

```sh
uv run python -m lumi_language_id.build
```

This reads only the local cache. It:

1. turns each document into two rows -- the full text and a short window of 10 to
   500 characters at a random offset -- so both regimes are calibrated;
2. weights each language by `size ** 0.3` (`SIZE_EXPONENT` in `build.py`). With
   uniform weights, Cantonese and Wu, which fastText calls `'zh'`, made up 59% of
   the Han text, and the classifier learned to withhold correct Mandarin answers;
3. fits the classifier (seeded, so a rebuild is reproducible) and **overwrites
   `lumi_language_id/data/tuned.npz`**;
4. prints validation and test accuracy and log loss, then a table by text length
   and script for the shipped classifier next to the retrained one, then per language
   how often fastText is right and how often a right answer is withheld.

The shipped `tuned.npz` is tracked in git; to discard a retrain:

```sh
git checkout lumi_language_id/data/tuned.npz
```

## 4. Check the result

```sh
uv run pytest
uv run python -m doctest lumi_language_id/__init__.py
```

`test_long_chinese_is_kept` uses feature rows from real long Chinese documents
that the previous classifier rejected; it fails on a classifier with that defect.

A held-out split drawn from the same source as the training data will look good by
construction. Before releasing, also check the retrained classifier on text from
somewhere else that you know the language of, and compare it with the previous
release on the same text.
