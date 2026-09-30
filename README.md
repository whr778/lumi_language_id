# `lumi_language_id`

Utilities for reliable-enough language detection.

This is a maintained fork of the original (archived) `lumi-language-id`,
updated to work with numpy 2.x. It is published on PyPI as
**`lumi-language-id-2`**, but the import name stays `lumi_language_id`:

    pip install lumi-language-id-2   # or: uv add lumi-language-id-2

This package wraps fastText's "lid.176" language-detection model with another
classifier, which is trained to produce better probability estimates. When that
probability is below 0.5 the answer is `'und'` rather than a probable mistake. It
also applies text cleaning, so that the text it detects is unaffected by
punctuation, digits, or emoji.

Example:

    >>> from lumi_language_id.tuned import TunedLanguageIdentifier
    >>> lid = TunedLanguageIdentifier.load()
    >>> lang, _prob = lid.detect_language("these are words")
    >>> lang
    'en'

    >>> lang, _prob = lid.detect_language("aquí hay algunas palabras")
    >>> lang
    'es'

## Changes in 3.0.0

- **Long Chinese and Japanese text is no longer rejected.** The original classifier
  was trained on tweets and Wikipedia introductions and withheld fastText's
  correct answer on long, spaceless CJK text: 47.7% of Mandarin web documents
  and 64.8% of one Chinese financial corpus came back `'und'`. It is now trained
  on FineWeb-2 and FineWeb, in full documents and short windows, with each
  language weighted by how much text it has on the web.
- **Kana-free simplified Chinese is no longer called Japanese.** fastText labels
  some short Chinese sentences `'ja'`. A `'ja'` answer is replaced by the most
  likely Chinese label only when the text has no kana *and* contains a character
  that does not exist in Japanese (`艺`, `发`, but not shared kanji such as `京`),
  so kanji-only Japanese still reads as Japanese.
- **`align_to_fasttext(language, script)`** maps a language code to the fastText
  label for exactly that language, or `'und'`. The older
  `align_language_to_fasttext` accepts a nearby language, so Cherokee and
  Hawaiian map to `'en'`.
- **The classifier knows which language fastText predicted.** Its features now
  include the web size of the predicted language, so a `'zh'` answer on Mandarin
  (right 98% of the time) is no longer judged like one on Cantonese (right 2%).
- **Cost:** on short text (10 to 500 characters) in common Latin-script languages,
  a correct answer is withheld as `'und'` somewhat more often (English 2.6% to
  5.1%). Full documents are unaffected. [LANGUAGES.md](LANGUAGES.md) gives the
  measured behaviour for every language.
- **Breaking:** the saved classifier is format version 2 (five features, counts
  log-scaled), and version 1 files are refused rather than silently misread. Some
  inputs get a different label than in 2.0.0 (kana-free Chinese, above).

See [LANGUAGES.md](LANGUAGES.md) for per-language accuracy, [TRAINING.md](TRAINING.md) to
rebuild the classifier, and
[INSTRUCTIONS.md](INSTRUCTIONS.md) for development setup and publishing.

## Data

The classifier is trained on samples of
[FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) and
[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (English), both
released by Hugging Face under the ODC-By 1.0 license. The training text itself is
not redistributed; only the fitted classifier weights ship with the package.
