# `lumi_language_id`

Utilities for reliable-enough language detection.

This is a maintained fork of the original (archived) `lumi-language-id`,
updated to work with numpy 2.x. It is published on PyPI as
**`lumi-language-id-2`**, but the import name stays `lumi_language_id`:

    pip install lumi-language-id-2   # or: uv add lumi-language-id-2

This package wraps fastText's "lid.176" language-detection model with another
classifier, which is trained to produce better probability estimates.  It also
applies text cleaning, so that the text it detects is unaffected by
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
