import numpy as np
import pytest

from lumi_language_id.tuned import TunedLanguageIdentifier


LID = TunedLanguageIdentifier.load()


def _get_language(text):
    lang, prob = LID.detect_language(text)
    return lang


def test_language_id():
    assert _get_language("here's some words") == 'en'
    assert _get_language("aquí hay algunas palabras") == 'es'
    assert _get_language("これらは言葉です") == 'ja'
    assert _get_language("这些是单词") == 'zh'


# Feature rows (cleaned length, fastText bits of information, spaces, Han characters)
# traced from real long Chinese documents that fastText identified correctly at 0.994 and
# 0.997. The shipped classifier withheld both as 'und' (probability 0.312 and 0.045),
# extrapolating past the tweets and Wikipedia introductions it was trained on.
LONG_CHINESE_ROWS = [
    [690.0, 7.47, 4.0, 631.0],
    [1203.0, 8.21, 64.0, 1002.0],
]


@pytest.mark.parametrize('row', LONG_CHINESE_ROWS)
def test_long_chinese_is_kept(row):
    counts = np.log1p([row[0], row[2], row[3]])
    zh_size = np.log1p(LID.language_identifier.language_sizes['zh'])
    features = np.array([counts[0], row[1], counts[1], counts[2], zh_size])
    assert LID.tuned_classifier.probability(features) >= 0.5


@pytest.mark.parametrize('text', [
    '女性艺人雪梨于昨天下午去世,生而为人请务必善良',
    '事发学校校长朱某已被停职检查。',
])
def test_kana_free_simplified_chinese_is_not_japanese(text):
    # fastText's raw answer on both is 'ja' (0.645 and 0.752).
    assert LID.language_identifier.detect_language(text)[0] == 'zh'


@pytest.mark.parametrize('text', ['日本銀行勤務', '東京大学大学院教育', 'これらは言葉です'])
def test_japanese_default_holds(text):
    # Kanji shared with Chinese, or any kana, never overrides a Japanese answer.
    assert LID.language_identifier.detect_language(text)[0] == 'ja'
