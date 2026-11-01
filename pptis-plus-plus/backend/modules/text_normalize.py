"""Analysis-only unfolding of leetspeak and light obfuscation.

Original user text is stored unchanged. This helper is used by the safety
layer so k1ll / k!ll still reach threat detectors.
"""

import re

LEET = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
        "!": "i",
        "*": "",
    }
)
EMOJI_WORDS = {
    "👊": " punch ",
    "🔪": " knife ",
    "🔫": " gun ",
    "💣": " bomb ",
}


def unfold_for_analysis(text):
    value = "" if text is None else str(text)
    for emoji, word in EMOJI_WORDS.items():
        value = value.replace(emoji, word)
    value = value.translate(LEET)
    value = re.sub(r"(?<=\w)\.(?=\w)", " ", value)
    value = re.sub(r"(.)\1{2,}", r"\1", value)
    value = re.sub(r"\bfck\b", "fuck", value, flags=re.I)
    value = re.sub(r"\bdum\b", "dumb", value, flags=re.I)
    value = re.sub(r"\bu\b", "you", value, flags=re.I)
    value = re.sub(r"\s+", " ", value)
    return value.strip()
