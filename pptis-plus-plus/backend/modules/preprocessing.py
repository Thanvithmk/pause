import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

CONTRACTIONS = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "\U00002B50"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"\w+(?:'\w+)?|[!?]+|[^\w\s]", re.UNICODE)
PUNCTUATION_RUN_PATTERN = re.compile(r"[!?]{2,}|[!?]")


class Preprocessor:
    def __init__(self):
        self.contractions = dict(CONTRACTIONS)
        self.emoji_pattern = EMOJI_PATTERN
        self.url_pattern = URL_PATTERN
        self.mention_pattern = MENTION_PATTERN
        self.hashtag_pattern = HASHTAG_PATTERN
        self.whitespace_pattern = WHITESPACE_PATTERN
        self.token_pattern = TOKEN_PATTERN
        logger.info("Preprocessor initialized with %s contractions", len(self.contractions))

    def preprocess(self, text):
        original_text = "" if text is None else str(text)
        logger.info("Preprocessing text length=%s", len(original_text))

        emoji_list = self.emoji_pattern.findall(original_text)
        urls = self.url_pattern.findall(original_text)
        mentions = self.mention_pattern.findall(original_text)
        hashtags = self.hashtag_pattern.findall(original_text)
        contractions_found = self._find_contractions(original_text)
        punctuation_marks = self._count_punctuation(original_text)

        cleaned = original_text
        cleaned = self.url_pattern.sub("[URL]", cleaned)
        cleaned = self.mention_pattern.sub("[MENTION]", cleaned)
        cleaned = self.hashtag_pattern.sub("[HASHTAG]", cleaned)
        cleaned = self.whitespace_pattern.sub(" ", cleaned).strip()

        tokens = self.tokenize(cleaned)
        preserved_info = {
            "emojis": True,
            "contractions": True,
            "meaningful_punctuation": True,
            "sentence_structure": True,
            "contractions_found": contractions_found,
            "urls": urls,
            "mentions": mentions,
            "hashtags": hashtags,
        }

        logger.info(
            "Preprocessing complete emojis=%s urls=%s mentions=%s hashtags=%s tokens=%s",
            len(emoji_list),
            len(urls),
            len(mentions),
            len(hashtags),
            len(tokens),
        )

        return {
            "original_text": original_text,
            "cleaned_text": cleaned,
            "emoji_list": emoji_list,
            "punctuation_marks": punctuation_marks,
            "tokens": tokens,
            "preserved_info": preserved_info,
        }

    def tokenize(self, text):
        if not text:
            return []
        return self.token_pattern.findall(text)

    def denormalize_for_output(self, text, preserved_info=None):
        if not text:
            return ""
        preserved_info = preserved_info or {}
        display = text
        display = self._restore_placeholders(display, "[URL]", preserved_info.get("urls", []))
        display = self._restore_placeholders(display, "[MENTION]", preserved_info.get("mentions", []))
        display = self._restore_placeholders(display, "[HASHTAG]", preserved_info.get("hashtags", []))
        display = self.whitespace_pattern.sub(" ", display).strip()
        logger.info("Denormalized text for display")
        return display

    def _restore_placeholders(self, text, placeholder, originals):
        result = text
        for original in originals:
            result = result.replace(placeholder, original, 1)
        return result

    def _find_contractions(self, text):
        found = []
        lowered = text.lower()
        for contraction in self.contractions:
            if re.search(r"\b" + re.escape(contraction) + r"\b", lowered):
                found.append(contraction)
        return found

    def _count_punctuation(self, text):
        marks = self.emoji_pattern.sub("", text)
        exclamation = marks.count("!")
        question = marks.count("?")
        runs = PUNCTUATION_RUN_PATTERN.findall(marks)
        return {
            "exclamation": exclamation,
            "question": question,
            "meaningful_runs": [run for run in runs if len(run) >= 2],
        }
