"""Phrase and word lookup for the Gen Z vocabulary.

This module reports saved meanings. It does not decide slang versus literal,
and it does not score toxicity.
"""

import csv
import os
import re

TOKEN = re.compile(r"\S+")
COLUMNS = ("term", "definition", "example", "context", "category", "synonyms", "source")


def default_vocabulary_path():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "database", "genz_vocabulary.csv")
    )


class SlangRecognizer:
    def __init__(self, path=None):
        self.path = path or default_vocabulary_path()
        self.entries = self._load(self.path)
        self._max_words = max((len(term.split()) for term in self.entries), default=1)

    def recognize(self, text):
        original = "" if text is None else str(text)
        lowered = original.lower().replace("’", "'")
        spans = list(TOKEN.finditer(lowered))
        used = []
        found = []
        total = len(spans)
        for start_index in range(total):
            last = min(self._max_words, total - start_index)
            for size in range(last, 0, -1):
                start = spans[start_index].start()
                end = spans[start_index + size - 1].end()
                term = re.sub(r"\s+", " ", lowered[start:end])
                senses = self.entries.get(term)
                if senses is None or _overlaps(start, end, used):
                    continue
                used.append((start, end))
                found.append(
                    {
                        "term": original[start:end],
                        "start": start,
                        "end": end,
                        "ambiguous": len(senses) > 1,
                        "senses": [dict(sense) for sense in senses[:8]],
                    }
                )
                break
        found.sort(key=lambda item: (item["start"], -(item["end"] - item["start"])))
        return {"terms": found}

    @staticmethod
    def _load(path):
        grouped = {}
        seen = set()
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                term = (row.get("term") or "").strip().lower()
                definition = (row.get("definition") or "").strip()
                if not term or not definition:
                    continue
                sense = {column: (row.get(column) or "").strip() for column in COLUMNS}
                sense["term"] = term
                identity = tuple(sense[column] for column in COLUMNS)
                if identity in seen:
                    continue
                seen.add(identity)
                grouped.setdefault(term, []).append(sense)
        return grouped


def _overlaps(start, end, spans):
    return any(start < span_end and end > span_start for span_start, span_end in spans)
