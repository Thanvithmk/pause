"""Build database/genz_vocabulary.csv from the local dictionary and two Hugging Face sets.

MLBtrio/genz-slang-dataset supplies Slang, Description, Example, and Context.
LM-Lexicon/Slang is inspected as train, valid, and test JSONL. Its fields are
term, definition, context, and synonyms.

Exact duplicate meanings are removed. Different meanings of one term stay.
Ordinary English words and entries without a usable definition are left out.
No row is given a safe or toxic label.
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "database" / "genz_vocabulary.csv"
COLUMNS = ("term", "definition", "example", "context", "category", "synonyms", "source")
WORD_LIST = Path("/usr/share/dict/words")
TERM_RE = re.compile(r"^[a-z0-9](?:[a-z0-9'+./-]*[a-z0-9])?$|^[a-z0-9](?:[a-z0-9'+./-]+ )+[a-z0-9'+./-]*[a-z0-9]$")


def normalize(value):
    text = "" if value is None else str(value)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def normalize_term(value):
    return normalize(value).lower()


def category_for(term):
    if " " not in term and len(term) <= 4 and term.isalpha():
        return "internet abbreviation"
    return "slang"


def english_words():
    words = set()
    if not WORD_LIST.is_file():
        return words
    with WORD_LIST.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            token = line.strip().lower()
            if token:
                words.add(token)
    return words


def accept_lexicon_term(term, definition, words):
    if not TERM_RE.match(term) or not re.search(r"[a-z]", term):
        return "bad_term"
    if len(term) > 40 or len(term.split()) > 6:
        return "bad_term"
    if len(definition) < 20 or len(definition) > 400 or definition.lower() == term:
        return "bad_definition"
    if " " not in term and term in words:
        return "ordinary_english"
    return ""


def add_row(rows, seen, counts, term, definition, example, context, category, synonyms, source):
    term = normalize_term(term)
    definition = normalize(definition)
    if not term or not definition:
        counts["empty"] += 1
        return
    identity = (term, definition.lower())
    if identity in seen:
        counts["duplicate_meanings"] += 1
        return
    seen.add(identity)
    rows.append(
        {
            "term": term,
            "definition": definition,
            "example": normalize(example)[:300],
            "context": normalize(context)[:300],
            "category": category or category_for(term),
            "synonyms": normalize(synonyms),
            "source": source,
        }
    )


def main():
    counts = Counter()
    rows = []
    seen = set()
    words = english_words()

    with OUTPUT.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (row.get("source") or "").strip() != "pptis-slang-dictionary":
                continue
            counts["pptis_read"] += 1
            add_row(
                rows,
                seen,
                counts,
                row.get("term"),
                row.get("definition"),
                row.get("example"),
                row.get("context"),
                row.get("category"),
                row.get("synonyms"),
                "pptis-slang-dictionary",
            )

    mlb_path = hf_hub_download(
        "MLBtrio/genz-slang-dataset", "all_slangs.csv", repo_type="dataset"
    )
    with open(mlb_path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            term = normalize_term(row.get("Slang"))
            counts["mlbtrio_downloaded"] += 1
            if " " not in term and term in words:
                counts["mlbtrio_ordinary_english"] += 1
                continue
            add_row(
                rows,
                seen,
                counts,
                row.get("Slang"),
                row.get("Description"),
                row.get("Example"),
                row.get("Context"),
                category_for(normalize_term(row.get("Slang"))),
                "",
                "MLBtrio/genz-slang-dataset",
            )

    for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        path = hf_hub_download("LM-Lexicon/Slang", name, repo_type="dataset")
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                counts["lexicon_downloaded"] += 1
                counts[f"lexicon_{name}"] += 1
                item = json.loads(line)
                term = normalize_term(item.get("term"))
                definition = normalize(item.get("definition"))
                reason = accept_lexicon_term(term, definition, words)
                if reason:
                    counts[f"lexicon_{reason}"] += 1
                    continue
                synonyms = item.get("synonyms") or []
                if isinstance(synonyms, list):
                    synonyms = "; ".join(str(part) for part in synonyms if str(part).strip())
                add_row(
                    rows,
                    seen,
                    counts,
                    term,
                    definition,
                    "",
                    item.get("context"),
                    category_for(term),
                    synonyms,
                    "LM-Lexicon/Slang",
                )

    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    by_source = Counter(row["source"] for row in rows)
    print("output", OUTPUT)
    print("retained", len(rows))
    print("by_source", dict(by_source))
    for key in sorted(counts):
        print(f"{key} {counts[key]}")


if __name__ == "__main__":
    main()
