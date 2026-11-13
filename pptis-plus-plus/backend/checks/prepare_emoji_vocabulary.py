"""Build the emoji vocabulary and the TweetEval emoji usage table.

badrex/LLM-generated-emoji-descriptions supplies the dictionary.
cardiffnlp/tweet_eval, config emoji, is stored separately. Its labels are
emoji classes from social-media posts. They are not toxicity labels.
"""

import csv
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
VOCABULARY = ROOT / "database" / "emoji_vocabulary.csv"
USAGE = ROOT / "database" / "tweet_eval_emoji.csv"
EMOJI_LABELS = {
    0: "❤",
    1: "😍",
    2: "😂",
    3: "💕",
    4: "🔥",
    5: "😊",
    6: "😎",
    7: "✨",
    8: "💙",
    9: "😘",
    10: "📷",
    11: "🇺🇸",
    12: "☀",
    13: "💜",
    14: "😉",
    15: "💯",
    16: "😁",
    17: "🎄",
    18: "📸",
    19: "😜",
}


def tags_to_text(value):
    if value is None:
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return "|".join(str(part).strip() for part in value if str(part).strip())
    return str(value).strip()


def write_vocabulary():
    path = hf_hub_download(
        "badrex/LLM-generated-emoji-descriptions",
        "data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    frame = pd.read_parquet(path)
    rows = []
    seen = set()
    for item in frame.to_dict(orient="records"):
        emoji = str(item.get("character") or "").strip()
        if not emoji or emoji in seen:
            continue
        seen.add(emoji)
        rows.append(
            {
                "emoji": emoji,
                "unicode": str(item.get("unicode") or "").strip(),
                "description": str(item.get("short description") or "").strip(),
                "tags": tags_to_text(item.get("tags")),
                "meaning": str(item.get("LLM description") or "").strip(),
                "source": "badrex/LLM-generated-emoji-descriptions",
            }
        )
    with VOCABULARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("emoji", "unicode", "description", "tags", "meaning", "source"),
        )
        writer.writeheader()
        writer.writerows(rows)
    print("vocabulary", len(rows), "downloaded", len(frame))
    return {row["emoji"] for row in rows}


def write_usage():
    counts = Counter()
    rows = []
    for split in ("train", "validation", "test"):
        path = hf_hub_download(
            "cardiffnlp/tweet_eval",
            f"emoji/{split}-00000-of-00001.parquet",
            repo_type="dataset",
        )
        frame = pd.read_parquet(path)
        for item in frame.to_dict(orient="records"):
            label = int(item["label"])
            counts[label] += 1
            rows.append(
                {
                    "split": split,
                    "text": item.get("text") or "",
                    "label_id": label,
                    "emoji": EMOJI_LABELS.get(label, ""),
                    "label_kind": "emoji_class",
                }
            )
        print(split, len(frame), list(frame.columns))
    with USAGE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("split", "text", "label_id", "emoji", "label_kind"),
        )
        writer.writeheader()
        writer.writerows(rows)
    print("usage_rows", len(rows))
    print("label_counts", dict(sorted(counts.items())))


if __name__ == "__main__":
    write_vocabulary()
    write_usage()
