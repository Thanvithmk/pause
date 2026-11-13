"""Run slang and emoji lookups together.

The two lookups read the same message and do not wait on each other.
Their notes are features for the analysis details. They do not change
the risk score.
"""

from concurrent.futures import ThreadPoolExecutor


def recognize_in_parallel(text, slang_recognizer, emoji_recognizer):
    with ThreadPoolExecutor(max_workers=2) as pool:
        slang_job = pool.submit(slang_recognizer.recognize, text)
        emoji_job = pool.submit(emoji_recognizer.recognize, text)
        slang_features = slang_job.result()
        emoji_features = emoji_job.result()
    return {"slang": slang_features, "emoji": emoji_features}


LAUGHTER_TAGS = {"laughing", "humor", "laughter"}
WEAPON_EMOJI = {"🔪", "🔫", "💣"}


def has_playful_laughter(emoji_features):
    """True when the emoji vocabulary marks laughter and no weapon emoji is present."""
    emojis = (emoji_features or {}).get("emojis") or []
    laughter = False
    for item in emojis:
        character = item.get("emoji") or ""
        if any(weapon in character for weapon in WEAPON_EMOJI):
            return False
        tags = {str(tag).strip().lower() for tag in (item.get("tags") or [])}
        if tags & LAUGHTER_TAGS:
            laughter = True
    return laughter


def emoji_annotations(emoji_features):
    """Short notes for the classifier. The emoji characters stay in the message."""
    notes = []
    for item in (emoji_features or {}).get("emojis") or []:
        description = str(item.get("description") or "").strip()
        character = item.get("emoji") or ""
        if not description or not character:
            continue
        notes.append(f"[Emoji meaning: {character} {description}]")
    return notes


def annotate_slang_usage(slang_features, detections):
    """Copy the existing slang-head label onto vocabulary matches.

    The head still decides slang versus literal. This only records that
    label beside the saved meanings.
    """
    by_term = {}
    for item in detections or []:
        key = (item.get("term") or "").strip().lower()
        if key and key not in by_term:
            by_term[key] = item
    for term in (slang_features or {}).get("terms") or []:
        match = by_term.get((term.get("term") or "").strip().lower())
        if match is None:
            term["usage"] = None
            term["confidence"] = None
            continue
        term["usage"] = match.get("usage")
        term["confidence"] = match.get("confidence")
    return slang_features
