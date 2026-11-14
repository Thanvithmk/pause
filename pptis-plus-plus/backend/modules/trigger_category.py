"""Ask Groq for the topic of a stopped message.

The message text is sent on every request. The reply is a topic, never an
intervention name. This does not change the risk score.
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

CATEGORIES = (
    "Community",
    "Insult",
    "Harassment",
    "Profanity",
    "Harm",
    "Self-harm",
)
_LINE = re.compile(r"^\s*(\d+)[.)]\s*(.+?)\s*$")
_INTERVENTION = re.compile(r"\b(allow|warn|rewrite|delayed[-\s]?send)\b", re.I)


def categorize_text(text):
    found = categorize_many([("current", text)])
    return found.get("current")


BATCH = 8


def categorize_many(items):
    """Ask Groq for one topic per (key, message) pair. Keys with no reply are omitted."""
    pending = []
    for key, text in items:
        message = " ".join(str(text or "").split())[:400]
        if message:
            pending.append((key, message))
    if not pending or not _calls_allowed():
        return {}
    found = {}
    for start in range(0, len(pending), BATCH):
        chunk = pending[start:start + BATCH]
        try:
            found.update(_categorize_chunk(chunk))
        except Exception:
            logger.exception("Trigger category batch failed")
    return found


def _categorize_chunk(pending):
    lines = [f"{index}. {message}" for index, (_key, message) in enumerate(pending, start=1)]
    prompt = (
        "Each numbered line is one message.\n"
        f"{chr(10).join(lines)}\n\n"
        "Name the topic of each message. Use only one of these topics:\n"
        "Community: an attack on a group, identity, caste, race, religion, or community.\n"
        "Insult: a personal insult or name-calling.\n"
        "Harassment: bullying, pestering, or targeting someone without a direct threat.\n"
        "Profanity: obscene language that is not aimed as an insult or a threat.\n"
        "Harm: a threat, violence, or intent to hurt someone else.\n"
        "Self-harm: intent to hurt oneself.\n"
        "Do not answer ALLOW, WARN, REWRITE, or DELAYED-SEND.\n"
        "Reply with one line per item, in the form: 1. Harm"
    )
    categories = _parse_reply(_complete(prompt), len(pending))
    return {
        pending[index - 1][0]: category
        for index, category in categories.items()
    }


def _calls_allowed():
    return os.getenv("FLASK_ENV") != "testing"


def _parse_reply(reply, count):
    found = {}
    for line in (reply or "").splitlines():
        match = _LINE.match(line.strip())
        if match is None:
            continue
        index = int(match.group(1))
        if index < 1 or index > count or index in found:
            continue
        name = _category_name(match.group(2))
        if name:
            found[index] = name
    if count == 1 and 1 not in found:
        name = _category_name(reply or "")
        if name and _mentions_category(reply or ""):
            found[1] = name
    return found


def _mentions_category(text):
    lowered = text.lower()
    return any(category.lower() in lowered for category in CATEGORIES)


def _category_name(text):
    cleaned = _INTERVENTION.sub(" ", text or "")
    lowered = cleaned.strip().lower()
    for category in sorted(CATEGORIES, key=len, reverse=True):
        if category.lower() in lowered:
            return category
    return None


def _complete(prompt):
    from modules.rewrite_engine import REQUIRED_GROQ_MODEL, RewriteServiceError, _setting

    api_key = _setting("GROQ_API_KEY")
    if not api_key:
        raise RewriteServiceError("trigger category: GROQ_API_KEY is not configured")
    configured = _setting("GROQ_MODEL") or REQUIRED_GROQ_MODEL
    if configured != REQUIRED_GROQ_MODEL:
        raise RewriteServiceError(
            f"trigger category: GROQ_MODEL must be {REQUIRED_GROQ_MODEL}, not {configured}"
        )
    try:
        from groq import Groq

        response = Groq(api_key=api_key).chat.completions.create(
            model=REQUIRED_GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You name the topic of a message as Community, Insult, Harassment, Profanity, Harm, or Self-harm. "
                        "Never answer with ALLOW, WARN, REWRITE, or DELAYED-SEND. Do not explain."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=512,
            extra_body={"reasoning_effort": "low"},
        )
    except RewriteServiceError:
        raise
    except Exception as exc:
        logger.exception("Trigger category request failed")
        raise RewriteServiceError(
            f"trigger category: Groq request failed for {REQUIRED_GROQ_MODEL}"
        ) from exc
    return (response.choices[0].message.content or "").strip()
