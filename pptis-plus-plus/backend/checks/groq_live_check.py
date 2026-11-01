"""Live Groq call. This file is not part of the default unit suite.

Inference: actual HTTP request to llama-3.1-8b-instant.
A failed response is printed and exits non-zero. No substitute model is called.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.rewrite_engine import REQUIRED_GROQ_MODEL, RewriteEngine, RewriteServiceError


def main():
    os.environ["FLASK_ENV"] = "testing"
    engine = RewriteEngine()
    print(f"inference=actual model={engine.groq_model}")
    if engine.groq_model != REQUIRED_GROQ_MODEL:
        print(f"refusing model {engine.groq_model}")
        return 1
    try:
        text = engine.generate_candidate_rewrite(
            "This whole plan is trash. Let's start over.",
            [],
        )
    except RewriteServiceError as exc:
        print(f"rewrite failed: {exc}")
        return 1
    if not text or "trash" in text.lower():
        print("rewrite did not return a distinct suggestion")
        return 1
    print("rewrite succeeded")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
