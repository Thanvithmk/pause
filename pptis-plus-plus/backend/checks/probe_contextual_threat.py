"""Score contrast sentences with the loaded Toxic-BERT model.

Prints the six sigmoid probabilities, the safety-pattern label, and the
risk that the locked formula produces for a new user (BT=0, E=0).
Does not write messages.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.safety_patterns import SafetyPatternAnalyzer
from modules.toxicity_model import ToxicityAnalyzer

PROBES = (
    "I will gently cut your nails.",
    "I will cut your hair.",
    "I will trim your nails.",
    "I will cut your head.",
    "I will cut you.",
    "I will cut the vegetables.",
    "I will cut the rope.",
    "I will hurt you.",
    "I will help you.",
    "I will clean your nails.",
)


def risk_for_new_user(toxic):
    pacl = min(1.0, max(0.0, toxic + 0.5 * toxic))
    p_esc = 1.0 / (1.0 + math.exp(-pacl))
    return 0.5 * pacl + 0.5 * p_esc


def band(score):
    if score < 0.30:
        return "ALLOW"
    if score < 0.55:
        return "WARN"
    if score < 0.80:
        return "REWRITE"
    return "DELAYED-SEND"


def main():
    analyzer = ToxicityAnalyzer()
    safety = SafetyPatternAnalyzer()
    print("id2label", analyzer.id2label)
    focus = (
        "I will gently cut your nails.",
        "I will cut your head.",
        "I will cut the vegetables.",
    )
    for text in focus:
        encoded = analyzer.tokenizer(text, add_special_tokens=True)
        tokens = analyzer.tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        print("TOKENS", text, tokens)
    details = analyzer.inspect_many(PROBES)
    for text, detail in zip(PROBES, details):
        scores = detail["model"]
        label = safety.score(text)["label"]
        final = risk_for_new_user(scores["toxic"])
        print(
            f"{text}\t"
            + "\t".join(f"{name}={scores[name]:.4f}" for name in (
                "toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate"
            ))
            + f"\tR={final:.4f}\tmath={band(final)}\tsafety={label}"
        )


if __name__ == "__main__":
    main()
