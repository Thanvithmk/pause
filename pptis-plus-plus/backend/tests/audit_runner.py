import os
from collections import Counter
from unittest.mock import patch

from modules.decision_engine import DecisionEngine
from modules.risk_calculation import RiskCalculator
from modules.toxicity_model import ToxicityAnalyzer

LABELS = ("toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate")
ORDER = ("ALLOW", "WARN", "REWRITE", "DELAYED-SEND")


def classify_text(text, context="", previous_vector=None):
    analyzer = classify_text._analyzer
    scores = analyzer.analyze(text, context=context)
    calc = classify_text._calc
    pacl = calc.calculate_pacl(scores["toxic"], 0.0)
    if previous_vector is None:
        cel = 0.0
    else:
        current = [float(scores[label]) for label in LABELS]
        delta = calc.calculate_delta_x(current, previous_vector)
        cel = calc.calculate_cel(delta)
    p_esc = calc.calculate_escalation_probability(cel, pacl)
    final_risk = calc.calculate_final_risk(pacl, p_esc)
    decision = classify_text._engine.make_decision(
        final_risk, toxicity_vector=scores
    )["decision"]
    return {
        "decision": decision,
        "risk_score": final_risk,
        "pacl": pacl,
        "cel": cel,
        "p_esc": p_esc,
        "toxicity": scores,
        "threat": scores.get("threat", 0.0),
    }


def _init():
    classify_text._analyzer = ToxicityAnalyzer()
    with patch("modules.risk_calculation.BehaviorProfileManager"):
        calc = RiskCalculator(lambda_const=0.5)
    calc.b, calc.w_E, calc.w_P, calc.alpha = 0.0, 1.0, 1.0, 0.5
    classify_text._calc = calc
    with patch("modules.decision_engine.RiskCalculator"), patch.object(
        DecisionEngine, "_init_rewrite_engine", return_value=None
    ):
        engine = DecisionEngine()
    engine.T_allow, engine.T_warn, engine.T_rewrite = 0.30, 0.55, 0.80
    classify_text._engine = engine


_init()


def evaluate(cases):
    rows = []
    for case in cases:
        result = classify_text(case["message"], context=case.get("context") or "")
        actual = result["decision"]
        expected = case["expected"]
        rows.append({**case, **result, "actual": actual, "pass": actual == expected})
    return rows


def metrics(rows):
    total = len(rows)
    correct = sum(1 for row in rows if row["pass"])
    confusion = {gold: Counter() for gold in ORDER}
    for row in rows:
        confusion[row["expected"]][row["actual"]] += 1
    by_label = {}
    for label in ORDER:
        gold_pos = [row for row in rows if row["expected"] == label]
        pred_pos = [row for row in rows if row["actual"] == label]
        tp = sum(1 for row in gold_pos if row["actual"] == label)
        fp = sum(1 for row in pred_pos if row["expected"] != label)
        fn = len(gold_pos) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        by_label[label] = {
            "n": len(gold_pos),
            "accuracy": tp / len(gold_pos) if gold_pos else 0.0,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "fp": fp,
            "fn": fn,
        }
    # false positive: predicted stricter than gold; false negative: predicted weaker
    rank = {name: index for index, name in enumerate(ORDER)}
    fp_rows = [row for row in rows if rank[row["actual"]] > rank[row["expected"]]]
    fn_rows = [row for row in rows if rank[row["actual"]] < rank[row["expected"]]]
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_label": by_label,
        "confusion": {gold: dict(counter) for gold, counter in confusion.items()},
        "false_positives": fp_rows,
        "false_negatives": fn_rows,
        "fp_rate": len(fp_rows) / total if total else 0.0,
        "fn_rate": len(fn_rows) / total if total else 0.0,
    }
