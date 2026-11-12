import csv
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.slang_layer import SlangInterpreter, probabilities_are_usable, target_char_span

BACKEND = Path(__file__).resolve().parents[1]
AUDIT_CSV = BACKEND / "data" / "slang_independent_audit.csv"
TRAIN_CSV = BACKEND / "data" / "slang_context_labels.csv"
METRICS_PATH = BACKEND / "models" / "slang_independent_audit.json"
PRIMARY = "xlm-roberta-classifier"
INFERENCE = "actual"


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metrics(rows):
    labels = ("literal", "slang")
    matrix = {gold: {pred: 0 for pred in ("literal", "slang", "uncertain", "unclassified")} for gold in labels}
    for row in rows:
        matrix[row["gold"]][row["predicted"]] += 1
    scored = [row for row in rows if row["predicted"] in labels]

    def prf(label):
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][pred] for pred in matrix[label] if pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[label].values()),
        }

    per_label = {label: prf(label) for label in labels}
    macro_f1 = sum(per_label[label]["f1"] for label in labels) / 2
    correct = sum(matrix[label][label] for label in labels)
    total = sum(sum(bucket.values()) for bucket in matrix.values())
    uncertain = sum(bucket["uncertain"] for bucket in matrix.values())
    confident_total = len(scored)
    confident_correct = sum(1 for row in scored if row["predicted"] == row["gold"])
    return {
        "confusion_matrix": {
            "rows_are": "actual",
            "columns_are": ["literal", "slang", "uncertain", "unclassified"],
            "labels": list(labels),
            "values": [[matrix[gold][pred] for pred in ("literal", "slang", "uncertain", "unclassified")] for gold in labels],
        },
        "literal": per_label["literal"],
        "slang": per_label["slang"],
        "macro_f1": macro_f1,
        "accuracy": (correct / total) if total else 0.0,
        "confident_accuracy": (confident_correct / confident_total) if confident_total else 0.0,
        "uncertain_rate": (uncertain / total) if total else 0.0,
        "count": total,
    }


def _by_category(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["category"], []).append(row)
    return {name: _metrics(items) for name, items in sorted(grouped.items())}


class PrimarySlangPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._previous_threshold = os.environ.get("SLANG_THRESHOLD")
        os.environ.pop("SLANG_THRESHOLD", None)
        cls.interpreter = SlangInterpreter()
        cls.inference = INFERENCE
        cls.audit_rows = _read_csv(AUDIT_CSV)
        cls.train_sentences = {row["sentence"] for row in _read_csv(TRAIN_CSV)}

    @classmethod
    def tearDownClass(cls):
        if cls._previous_threshold is None:
            os.environ.pop("SLANG_THRESHOLD", None)
        else:
            os.environ["SLANG_THRESHOLD"] = cls._previous_threshold

    def test_primary_checkpoint_is_loaded(self):
        status = self.interpreter.runtime_status()
        self.assertEqual(self.inference, "actual")
        self.assertEqual(status["inference"], "actual")
        self.assertTrue(status["trained_head"])
        self.assertEqual(status["encoder_name"], "xlm-roberta-base")
        self.assertTrue(str(status["classifier_path"]).endswith("slang_literal_head.pt"))
        self.assertAlmostEqual(status["transformer_threshold"], 0.58, places=2)

    def test_primary_probabilities_and_source(self):
        verdict = self.interpreter.disambiguate_slang_literal("cooked", "The night shift left the new interns cooked.")
        self.assertEqual(verdict["source"], PRIMARY)
        self.assertTrue(probabilities_are_usable(verdict["p_literal"], verdict["p_slang"]))
        self.assertAlmostEqual(verdict["p_slang"] + verdict["p_literal"], 1.0, places=5)
        self.assertAlmostEqual(verdict["threshold"], 0.58, places=2)

    def test_target_span_uses_the_word_inside_the_sentence(self):
        sentence = "The closing set at the college fest was pure fire."
        span = target_char_span(sentence, "fire")
        self.assertIsNotNone(span)
        self.assertEqual(sentence[span[0] : span[1]].lower(), "fire")
        self.assertNotEqual(sentence[:15].lower().find("fire"), 0)

    def test_classification_context_is_the_current_sentence_only(self):
        seen = []
        original = self.interpreter.disambiguate_slang_literal

        def record(term, context):
            seen.append(context)
            return original(term, context)

        sentence = (
            "After the long review, the quiet corridor, and the missed bus, "
            "the launch note still felt cooked to everyone waiting."
        )
        self.interpreter.disambiguate_slang_literal = record
        try:
            self.interpreter.interpret(sentence)
        finally:
            self.interpreter.disambiguate_slang_literal = original
        self.assertTrue(seen)
        self.assertTrue(all(item == sentence for item in seen))
        self.assertNotIn("yesterday someone else said", seen[0].lower())

    def test_original_text_stays_and_only_slang_is_enriched(self):
        slang = self.interpreter.interpret("Bro you're cooked 💀")
        literal = self.interpreter.interpret("The building caught fire.")
        self.assertEqual(slang["detections"][0]["usage"], "slang")
        self.assertEqual(slang["detections"][0]["source"], PRIMARY)
        self.assertTrue(slang["enriched_text_with_context"].startswith("Bro you're cooked 💀"))
        self.assertIn("[Contextual meaning:", slang["enriched_text_with_context"])
        self.assertNotIn("Bro you're defeated", slang["enriched_text_with_context"])
        self.assertEqual(literal["detections"][0]["usage"], "literal")
        self.assertEqual(literal["enriched_text_with_context"], "The building caught fire.")
        self.assertIsNone(literal["detections"][0]["meaning"])

    def test_three_outcomes_are_distinct(self):
        original = self.interpreter.disambiguate_slang_literal

        def scripted(term, context):
            mapping = {
                "slangcase": (0.20, 0.80),
                "literalcase": (0.80, 0.20),
                "uncertaincase": (0.45, 0.55),
            }
            p_literal, p_slang = mapping[term]
            self.interpreter._transformer_probabilities = lambda _term, _context: (p_literal, p_slang)
            return original(term, context)

        self.interpreter.disambiguate_slang_literal = scripted
        try:
            slang = self.interpreter.disambiguate_slang_literal("slangcase", "placeholder slangcase here")
            literal = self.interpreter.disambiguate_slang_literal("literalcase", "placeholder literalcase here")
            uncertain = self.interpreter.disambiguate_slang_literal("uncertaincase", "placeholder uncertaincase here")
        finally:
            self.interpreter.disambiguate_slang_literal = original
            if "_transformer_probabilities" in self.interpreter.__dict__:
                del self.interpreter._transformer_probabilities
        self.assertEqual(slang["classification"], "slang")
        self.assertEqual(slang["source"], PRIMARY)
        self.assertEqual(literal["classification"], "literal")
        self.assertEqual(uncertain["classification"], "uncertain")
        self.assertNotEqual(uncertain["classification"], "literal")
        enriched = self.interpreter.enrich_with_context(
            "placeholder uncertaincase here",
            [{
                "term": "uncertaincase",
                "pos": 12,
                "classification": "uncertain",
                "p_slang": 0.55,
                "p_literal": 0.45,
                "threshold": 0.58,
            }],
        )
        self.assertEqual(enriched["enriched_text_with_context"], "placeholder uncertaincase here")
        self.assertNotIn("meaning", json.dumps(enriched["unknown_slang"]))

    def test_unknown_term_is_separate_from_dictionary_candidates(self):
        sentence = "Bro is completely glorped after the rehearsal."
        candidates = self.interpreter.detect_candidates(sentence)
        self.assertNotIn("glorped", [item["term"].lower() for item in candidates])
        result = self.interpreter.interpret(sentence)
        self.assertEqual(result["original_text"], sentence)
        self.assertEqual(result["enriched_text_with_context"], sentence)
        unknown = [item for item in result["unknown_slang"] if item["term"].lower() == "glorped"]
        self.assertTrue(unknown)
        self.assertEqual(unknown[0]["label"], "UNKNOWN_SLANG")
        detection = next(item for item in result["detections"] if item["term"].lower() == "glorped")
        self.assertEqual(detection["usage"], "unknown")
        self.assertIsNone(detection["confidence"])
        self.assertIsNone(detection["meaning"])
        self.assertNotIn("toxic", detection)
        enemies = self.interpreter.interpret("I have enemies.")
        self.assertFalse(any(item.get("usage") == "unknown" for item in enemies["detections"]))
        self.assertEqual(enemies["enriched_text_with_context"], "I have enemies.")

    def test_failure_modes_do_not_invent_probabilities(self):
        from modules.slang_layer import SlangModelError

        encoder = self.interpreter.encoder
        trained = self.interpreter._use_trained_head
        try:
            self.interpreter.encoder = None
            with self.assertRaises(SlangModelError):
                self.interpreter.disambiguate_slang_literal(
                    "cooked", "The night shift left the new interns cooked."
                )
        finally:
            self.interpreter.encoder = encoder
            self.interpreter._use_trained_head = trained

        self.interpreter._transformer_probabilities = lambda term, context: (float("nan"), 0.4)
        try:
            with self.assertRaises(SlangModelError):
                self.interpreter.disambiguate_slang_literal(
                    "cooked", "The night shift left the new interns cooked."
                )
        finally:
            del self.interpreter._transformer_probabilities

    def test_missing_dictionary_and_missing_meaning(self):
        snapshot = dict(self.interpreter.slang_dictionary)
        saved_terms = self.interpreter._terms_by_length
        try:
            self.interpreter.slang_dictionary = {}
            self.interpreter._terms_by_length = ()
            result = self.interpreter.interpret("Bro is completely glorped after the rehearsal.")
            self.assertEqual(result["enriched_text_with_context"], result["original_text"])
            self.assertIn("glorped", [item["term"].lower() for item in result["unknown_slang"]])
        finally:
            self.interpreter.slang_dictionary = dict(snapshot)
            self.interpreter._terms_by_length = saved_terms

        self.interpreter.slang_dictionary.pop("cooked", None)
        try:
            missing = self.interpreter.enrich_with_context(
                "The night shift left the new interns cooked.",
                [{
                    "term": "cooked",
                    "pos": 36,
                    "classification": "slang",
                    "p_slang": 0.91,
                    "p_literal": 0.09,
                    "threshold": 0.58,
                }],
            )
        finally:
            self.interpreter.slang_dictionary = dict(snapshot)
        self.assertEqual(
            missing["enriched_text_with_context"],
            "The night shift left the new interns cooked.",
        )
        self.assertTrue(missing["unknown_slang"])

    def test_independent_audit_set_and_architecture_locks(self):
        self.assertGreaterEqual(len(self.audit_rows), 150)
        overlap = {row["sentence"] for row in self.audit_rows} & self.train_sentences
        self.assertFalse(overlap)
        from config import Config

        self.assertEqual(Config.PACL_LAMBDA, 0.5)
        self.assertEqual(Config.FINAL_RISK_ALPHA, 0.5)
        from api.routes import run_analysis_pipeline
        import inspect

        source = inspect.getsource(run_analysis_pipeline)
        self.assertLess(source.index("slang.interpret"), source.index("toxicity.inspect"))

    def test_score_independent_set_without_refitting(self):
        classified = []
        unknown_hits = 0
        unknown_total = 0
        for row in self.audit_rows:
            if row["label"] == "unknown":
                unknown_total += 1
                result = self.interpreter.interpret(row["sentence"])
                terms = [item["term"].lower() for item in result["unknown_slang"]]
                if row["target"].lower() in terms and result["enriched_text_with_context"] == row["sentence"]:
                    unknown_hits += 1
                continue
            primary = self.interpreter.disambiguate_slang_literal(row["target"], row["sentence"])
            self.assertEqual(primary["source"], PRIMARY)
            self.assertEqual(self.inference, "actual")
            classified.append((row, primary))

        def pack():
            packed = []
            for row, primary in classified:
                verdict = primary
                packed.append(
                    {
                        "category": row["category"],
                        "sentence": row["sentence"],
                        "target": row["target"],
                        "gold": row["label"],
                        "predicted": verdict["classification"],
                        "p_slang": verdict["p_slang"],
                        "p_literal": verdict["p_literal"],
                        "source": verdict["source"],
                    }
                )
            return packed

        primary_rows = pack()
        report = {
            "encoder_name": "xlm-roberta-base",
            "checkpoint": "slang_literal_head.pt",
            "threshold": 0.58,
            "trained_on_this_file": False,
            "threshold_refit": False,
            "audit_rows": len(self.audit_rows),
            "classified_rows": len(classified),
            "unknown_rows": unknown_total,
            "unknown_detected": unknown_hits,
            "unknown_detection_rate": (unknown_hits / unknown_total) if unknown_total else 0.0,
            "primary": {
                "source": PRIMARY,
                "overall": _metrics(primary_rows),
                "by_category": _by_category(primary_rows),
            },
        }
        METRICS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        self.assertGreater(report["classified_rows"], 0)
        self.assertEqual(report["primary"]["overall"]["count"], len(classified))
        for row in primary_rows:
            if row["p_slang"] is not None:
                self.assertAlmostEqual(row["p_slang"] + row["p_literal"], 1.0, places=5)
                self.assertIn(row["predicted"], ("slang", "literal", "uncertain"))


if __name__ == "__main__":
    unittest.main()
