import logging
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from modules.database import SlangDictionary, UnknownSlang, close_session, get_session

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.7
DEFAULT_MODEL = os.getenv("SLANG_MODEL_NAME", "xlm-roberta-base")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+", re.UNICODE)

SLANG_PROTOTYPES = (
    "bro you're {term} fr ngl",
    "that's so {term} dude",
    "we {term} out here",
)
LITERAL_PROTOTYPES = (
    "the {term} is ready in the kitchen",
    "chicken is {term} on the stove",
    "a literal dictionary use of {term}",
)


class SlangLiteralHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, cls_embedding):
        return self.linear(cls_embedding)


class SlangInterpreter:
    def __init__(self, threshold=DEFAULT_THRESHOLD, model_name=None):
        self.threshold = float(os.getenv("SLANG_THRESHOLD", threshold))
        self.model_name = model_name or DEFAULT_MODEL
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.slang_dictionary = {}
        self._load_slang_dictionary()

        self.light_mode = os.getenv("PPTIS_LIGHT_MODE", "true").lower() in ("1", "true", "yes")
        self.tokenizer = None
        self.encoder = None
        self.classifier = None
        self._use_trained_head = False
        if self.light_mode:
            logger.info("SlangInterpreter light mode (skip Hugging Face download)")
        else:
            logger.info("Loading slang context model %s on %s", self.model_name, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.encoder = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.encoder.eval()
            hidden_size = self.encoder.config.hidden_size
            self.classifier = SlangLiteralHead(hidden_size).to(self.device)
            self._use_trained_head = self._load_classifier_weights()
        logger.info(
            "SlangInterpreter ready terms=%s threshold=%.2f trained_head=%s light=%s",
            len(self.slang_dictionary),
            self.threshold,
            self._use_trained_head,
            self.light_mode,
        )

    def _load_slang_dictionary(self):
        session = get_session()
        try:
            rows = session.query(SlangDictionary).all()
            dictionary = {}
            for row in rows:
                key = row.slang_term.strip().lower()
                dictionary[key] = {
                    "term": row.slang_term,
                    "meaning": row.meaning,
                    "category": row.category,
                    "confidence": row.confidence,
                }
            self.slang_dictionary = dictionary
            self._terms_by_length = tuple(
                sorted(dictionary.keys(), key=len, reverse=True)
            )
            logger.info("Loaded %s slang dictionary entries", len(dictionary))
        except Exception as exc:
            logger.exception("Failed to load slang_dictionary: %s", exc)
            self.slang_dictionary = {}
            self._terms_by_length = ()
        finally:
            close_session()

    def _load_classifier_weights(self):
        path = os.getenv("SLANG_CLASSIFIER_PATH")
        if not path or not os.path.isfile(path):
            logger.info("No slang classifier checkpoint; using contextual prototype scoring")
            return False
        state = torch.load(path, map_location=self.device)
        self.classifier.load_state_dict(state)
        self.classifier.eval()
        logger.info("Loaded slang classifier head from %s", path)
        return True

    def reload_dictionary(self):
        self._load_slang_dictionary()

    def detect_candidates(self, text):
        if not text:
            return []
        lowered = text.lower()
        used_spans = []
        candidates = []

        for term in self._terms_by_length:
            start = 0
            while True:
                idx = lowered.find(term, start)
                if idx < 0:
                    break
                end = idx + len(term)
                if self._is_whole_term(lowered, idx, end) and not self._overlaps(idx, end, used_spans):
                    surface = text[idx:end]
                    candidates.append({"term": surface, "pos": idx})
                    used_spans.append((idx, end))
                    logger.info("Slang candidate term=%s pos=%s", surface, idx)
                start = idx + 1

        candidates.sort(key=lambda item: item["pos"])
        return candidates

    def extract_context(self, text, term_position, context_window=15, term=None):
        if not text:
            return ""
        start = max(0, int(term_position) - int(context_window))
        if term:
            end = min(len(text), int(term_position) + len(term) + int(context_window))
        else:
            end = min(len(text), int(term_position) + int(context_window) * 2)
        context = text[start:end].strip()
        logger.info("Extracted slang context pos=%s window=%s", term_position, context_window)
        return context

    def disambiguate_slang_literal(self, term, context):
        if not term:
            raise ValueError("term is required")
        context = context or term
        if self.encoder is None:
            p_slang, p_literal = self._heuristic_scores(term, context)
            classification = "slang" if p_slang >= p_literal else "literal"
            return {
                "term": term,
                "p_slang": p_slang,
                "p_literal": p_literal,
                "classification": classification,
            }
        embedding = self._embed(f"{term} {context}")
        if self._use_trained_head:
            with torch.no_grad():
                logits = self.classifier(embedding)
                probs = F.softmax(logits, dim=-1).squeeze(0)
                p_literal = float(probs[0].item())
                p_slang = float(probs[1].item())
        else:
            p_slang, p_literal = self._prototype_scores(term, context, embedding)

        classification = "slang" if p_slang >= p_literal else "literal"
        result = {
            "term": term,
            "p_slang": p_slang,
            "p_literal": p_literal,
            "classification": classification,
        }
        logger.info(
            "Disambiguation term=%s p_slang=%.3f p_literal=%.3f label=%s",
            term,
            p_slang,
            p_literal,
            classification,
        )
        return result

    def disambiguate_batch(self, items):
        results = []
        for item in items:
            term = item.get("term")
            context = item.get("context") or ""
            try:
                results.append(self.disambiguate_slang_literal(term, context))
            except Exception as exc:
                logger.exception("Batch disambiguation failed term=%s error=%s", term, exc)
                results.append(
                    {
                        "term": term,
                        "p_slang": 0.0,
                        "p_literal": 1.0,
                        "classification": "literal",
                        "error": str(exc),
                    }
                )
        return results

    def enrich_with_context(self, text, slang_detections, user_id=None):
        original_text = text or ""
        annotations = []
        unknown = []

        detections = slang_detections or []
        if detections and isinstance(detections[0], str):
            detections = [{"term": term, "pos": original_text.lower().find(term.lower())} for term in detections]

        for detection in detections:
            term = detection.get("term")
            pos = detection.get("pos", 0)
            try:
                context = detection.get("context") or self.extract_context(
                    original_text, pos, term=term
                )
                verdict = self.disambiguate_slang_literal(term, context)
                if verdict["p_slang"] <= self.threshold:
                    continue
                entry = self._lookup_term(term)
                if not entry:
                    unknown.append(self.handle_unknown_slang(term, original_text, user_id))
                    continue
                annotations.append(f"[{term} = {entry['meaning']}]")
            except Exception as exc:
                logger.exception("Failed to enrich term=%s error=%s", term, exc)

        enriched = original_text
        if annotations:
            enriched = f"{original_text} {' '.join(annotations)}"

        logger.info("Enriched text annotations=%s unknown=%s", len(annotations), len(unknown))
        return {
            "original_text": original_text,
            "enriched_text_with_context": enriched,
            "unknown_slang": unknown,
        }

    def interpret(self, text, user_id=None, context_window=15):
        candidates = self.detect_candidates(text)
        for candidate in candidates:
            candidate["context"] = self.extract_context(
                text, candidate["pos"], context_window=context_window, term=candidate["term"]
            )
        verdicts = self.disambiguate_batch(
            [{"term": item["term"], "context": item["context"]} for item in candidates]
        )
        for candidate, verdict in zip(candidates, verdicts):
            candidate.update(verdict)
        return self.enrich_with_context(text, candidates, user_id=user_id)

    def handle_unknown_slang(self, term, sentence, user_id):
        payload = {
            "term": term,
            "unknown": True,
            "message": "Term is not in slang_dictionary",
        }
        if not term:
            payload["error"] = "missing_term"
            return payload
        if user_id is None:
            logger.warning("Unknown slang term=%s not stored (no user_id)", term)
            payload["stored"] = False
            return payload

        session = get_session()
        try:
            UnknownSlang.create(
                session,
                term=term,
                sentence=sentence or "",
                user_id=user_id,
                review_status="pending",
            )
            logger.info("Stored unknown slang term=%s user_id=%s", term, user_id)
            payload["stored"] = True
        except Exception as exc:
            session.rollback()
            logger.exception("Failed to store unknown slang term=%s error=%s", term, exc)
            payload["stored"] = False
            payload["error"] = str(exc)
        finally:
            close_session()
        return payload

    def _lookup_term(self, term):
        if not term:
            return None
        entry = self.slang_dictionary.get(term.lower())
        if not entry:
            logger.warning("Missing slang dictionary term=%s", term)
        return entry

    def _heuristic_scores(self, term, context):
        lowered = f"{term} {context}".lower()
        slang_cues = (
            "bro",
            "dude",
            "fr",
            "ngl",
            "you're",
            "youre",
            "rizz",
            "lol",
            "lmao",
            "dead",
            "no cap",
            "lowkey",
            "highkey",
            "gg",
            "npc",
            "mid",
            "ate",
            "slay",
        )
        literal_cues = ("chicken", "food", "oven", "kitchen", "recipe", "dinner", "meal", "stove")
        slang_hits = sum(1 for cue in slang_cues if cue in lowered)
        literal_hits = sum(1 for cue in literal_cues if cue in lowered)
        p_slang = 0.85 if slang_hits > literal_hits else 0.35 if slang_hits == literal_hits else 0.12
        if term.lower() in self.slang_dictionary and slang_hits:
            p_slang = max(p_slang, 0.9)
        return p_slang, 1.0 - p_slang

    def _embed(self, text):
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.encoder(**encoded)
            cls_embedding = output.last_hidden_state[:, 0, :]
        return cls_embedding

    def _prototype_scores(self, term, context, query_embedding):
        slang_vecs = [self._embed(prompt.format(term=term)) for prompt in SLANG_PROTOTYPES]
        literal_vecs = [self._embed(prompt.format(term=term)) for prompt in LITERAL_PROTOTYPES]
        slang_sim = max(self._cosine(query_embedding, vec) for vec in slang_vecs)
        literal_sim = max(self._cosine(query_embedding, vec) for vec in literal_vecs)
        logits = torch.tensor([literal_sim, slang_sim], device=self.device)
        probs = F.softmax(logits, dim=0)
        return float(probs[1].item()), float(probs[0].item())

    @staticmethod
    def _cosine(left, right):
        left = F.normalize(left.squeeze(0), dim=0)
        right = F.normalize(right.squeeze(0), dim=0)
        return float(torch.dot(left, right).item())

    @staticmethod
    def _is_whole_term(text, start, end):
        if start > 0 and TOKEN_PATTERN.match(text[start - 1]):
            return False
        if end < len(text) and TOKEN_PATTERN.match(text[end]):
            return False
        return True

    @staticmethod
    def _overlaps(start, end, spans):
        return any(start < span_end and end > span_start for span_start, span_end in spans)
