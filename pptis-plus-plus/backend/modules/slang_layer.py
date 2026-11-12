import csv
import logging
import math
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from modules.database import SlangDictionary, UnknownSlang, close_session, get_session

logger = logging.getLogger(__name__)

REQUIRED_ENCODER = "xlm-roberta-base"
REQUIRED_THRESHOLD = 0.58
DEFAULT_MODEL = REQUIRED_ENCODER
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+", re.UNICODE)
CHAT_ALLOW = {
    "bro",
    "bruh",
    "dude",
    "sis",
    "bestie",
    "gonna",
    "wanna",
    "kinda",
    "yeah",
    "yep",
    "nah",
    "nope",
    "ok",
    "okay",
    "lol",
    "lmao",
    "omg",
    "idk",
    "tbh",
    "imo",
    "rn",
    "ur",
    "u",
    "im",
    "ive",
    "dont",
    "cant",
    "wont",
    "thats",
    "youre",
    "its",
    "cuz",
    "cause",
    "tho",
    "gng",
}
ENGLISH_WORDS_PATH = os.getenv("ENGLISH_WORDS_PATH", "/usr/share/dict/words")


def contextual_pair(sentence, term):
    return sentence or "", f"target: {term or ''}"


def target_char_span(sentence, term):
    if not sentence or not term:
        return None
    match = re.search(rf"(?<!\w){re.escape(term)}(?!\w)", sentence, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.start(), match.end()


def contextual_vector(tokenizer, encoder, sentence, term, device):
    """Mean of the encoder states covering the target term in the current sentence."""
    encoded = tokenizer(
        sentence or "",
        return_tensors="pt",
        truncation=True,
        max_length=96,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        hidden = encoder(**encoded).last_hidden_state[0]
    span = target_char_span(sentence or "", term or "")
    indexes = []
    if span is not None:
        start, end = span
        for index, (left, right) in enumerate(offsets):
            if right <= left:
                continue
            if left < end and right > start:
                indexes.append(index)
    if indexes:
        vector = hidden[indexes].mean(dim=0)
    else:
        vector = hidden[0]
    return vector.detach().cpu()


class SlangModelError(RuntimeError):
    """The contextual slang model could not classify. No probability is invented."""


def format_contextual_meaning(meaning):
    parts = [part.strip() for part in str(meaning or "").split("/") if part.strip()]
    return " / ".join(parts)


def default_classifier_path():
    configured = (os.getenv("SLANG_CLASSIFIER_PATH") or "").strip()
    if configured:
        return configured
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "slang_literal_head.pt"))


def default_labels_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "slang_context_labels.csv"))


def default_metrics_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "slang_literal_metrics.json"))


def probabilities_are_usable(p_literal, p_slang):
    try:
        pair = (float(p_literal), float(p_slang))
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in pair):
        return False
    return abs(sum(pair) - 1.0) <= 1e-3


def _reject_threshold_override():
    raw = os.getenv("SLANG_THRESHOLD")
    if raw is None or not str(raw).strip():
        return
    try:
        value = float(raw)
    except ValueError as exc:
        raise SlangModelError("slang threshold must be 0.58") from exc
    if abs(value - REQUIRED_THRESHOLD) > 1e-9:
        raise SlangModelError(f"slang threshold must stay {REQUIRED_THRESHOLD}")


_ENCODER_CACHE = {}


def _shared_slang_encoder(model_name, device):
    cached = _ENCODER_CACHE.get(model_name)
    if cached is not None:
        return cached
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()
    classifier = SlangLiteralHead(encoder.config.hidden_size).to(device)
    _ENCODER_CACHE[model_name] = (tokenizer, encoder, classifier)
    return _ENCODER_CACHE[model_name]


class SlangLiteralHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, cls_embedding):
        return self.linear(cls_embedding)


class SlangInterpreter:
    def __init__(self, threshold=None, model_name=None):
        _reject_threshold_override()
        requested = model_name or os.getenv("SLANG_MODEL_NAME") or DEFAULT_MODEL
        if requested != REQUIRED_ENCODER:
            raise SlangModelError(f"slang encoder must be {REQUIRED_ENCODER}, not {requested}")
        if threshold is not None and abs(float(threshold) - REQUIRED_THRESHOLD) > 1e-9:
            raise SlangModelError(f"slang threshold must stay {REQUIRED_THRESHOLD}")
        self.threshold = REQUIRED_THRESHOLD
        self.transformer_threshold = None
        self.model_name = REQUIRED_ENCODER
        self.inference = "actual"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.slang_dictionary = {}
        self._load_slang_dictionary()
        self._english_words = None
        self._classifier_path = None
        self._use_trained_head = False
        logger.info("Loading slang context encoder %s on %s", self.model_name, self.device)
        self.tokenizer, self.encoder, self.classifier = _shared_slang_encoder(self.model_name, self.device)
        self._load_classifier_weights()
        self._use_trained_head = True
        logger.info(
            "SlangInterpreter ready terms=%s trained_head=%s threshold=%s device=%s",
            len(self.slang_dictionary),
            self._use_trained_head,
            self.threshold,
            self.device,
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
            self._merge_csv_dictionary()
            if not self.slang_dictionary:
                raise SlangModelError("slang dictionary is empty")
            logger.info("Loaded %s slang dictionary entries", len(self.slang_dictionary))
        except SlangModelError:
            raise
        except Exception as exc:
            logger.exception("Failed to load slang_dictionary: %s", exc)
            self.slang_dictionary = {}
            self._terms_by_length = ()
            self._merge_csv_dictionary()
            if not self.slang_dictionary:
                raise SlangModelError("slang dictionary is unavailable") from exc
        finally:
            close_session()

    def _load_classifier_weights(self):
        path = default_classifier_path()
        if not os.path.isfile(path):
            raise SlangModelError(f"trained slang head is missing at {path}")
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)
        encoder_name = payload.get("encoder_name")
        if encoder_name and encoder_name != self.model_name:
            raise SlangModelError(
                f"slang head was trained for {encoder_name} and cannot be applied to {self.model_name}"
            )
        state = payload["state_dict"]
        if "weight" in state and "linear.weight" not in state:
            state = {"linear.weight": state["weight"], "linear.bias": state["bias"]}
        self.classifier.load_state_dict(state)
        self.classifier.eval()
        self.transformer_threshold = float(payload["threshold"])
        if abs(self.transformer_threshold - REQUIRED_THRESHOLD) > 1e-6:
            raise SlangModelError(
                f"slang head threshold is {self.transformer_threshold}, required {REQUIRED_THRESHOLD}"
            )
        self.threshold = REQUIRED_THRESHOLD
        self._classifier_path = path
        version = getattr(torch, "__version__", "unknown")
        logger.info(
            "Loaded trained slang head %s encoder=%s threshold=%.4f torch=%s device=%s",
            path,
            encoder_name,
            self.transformer_threshold,
            version,
            self.device,
        )
        return True

    def runtime_status(self):
        return {
            "inference": "actual",
            "encoder_loaded": getattr(self, "encoder", None) is not None,
            "encoder_name": self.model_name if getattr(self, "encoder", None) is not None else None,
            "trained_head": bool(getattr(self, "_use_trained_head", False)),
            "classifier_path": getattr(self, "_classifier_path", None),
            "transformer_threshold": getattr(self, "transformer_threshold", None),
            "device": str(getattr(self, "device", "")),
            "torch_version": getattr(torch, "__version__", None),
        }

    def reload_dictionary(self):
        self._load_slang_dictionary()

    def _resolve_csv_path(self):
        configured = os.getenv("SLANG_CSV_PATH")
        module_csv = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "database", "slang_dictionary.csv")
        )
        candidates = []
        if configured:
            candidates.append(configured)
            candidates.append(os.path.abspath(configured))
        candidates.append(module_csv)
        for path in candidates:
            if path and os.path.isfile(path):
                return path
        return None

    def _merge_csv_dictionary(self):
        path = self._resolve_csv_path()
        if not path:
            logger.info("Slang CSV not found; dictionary stays at database rows")
            return
        added = 0
        try:
            handle = open(path, newline="", encoding="utf-8-sig")
        except OSError as exc:
            logger.exception("Could not read slang CSV %s: %s", path, exc)
            return
        with handle:
            for row in csv.DictReader(handle):
                term = (row.get("slang_term") or "").strip()
                meaning = (row.get("meaning") or "").strip()
                if not term or not meaning:
                    continue
                key = term.lower()
                if key in self.slang_dictionary:
                    continue
                try:
                    confidence = float(row.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    confidence = 0.0
                self.slang_dictionary[key] = {
                    "term": term,
                    "meaning": meaning,
                    "category": (row.get("category") or "").strip(),
                    "confidence": confidence,
                }
                added += 1
        self._terms_by_length = tuple(sorted(self.slang_dictionary.keys(), key=len, reverse=True))
        logger.info("Merged %s CSV slang terms; total=%s", added, len(self.slang_dictionary))

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
        context = context or ""
        if not (
            getattr(self, "_use_trained_head", False)
            and getattr(self, "encoder", None) is not None
            and getattr(self, "classifier", None) is not None
        ):
            raise SlangModelError("contextual slang classifier is not loaded")
        p_literal, p_slang = self._transformer_probabilities(term, context)
        source = "xlm-roberta-classifier"
        if not probabilities_are_usable(p_literal, p_slang):
            logger.error("Rejected non-probability slang output term=%s source=%s", term, source)
            raise SlangModelError("contextual slang classifier returned an invalid probability")
        threshold = self._active_threshold(source)
        if p_slang >= threshold:
            classification = "slang"
        elif p_literal >= threshold:
            classification = "literal"
        else:
            classification = "uncertain"
        result = {
            "term": term,
            "p_slang": p_slang,
            "p_literal": p_literal,
            "classification": classification,
            "threshold": threshold,
            "source": source,
        }
        logger.info(
            "Disambiguation term=%s p_slang=%.3f p_literal=%.3f label=%s source=%s",
            term,
            p_slang,
            p_literal,
            classification,
            source,
        )
        return result

    def disambiguate_batch(self, items):
        results = []
        for item in items:
            term = item.get("term")
            context = item.get("context") or ""
            results.append(self.disambiguate_slang_literal(term, context))
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
                if "p_slang" in detection and detection.get("classification"):
                    verdict = detection
                else:
                    verdict = self.disambiguate_slang_literal(term, context)
                if verdict.get("classification") != "slang" or verdict.get("p_slang") is None:
                    continue
                cutoff = verdict.get("threshold")
                if cutoff is None:
                    cutoff = self.threshold if self.threshold is not None else 1.0
                if float(verdict["p_slang"]) < float(cutoff):
                    continue
                entry = self._lookup_term(term)
                if not entry:
                    unknown.append(self.handle_unknown_slang(term, original_text, user_id))
                    continue
                annotations.append(
                    f"[Contextual meaning: {format_contextual_meaning(entry['meaning'])}]"
                )
            except SlangModelError:
                raise
            except Exception as exc:
                logger.exception("Failed to enrich term=%s error=%s", term, exc)
                raise SlangModelError(f"slang enrichment failed for {term}") from exc

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
            candidate["local_context"] = self.extract_context(
                text, candidate["pos"], context_window=context_window, term=candidate["term"]
            )
            candidate["context"] = text or candidate["local_context"]
        verdicts = self.disambiguate_batch(
            [{"term": item["term"], "context": item["context"]} for item in candidates]
        )
        detections = []
        for candidate, verdict in zip(candidates, verdicts):
            candidate.update(verdict)
            entry = None
            p_slang = verdict.get("p_slang")
            p_literal = verdict.get("p_literal")
            usage = verdict.get("classification") or "unclassified"
            if usage == "slang" and p_slang is not None:
                entry = self._lookup_term(candidate["term"])
                confidence = round(float(p_slang), 4)
                if entry is None:
                    usage = "unknown"
            elif usage == "literal" and p_literal is not None:
                confidence = round(float(p_literal), 4)
            elif p_slang is not None and p_literal is not None:
                usage = "uncertain" if usage not in ("slang", "literal") else usage
                confidence = round(max(float(p_slang), float(p_literal)), 4)
            else:
                usage = "unclassified"
                confidence = None
            detections.append(
                {
                    "term": candidate["term"],
                    "usage": usage,
                    "confidence": confidence,
                    "meaning": format_contextual_meaning(entry["meaning"]) if entry else None,
                    "source": verdict.get("source"),
                }
            )
        unknown_records = []
        for token, pos in self._unknown_tokens(text, candidates):
            record = self.handle_unknown_slang(token, text, user_id)
            unknown_records.append(record)
            detections.append(
                {
                    "term": token,
                    "usage": "unknown",
                    "confidence": None,
                    "meaning": None,
                    "label": "UNKNOWN_SLANG",
                    "pos": pos,
                    "stored": record.get("stored"),
                }
            )
        result = self.enrich_with_context(text, candidates, user_id=user_id)
        result["unknown_slang"] = list(result.get("unknown_slang") or []) + unknown_records
        result["detections"] = detections
        return result

    def handle_unknown_slang(self, term, sentence, user_id):
        payload = {
            "term": term,
            "unknown": True,
            "label": "UNKNOWN_SLANG",
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

    def _active_threshold(self, source):
        if source != "xlm-roberta-classifier":
            raise SlangModelError(f"unexpected slang classifier source {source}")
        if abs(float(self.threshold) - REQUIRED_THRESHOLD) > 1e-9:
            raise SlangModelError(f"slang threshold must stay {REQUIRED_THRESHOLD}")
        return REQUIRED_THRESHOLD

    def _transformer_probabilities(self, term, context):
        vector = contextual_vector(self.tokenizer, self.encoder, context, term, self.device)
        with torch.no_grad():
            probs = F.softmax(self.classifier(vector.unsqueeze(0).to(self.device)), dim=-1).squeeze(0)
        return float(probs[0].item()), float(probs[1].item())

    def _english_lexicon(self):
        cached = getattr(self, "_english_words", None)
        if cached is not None:
            return cached
        words = set()
        path = ENGLISH_WORDS_PATH
        if path and os.path.isfile(path):
            with open(path, encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    token = line.strip().lower()
                    if token:
                        words.add(token)
        self._english_words = words
        return words

    def _is_known_english(self, token):
        key = token.lower()
        if key in CHAT_ALLOW or "'" in key:
            return True
        words = self._english_lexicon()
        if key in words:
            return True
        stems = []
        for suffix in ("ing", "ed", "es", "ly", "s"):
            if key.endswith(suffix) and len(key) - len(suffix) >= 3:
                stem = key[: -len(suffix)]
                stems.append(stem)
                stems.append(stem + "e")
        if key.endswith("ies") and len(key) - 3 >= 3:
            stems.append(key[:-3] + "y")
        return any(stem in words for stem in stems)

    def _unknown_tokens(self, text, candidates):
        if not text:
            return []
        spans = []
        for candidate in candidates or []:
            pos = candidate.get("pos")
            term = candidate.get("term") or ""
            if isinstance(pos, int) and pos >= 0:
                spans.append((pos, pos + len(term)))
        found = []
        seen = set()
        for match in TOKEN_PATTERN.finditer(text):
            token = match.group()
            if len(token) < 4 or not token.isalpha():
                continue
            if self._overlaps(match.start(), match.end(), spans):
                continue
            key = token.lower()
            if key in seen or key in self.slang_dictionary or self._is_known_english(token):
                continue
            seen.add(key)
            found.append((token, match.start()))
        return found

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
