import hashlib
import logging
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from modules.safety_patterns import SafetyPatternAnalyzer

logger = logging.getLogger(__name__)


class ToxicityAnalysisError(RuntimeError):
    """Raised when model inference fails. Callers must not treat this as ALLOW."""

PRIMARY_MODEL = "unitary/toxic-bert"
TOXICITY_LABELS = (
    "toxic",
    "threat",
    "insult",
    "severe_toxic",
    "obscene",
    "identity_hate",
)
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("TOXICITY_TIMEOUT_SECONDS", "30"))
MAX_LENGTH = int(os.getenv("TOXICITY_MAX_LENGTH", "512"))
CACHE_SIZE = int(os.getenv("TOXICITY_CACHE_SIZE", "256"))
BATCH_SIZE = int(os.getenv("TOXICITY_BATCH_SIZE", "8"))
_MODEL_CACHE = {}


class ToxicityAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = PRIMARY_MODEL
        self.inference = "actual"
        self.multilabel = True
        self._cache = OrderedDict()
        self.model = None
        self.tokenizer = None
        self.safety = SafetyPatternAnalyzer()
        self.id2label = {}
        self._load_model()
        logger.info(
            "ToxicityAnalyzer ready model=%s device=%s torch=%s",
            self.model_name,
            self.device,
            torch.__version__,
        )

    def _load_model(self):
        cached = _MODEL_CACHE.get(PRIMARY_MODEL)
        if cached is not None:
            self.tokenizer, self.model, self.id2label = cached
            self.model_name = PRIMARY_MODEL
            return
        logger.info("Loading toxicity model %s", PRIMARY_MODEL)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(PRIMARY_MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(PRIMARY_MODEL)
        except Exception as exc:
            logger.exception("Failed to load %s", PRIMARY_MODEL)
            raise ToxicityAnalysisError(f"toxicity model {PRIMARY_MODEL} failed to load") from exc
        self.model_name = PRIMARY_MODEL
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {
            int(key): str(value).lower()
            for key, value in getattr(self.model.config, "id2label", {}).items()
        }
        present = set(self.id2label.values())
        missing = [label for label in TOXICITY_LABELS if label not in present]
        if missing:
            raise ToxicityAnalysisError(
                f"toxicity model {PRIMARY_MODEL} is missing labels {missing}"
            )
        _MODEL_CACHE[PRIMARY_MODEL] = (self.tokenizer, self.model, dict(self.id2label))

    def analyze(self, text, timeout_seconds=None, context=""):
        if text is None:
            text = ""
        context = context or ""
        cache_key = f"{context}\n{text}" if context else text
        logger.info("Toxicity analysis started chars=%s context_chars=%s", len(text), len(context))
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.info("Toxicity cache hit")
            return cached

        if getattr(self, "safety", None) is None:
            self.safety = SafetyPatternAnalyzer()

        if self.model is None or self.model_name != PRIMARY_MODEL:
            raise ToxicityAnalysisError(f"toxicity model {PRIMARY_MODEL} is not loaded")

        timeout_seconds = timeout_seconds or DEFAULT_TIMEOUT_SECONDS
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self._score_texts, [text])
                scores = future.result(timeout=timeout_seconds)[0]
        except FuturesTimeout as exc:
            logger.error("Toxicity analysis timed out chars=%s", len(text))
            raise ToxicityAnalysisError("Toxicity analysis timed out") from exc
        except ToxicityAnalysisError:
            raise
        except Exception as exc:
            logger.exception("Toxicity analysis failed: %s", exc)
            raise ToxicityAnalysisError("Toxicity analysis failed") from exc
        finally:
            self._clear_cuda()

        scores = self.safety.merge(scores, text, context=context)["vector"]
        self._cache_put(cache_key, scores)
        logger.info(
            "Toxicity analysis finished primary=%s",
            self.get_primary_toxicity_dimension(scores),
        )
        return scores

    def analyze_batch(self, text_list, timeout_seconds=None):
        texts = [("" if item is None else str(item)) for item in (text_list or [])]
        logger.info("Batch toxicity analysis count=%s", len(texts))
        if not texts:
            return []

        results = [None] * len(texts)
        pending_indices = []
        pending_texts = []
        for index, text in enumerate(texts):
            cached = self._cache_get(text)
            if cached is not None:
                results[index] = cached
            else:
                pending_indices.append(index)
                pending_texts.append(text)

        timeout_seconds = timeout_seconds or DEFAULT_TIMEOUT_SECONDS
        try:
            scored = []
            for start in range(0, len(pending_texts), BATCH_SIZE):
                chunk = pending_texts[start : start + BATCH_SIZE]
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(self._score_texts, chunk)
                    scored.extend(future.result(timeout=timeout_seconds))
            for index, vector in zip(pending_indices, scored):
                results[index] = vector
                self._cache_put(texts[index], vector)
        except FuturesTimeout as exc:
            logger.error("Batch toxicity analysis timed out count=%s", len(texts))
            raise ToxicityAnalysisError("Toxicity analysis timed out") from exc
        except ToxicityAnalysisError:
            raise
        except Exception as exc:
            logger.exception("Batch toxicity analysis failed: %s", exc)
            raise ToxicityAnalysisError("Toxicity analysis failed") from exc
        finally:
            self._clear_cuda()

        return results

    def get_primary_toxicity_dimension(self, toxicity_vector):
        if not toxicity_vector:
            return None
        return max(TOXICITY_LABELS, key=lambda label: float(toxicity_vector.get(label, 0.0)))

    def _score_texts(self, texts):
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.sigmoid(logits)
        return [self._vector_from_probs(row) for row in probs]

    def _vector_from_probs(self, prob_row):
        values = {}
        for index, score in enumerate(prob_row.tolist()):
            label = self.id2label.get(index)
            if label in TOXICITY_LABELS:
                values[label] = self._clamp(score)
        missing = [label for label in TOXICITY_LABELS if label not in values]
        if missing:
            raise ToxicityAnalysisError(
                f"toxicity model {PRIMARY_MODEL} did not return labels {missing}"
            )
        return {label: float(values[label]) for label in TOXICITY_LABELS}

    def _cache_get(self, text):
        key = self._cache_key(text)
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return dict(self._cache[key])

    def _cache_put(self, text, vector):
        key = self._cache_key(text)
        self._cache[key] = dict(vector)
        self._cache.move_to_end(key)
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)

    @staticmethod
    def _cache_key(text):
        return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

    @staticmethod
    def _clamp(value):
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _empty_vector():
        return {label: 0.0 for label in TOXICITY_LABELS}

    def _clear_cuda(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
