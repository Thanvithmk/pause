import gc
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REQUIRED_MODEL = "base.en"
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".webm", ".aac", ".wma"}
MODEL_SIZES = {
    "tiny": "39 MB",
    "tiny.en": "39 MB",
    "base": "74 MB",
    "base.en": "74 MB",
    "small": "244 MB",
    "small.en": "244 MB",
    "medium": "769 MB",
    "medium.en": "769 MB",
    "large-v1": "1550 MB",
    "large-v2": "1550 MB",
    "large-v3": "1550 MB",
    "large": "1550 MB",
}
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("WHISPER_TIMEOUT_SECONDS", "120"))
_WHISPER_CACHE = {}


class VoiceModelError(RuntimeError):
    """Whisper base.en could not transcribe. No transcript is invented."""


class VoiceTranscriber:
    def __init__(self, model_name="base.en"):
        if model_name != REQUIRED_MODEL:
            raise VoiceModelError(f"voice model must be {REQUIRED_MODEL}, not {model_name}")
        configured = (os.getenv("WHISPER_MODEL") or REQUIRED_MODEL).strip()
        if configured != REQUIRED_MODEL:
            raise VoiceModelError(f"voice model must be {REQUIRED_MODEL}, not {configured}")
        self.model_name = REQUIRED_MODEL
        self.inference = "actual"
        self.compute_type = "int8"
        self.language = "en"
        cached = _WHISPER_CACHE.get(REQUIRED_MODEL)
        if cached is not None:
            self.model, self.device = cached
            return
        from faster_whisper import WhisperModel

        logger.info("Initializing Faster-Whisper model=%s", REQUIRED_MODEL)
        try:
            self.model = WhisperModel(REQUIRED_MODEL, device="cpu", compute_type=self.compute_type)
        except Exception as exc:
            logger.exception("Failed to load Whisper %s", REQUIRED_MODEL)
            raise VoiceModelError(f"voice model {REQUIRED_MODEL} failed to load") from exc
        self.device = self._resolve_device()
        _WHISPER_CACHE[REQUIRED_MODEL] = (self.model, self.device)
        logger.info("Faster-Whisper ready model=%s device=%s", REQUIRED_MODEL, self.device)

    def _resolve_device(self):
        try:
            raw = str(self.model.model.device)
        except AttributeError:
            raw = str(self.device)
        lowered = raw.lower()
        if "cuda" in lowered or "gpu" in lowered:
            return "CUDA"
        if "cpu" in lowered:
            return "CPU"
        return raw or "CPU"

    def _language_arg(self, language):
        if language in (None, "", "auto"):
            return None
        return language

    def _collect_text(self, segments):
        parts = []
        for segment in segments:
            text = getattr(segment, "text", "") or ""
            if text:
                parts.append(text.strip())
        return " ".join(parts).strip()

    def _transcribe(self, audio, language, beam_size=5):
        segments, info = self.model.transcribe(
            audio,
            language=self._language_arg(language),
            beam_size=beam_size,
        )
        text = self._collect_text(segments)
        del segments
        gc.collect()
        return text, info

    def _transcribe_with_timeout(self, audio, language, timeout_seconds):
        timeout_seconds = timeout_seconds or DEFAULT_TIMEOUT_SECONDS
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._transcribe, audio, language, 5)
            return future.result(timeout=timeout_seconds)

    def transcribe_audio_file(self, audio_path, language="en", timeout_seconds=None):
        path = Path(audio_path)
        started = datetime.now(timezone.utc)
        logger.info("Transcription started file=%s language=%s", path, language)
        if self.model is None:
            raise VoiceModelError(f"voice model {REQUIRED_MODEL} is not loaded")

        if not path.is_file():
            logger.error("Audio file not found: %s", path)
            return {
                "success": False,
                "text": "",
                "language": language,
                "duration": 0,
                "timestamp": started.isoformat(),
                "error": "file_not_found",
            }

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.error("Unsupported audio format: %s", path.suffix)
            return {
                "success": False,
                "text": "",
                "language": language,
                "duration": 0,
                "timestamp": started.isoformat(),
                "error": "unsupported_format",
            }

        try:
            text, info = self._transcribe_with_timeout(str(path), language, timeout_seconds)
            detected = getattr(info, "language", language)
            duration = float(getattr(info, "duration", 0) or 0)
            logger.info(
                "Transcription finished file=%s language=%s duration=%.2fs chars=%s",
                path,
                detected,
                duration,
                len(text),
            )
            return {
                "success": True,
                "text": text,
                "language": detected,
                "duration": duration,
                "timestamp": started.isoformat(),
            }
        except FuturesTimeout:
            logger.error("Transcription timed out file=%s", path)
            return {
                "success": False,
                "text": "",
                "language": language,
                "duration": 0,
                "timestamp": started.isoformat(),
                "error": "timeout",
            }
        except Exception as exc:
            logger.exception("Corrupted or unreadable audio file=%s error=%s", path, exc)
            return {
                "success": False,
                "text": "",
                "language": language,
                "duration": 0,
                "timestamp": started.isoformat(),
                "error": "corrupted_audio",
            }
        finally:
            gc.collect()

    def transcribe_audio_stream(self, audio_data, language="en", timeout_seconds=None):
        logger.info(
            "Stream transcription started type=%s language=%s",
            type(audio_data).__name__,
            language,
        )
        tmp_path = None
        try:
            audio = self._normalize_audio(audio_data)
            if isinstance(audio, str):
                tmp_path = audio
                source = tmp_path
            else:
                source = audio

            text, info = self._transcribe_with_timeout(source, language, timeout_seconds)
            detected = getattr(info, "language", language)
            logger.info(
                "Stream transcription finished language=%s chars=%s",
                detected,
                len(text),
            )
            return text
        except FuturesTimeout as exc:
            logger.error("Stream transcription timed out")
            raise VoiceModelError("voice transcription timed out") from exc
        except VoiceModelError:
            raise
        except Exception as exc:
            logger.exception("Stream transcription failed: %s", exc)
            raise VoiceModelError("voice transcription failed") from exc
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            gc.collect()

    def _normalize_audio(self, audio_data):
        if isinstance(audio_data, (bytes, bytearray, memoryview)):
            suffix = ".wav"
            handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            handle.write(bytes(audio_data))
            handle.close()
            logger.info("Wrote raw audio bytes to temp file size=%s", len(audio_data))
            return handle.name

        array = np.asarray(audio_data)
        if array.size == 0:
            raise ValueError("Empty audio stream")
        if np.issubdtype(array.dtype, np.integer):
            max_int = float(np.iinfo(array.dtype).max) or 32768.0
            array = array.astype(np.float32) / max_int
        else:
            array = array.astype(np.float32, copy=False)

        peak = float(np.max(np.abs(array))) if array.size else 0.0
        if peak > 1.0:
            array = array / peak
        else:
            array = np.clip(array, -1.0, 1.0)

        if array.ndim > 1:
            array = np.mean(array, axis=-1).astype(np.float32)

        return np.ascontiguousarray(array, dtype=np.float32)

    def get_model_info(self):
        return {
            "inference": "actual",
            "model_name": self.model_name,
            "size": MODEL_SIZES.get(self.model_name, "unknown"),
            "language": self.language,
            "device": self.device,
        }
