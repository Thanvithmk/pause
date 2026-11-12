"""Load the mandatory local models and refuse to serve if one of them fails."""

import logging
import os

logger = logging.getLogger(__name__)


def verify_primary_models():
    from modules.rewrite_engine import REQUIRED_GROQ_MODEL, RewriteServiceError, _setting
    from modules.slang_layer import REQUIRED_ENCODER, REQUIRED_THRESHOLD, SlangInterpreter, SlangModelError
    from modules.toxicity_model import PRIMARY_MODEL, TOXICITY_LABELS, ToxicityAnalysisError, ToxicityAnalyzer

    import torch
    from transformers import __version__ as transformers_version

    report = {
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda": bool(torch.cuda.is_available()),
    }

    slang = SlangInterpreter()
    verdict = slang.disambiguate_slang_literal("fire", "The building caught fire.")
    if verdict.get("source") != "xlm-roberta-classifier":
        raise SlangModelError("slang classifier did not use xlm-roberta-base")
    if verdict.get("p_slang") is None or verdict.get("p_literal") is None:
        raise SlangModelError("slang classifier did not return probabilities")
    status = slang.runtime_status()
    report["slang"] = {
        "inference": "actual",
        "encoder": REQUIRED_ENCODER,
        "checkpoint": os.path.basename(status["classifier_path"] or ""),
        "threshold": REQUIRED_THRESHOLD,
        "device": status["device"],
        "trained_head": status["trained_head"],
    }

    toxicity = ToxicityAnalyzer()
    scores = toxicity.analyze("Thanks for the notes.")
    if toxicity.model_name != PRIMARY_MODEL:
        raise ToxicityAnalysisError(f"toxicity model must be {PRIMARY_MODEL}")
    if set(scores) != set(TOXICITY_LABELS):
        raise ToxicityAnalysisError("toxicity output is not the six-dimensional vector")
    report["toxicity"] = {
        "inference": "actual",
        "model": PRIMARY_MODEL,
        "device": str(toxicity.device),
        "labels": list(TOXICITY_LABELS),
    }

    if not _setting("GROQ_API_KEY"):
        raise RewriteServiceError("rewrite: GROQ_API_KEY is not configured")
    configured = _setting("GROQ_MODEL") or REQUIRED_GROQ_MODEL
    if configured != REQUIRED_GROQ_MODEL:
        raise RewriteServiceError(
            f"rewrite: GROQ_MODEL must be {REQUIRED_GROQ_MODEL}, not {configured}"
        )
    report["rewrite"] = _groq_catalog_status()
    report["local_models_ready"] = True
    report["production_ready"] = bool(report["rewrite"].get("available"))
    logger.info("Primary models verified %s", report)
    return report


def _groq_catalog_status():
    from groq import Groq
    from modules.rewrite_engine import REQUIRED_GROQ_MODEL, _setting

    status = {
        "inference": "unavailable",
        "model": REQUIRED_GROQ_MODEL,
        "api_key_configured": True,
        "available": False,
    }
    try:
        page = Groq(api_key=_setting("GROQ_API_KEY")).models.list()
        identifiers = {item.id for item in page.data}
    except Exception as exc:
        status["error"] = f"rewrite: Groq model list failed ({type(exc).__name__})"
        logger.exception("Groq model list failed")
        return status
    if REQUIRED_GROQ_MODEL in identifiers:
        status["inference"] = "configuration-checked"
        status["available"] = True
        status["catalog_count"] = len(identifiers)
        return status
    status["catalog_count"] = len(identifiers)
    status["error"] = (
        f"rewrite: {REQUIRED_GROQ_MODEL} is not available to this API key"
    )
    return status
