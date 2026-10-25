"""Lazy local sentiment inference with explicit CPU fallback and rule-only complaints."""

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Protocol
from uuid import uuid4

from signaldesk.contracts import (
    AnalyzeResponse,
    ComplaintCategory,
    ComplaintDetection,
    DetectionState,
    LiveSentiment,
    ModelAttempt,
    ReviewExcerpt,
    SentimentLabel,
)
from signaldesk_data.source import file_hash
from signaldesk_ml.complaint_rules import RULE_VERSION, TAXONOMY_VERSION, detect_complaints

WARNINGS = [
    "Provisional sentiment prediction; mixed/short/neutral text needs interpretation.",
    "Global sentiment never clears a complaint; rules are unvalidated customer-report aids.",
    "No rule match is unknown, not proof of no issue. No verified incident or automatic action.",
    "No explanation spans or confidence scores are supplied by the sentiment label interface.",
]
RELIABILITY = {
    ComplaintCategory.PRODUCT_DEFECT: "Low rule coverage; functional scope/design can be confused.",
    ComplaintCategory.ARRIVAL_DAMAGE: "Sparse positive support; detection quality unestablished.",
    ComplaintCategory.WRONG_ITEM: "Rules miss known examples; unreliable NLI/student are not used.",
    ComplaintCategory.DELIVERY_ISSUE: "Unreliable detection; delivery student unavailable.",
    ComplaintCategory.SERVICE_RETURN_ISSUE: "Sparse positives; return mentions may be ambiguous.",
}


def complaint_results(text: str | None, review_id: str) -> list[ComplaintDetection]:
    """Detect on unmodified permitted text; never gate complaints on sentiment."""
    detection = detect_complaints(text) if text is not None else None
    results = []
    for category in ComplaintCategory:
        evidence = []
        if detection is not None:
            for item in detection.evidence:
                if item.category.value != category.value:
                    continue
                span, resolution = item.span, item.resolution_span
                evidence.append(
                    ReviewExcerpt(
                        review_id=review_id,
                        start=span.start,
                        end=span.end,
                        text=span.text,
                        method="rule_match",
                        rule_id=item.rule_id,
                        resolution=item.resolution,
                        resolution_excerpt=resolution.text if resolution else None,
                        resolution_start=resolution.start if resolution else None,
                        resolution_end=resolution.end if resolution else None,
                    )
                )
        state = (
            DetectionState.UNAVAILABLE
            if detection is None
            else DetectionState.DETECTED
            if evidence
            else DetectionState.UNKNOWN
        )
        results.append(
            ComplaintDetection(
                category=category,
                state=state,
                evidence=evidence,
                rule_version=RULE_VERSION,
                taxonomy_version=TAXONOMY_VERSION,
                reason="body_unavailable"
                if detection is None
                else "explicit_rule_match"
                if evidence
                else "no_detected_rule_evidence",
                reliability_warning=RELIABILITY[category],
                learned_model_state="unavailable"
                if category == ComplaintCategory.DELIVERY_ISSUE
                else "not_used",
            )
        )
    return results


class TextPredictor(Protocol):
    def predict_texts(self, texts: list[str]) -> list[str]: ...


class _CudaUnavailable(RuntimeError):
    pass


class _BaselinePredictor:
    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def predict_texts(self, texts: list[str]) -> list[str]:
        from threadpoolctl import threadpool_limits

        from signaldesk_ml.inference import predict_texts

        with threadpool_limits(limits=4):
            return predict_texts(self.pipeline, texts)


def _load_transformer(path: Path) -> tuple[TextPredictor, dict[str, Any]]:
    # Importing torch/transformers is deferred until a submitted review needs it.
    from signaldesk_ml.transformer_inference import load_transformer, require_cuda

    try:
        require_cuda()
    except RuntimeError:
        raise _CudaUnavailable from None
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    if run["configuration"]["version"] != "sentiment-distilbert-weighted-v1":
        raise ValueError("Expected the frozen weighted checkpoint")
    model = load_transformer(path)
    return model, {
        "model_version": run["configuration"]["version"],
        "device": "cuda:0",
        "manifest_sha256": file_hash(path / "run.json"),
        "artifact_sha256": run["checkpoint_files"],
        "max_tokens": run["max_length"],
        "preprocessing": "saved_local_tokenizer:right_truncation",
    }


def _load_baseline(path: Path) -> tuple[TextPredictor, dict[str, Any]]:
    from signaldesk_ml.inference import load_selected

    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    if run["selected_model"] != "logistic_regression_balanced":
        raise ValueError("Expected the frozen balanced logistic regression fallback")
    model = load_selected(path, trusted=True)  # Fixed administrator-owned local artifacts only.
    entry = run["models"][run["selected_model"]]
    return _BaselinePredictor(model), {
        "model_version": f"{run['version']}:{run['selected_model']}",
        "device": "cpu",
        "manifest_sha256": file_hash(path / "run.json"),
        "artifact_sha256": {entry["artifact"]: entry["artifact_sha256"]},
        "preprocessing": "saved_training_only_tfidf_pipeline",
    }


def _failure_reason(exc: Exception) -> str:
    """Stable codes only: exception messages may contain input or private paths."""
    if isinstance(exc, _CudaUnavailable):
        return "cuda_unavailable"
    if isinstance(exc, FileNotFoundError):
        return "artifact_missing"
    if isinstance(exc, ImportError):
        return "dependency_unavailable"
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return "artifact_or_output_invalid"
    return "device_or_runtime_unavailable"


class ReviewAnalyzer:
    """One instance per app; serialize first load/inference and cache failures until restart."""

    def __init__(self, transformer_path: Path, baseline_path: Path) -> None:
        self.paths = {"weighted_distilbert": transformer_path, "balanced_lr": baseline_path}
        self.models: dict[str, TextPredictor] = {}
        self.identities: dict[str, dict[str, Any]] = {}
        self.failures: dict[str, str] = {}
        self._lock = Lock()

    def sentiment(self, text: str) -> LiveSentiment:
        attempts: list[ModelAttempt] = []
        with self._lock:
            for name, loader in (
                ("weighted_distilbert", _load_transformer),
                ("balanced_lr", _load_baseline),
            ):
                if name not in self.failures:
                    try:
                        if name not in self.models:
                            # Check presence before costly library import or checkpoint loading.
                            if not (self.paths[name] / "run.json").is_file():
                                raise FileNotFoundError
                            model, identity = loader(self.paths[name])
                            self.models[name], self.identities[name] = model, identity
                        label = SentimentLabel(self.models[name].predict_texts([text])[0])
                        attempts.append(ModelAttempt(model=name, state="used"))
                        return LiveSentiment(
                            state=DetectionState.DETECTED,
                            label=label,
                            fallback_reason=self.failures.get("weighted_distilbert"),
                            attempts=attempts,
                            **self.identities[name],
                        )
                    except Exception as exc:
                        self.failures[name] = _failure_reason(exc)
                        self.models.pop(name, None)
                        self.identities.pop(name, None)
                attempts.append(
                    ModelAttempt(model=name, state="unavailable", reason=self.failures[name])
                )
        return LiveSentiment(
            state=DetectionState.UNAVAILABLE,
            attempts=attempts,
            fallback_reason=self.failures.get("weighted_distilbert"),
        )

    def analyze(self, text: str) -> AnalyzeResponse:
        review_id = f"submitted:{uuid4()}"
        sentiment = self.sentiment(text)
        return AnalyzeResponse(
            review_id=review_id,
            analyzed_at=datetime.now(UTC),
            status="complete" if sentiment.state == DetectionState.DETECTED else "partial",
            sentiment=sentiment,
            complaints=complaint_results(text, review_id),
            warnings=WARNINGS,
        )
