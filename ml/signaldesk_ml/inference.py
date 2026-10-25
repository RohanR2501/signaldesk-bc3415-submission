"""Load a trusted local baseline pipeline without importing any training workflow."""

import json
from pathlib import Path
from typing import Any

import joblib

from signaldesk.contracts import SentimentLabel
from signaldesk_data.source import file_hash


def predict_texts(pipeline: Any, texts: list[str]) -> list[str]:
    if not isinstance(texts, list) or not texts:
        raise ValueError("Provide a nonempty list of review text strings only")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every model input must be nonblank review text")
    predictions = [str(value) for value in pipeline.predict(texts)]
    if not set(predictions) <= {label.value for label in SentimentLabel}:
        raise ValueError("Artifact produced labels outside the sentiment contract")
    return predictions


def load_selected(run_directory: Path, *, trusted: bool = False) -> Any:
    """Joblib can execute code: explicit trust is required even with matching checksums."""
    if not trusted:
        raise ValueError("Only load artifacts produced by a trusted local run")
    metadata = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    selected = metadata["selected_model"]
    path = (run_directory / metadata["models"][selected]["artifact"]).resolve()
    if not path.is_relative_to(run_directory.resolve()):
        raise ValueError("Model artifact must remain within the run directory")
    if file_hash(path) != metadata["models"][selected]["artifact_sha256"]:
        raise ValueError("Selected model artifact checksum mismatch")
    return joblib.load(path)
