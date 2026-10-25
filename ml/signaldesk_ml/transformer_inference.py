"""Local-only saved transformer inference, independent of training/evaluation code."""

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from signaldesk_data.source import file_hash

LABELS = ["negative", "neutral", "positive"]


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")


class SentimentTransformer:
    """Text-only predictions; scores are uncalibrated, and long text is truncated."""

    def __init__(self, model: Any, tokenizer: Any, max_length: int) -> None:
        require_cuda()
        self.model = model.to("cuda:0").eval()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def predict_texts(self, texts: list[str], batch_size: int = 32) -> list[str]:
        if not isinstance(texts, list) or not texts:
            raise ValueError("Provide a nonempty list of review text strings only")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every model input must be nonblank review text")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        predictions: list[str] = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                encoded = self.tokenizer(
                    texts[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to("cuda:0")
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = self.model(**encoded).logits
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("Nonfinite transformer output")
                predictions.extend(LABELS[i] for i in logits.argmax(-1).tolist())
        return predictions


def load_transformer(directory: Path) -> SentimentTransformer:
    """Verify saved files, then load safetensors/tokenizer without network access."""
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    for name, checksum in metadata["checkpoint_files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or file_hash(path) != checksum:
            raise ValueError("Transformer checkpoint checksum mismatch")
    checkpoint = directory / "checkpoint"
    mapping = json.loads((checkpoint / "class_mapping.json").read_text(encoding="utf-8"))
    if mapping != {str(i): label for i, label in enumerate(LABELS)}:
        raise ValueError("Unexpected sentiment class mapping")
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    if model.config.id2label != dict(enumerate(LABELS)):
        raise ValueError("Checkpoint labels differ from canonical sentiment labels")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    return SentimentTransformer(model, tokenizer, metadata["max_length"])
