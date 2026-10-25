"""Offline complaint inference only; no training, evaluation, or backend routes.

Scores are uncalibrated. Rule references are explicit matches, never model attribution.
All approaches retain independent categories and abstention; sentiment is not an input.
"""

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from signaldesk_data.source import file_hash
from signaldesk_ml.complaint_rules import (
    RULE_VERSION,
    TAXONOMY_VERSION,
    ComplaintCategory,
    detect_complaints,
)

LABELS = tuple(label.value for label in ComplaintCategory)
Mode = Literal["rules", "nli", "linear"]
ScoreArray = NDArray[np.float64]


def validate_texts(texts: list[str]) -> None:
    if not isinstance(texts, list) or not texts:
        raise ValueError("Provide a nonempty list of original review bodies")
    if any(not isinstance(t, str) or not t.strip() or len(t) > 50000 for t in texts):
        raise ValueError("Review bodies must be nonblank strings of at most 50000 characters")


class ComplaintNLI:
    """CUDA-only entailment vs contradiction, independently for each hypothesis."""

    def __init__(self, path: Path, config: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; complaint NLI never falls back to CPU")
        torch.set_num_threads(config["cpu_threads"])
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                path,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=torch.float16,
                attn_implementation="sdpa",
            )
            .to("cuda")
            .eval()
        )
        labels = {str(v).lower(): int(k) for k, v in self.model.config.id2label.items()}
        if "contradiction" not in labels or "entailment" not in labels:
            raise ValueError("Checkpoint does not declare NLI contradiction/entailment classes")
        self.indices = [labels["contradiction"], labels["entailment"]]

    def score_texts(
        self,
        texts: list[str],
        *,
        max_seconds: float | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> ScoreArray:
        import torch

        validate_texts(texts)
        started = time.perf_counter()
        result: list[float] = []
        total = len(texts) * len(LABELS)
        batch_size = self.config["pair_batch_size"]
        with torch.inference_mode():
            for start in range(0, total, batch_size):
                if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                    raise TimeoutError("Bounded initial NLI processing exceeded its wall budget")
                pairs = range(start, min(start + batch_size, total))
                premises = [texts[p // len(LABELS)] for p in pairs]
                hypotheses = [self.config["hypotheses"][LABELS[p % len(LABELS)]] for p in pairs]
                tokens = self.tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation="only_first",
                    max_length=self.config["max_length"],
                    return_tensors="pt",
                ).to("cuda")
                logits = self.model(**tokens).logits
                scores = logits[:, self.indices].float().softmax(dim=-1)[:, 1]
                result.extend(scores.cpu().tolist())
                if progress is not None:
                    progress(min(start + batch_size, total), total)
        array = np.asarray(result, dtype=np.float64).reshape(len(texts), len(LABELS))
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite NLI scores")
        return array


@dataclass(frozen=True, slots=True)
class CategoryPrediction:
    category: str
    source: Mode
    status: Literal["selected", "not_selected", "abstain", "unavailable"]
    score: float | None
    score_kind: str | None
    rule_reference: tuple[dict[str, Any], ...]
    rule_disagreement: bool


@dataclass(frozen=True, slots=True)
class ComplaintPrediction:
    categories: tuple[str, ...]
    predictions: tuple[CategoryPrediction, ...]
    status: Literal["provisional_customer_reported_issues", "unknown_no_selected_category"]
    source: Mode
    model_version: str
    taxonomy_version: str = TAXONOMY_VERSION
    rule_version: str = RULE_VERSION
    independent_validation: str = "not_performed"


def assemble_prediction(
    text: str, mode: Mode, scores: list[float | None], config: dict[str, Any]
) -> ComplaintPrediction:
    validate_texts([text])
    if len(scores) != len(LABELS):
        raise ValueError("Expected one independent score slot per category")
    rules = detect_complaints(text)
    predictions = []
    for category, score in zip(LABELS, scores, strict=True):
        evidence = tuple(
            {**asdict(item), "provenance": "explicit_rule_match", "is_model_attribution": False}
            for item in rules.evidence
            if item.category == category
        )
        state: Literal["selected", "not_selected", "abstain", "unavailable"]
        if mode == "rules":
            if score is not None:
                raise ValueError("Explicit rules cannot supply model scores")
            state = "selected" if evidence else "abstain"
        elif score is None:
            state = "unavailable"
        else:
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid uncalibrated score")
            state = (
                "selected"
                if score >= config[f"{mode}_positive"]
                else "not_selected"
                if score <= config[f"{mode}_negative"]
                else "abstain"
            )
        predictions.append(
            CategoryPrediction(
                category,
                mode,
                state,
                score,
                None if score is None else "uncalibrated_model_score",
                evidence,
                mode != "rules"
                and state in {"selected", "not_selected"}
                and (state == "selected") != bool(evidence),
            )
        )
    selected = tuple(p.category for p in predictions if p.status == "selected")
    return ComplaintPrediction(
        selected,
        tuple(predictions),
        "provisional_customer_reported_issues" if selected else "unknown_no_selected_category",
        mode,
        config["version"],
    )


class ComplaintModel:
    def __init__(
        self,
        config: dict[str, Any],
        mode: Mode,
        *,
        nli: ComplaintNLI | None = None,
        pipelines: dict[str, Any] | None = None,
    ) -> None:
        self.config, self.mode, self.nli = config, mode, nli
        self.pipelines = pipelines or {}

    def predict_texts(self, texts: list[str]) -> list[ComplaintPrediction]:
        validate_texts(texts)
        scores: list[list[float | None]] = [[None] * len(LABELS) for _ in texts]
        if self.mode == "nli":
            if self.nli is None:
                raise RuntimeError("NLI model unavailable")
            scores = self.nli.score_texts(texts).tolist()
        elif self.mode == "linear":
            from threadpoolctl import threadpool_limits

            with threadpool_limits(limits=self.config["cpu_threads"]):
                for j, category in enumerate(LABELS):
                    if category in self.pipelines:
                        pipeline = self.pipelines[category]
                        positive_index = list(pipeline.classes_).index(1)
                        values = pipeline.predict_proba(texts)[:, positive_index]
                        for i, value in enumerate(values):
                            scores[i][j] = float(value)
        return [
            assemble_prediction(text, self.mode, values, self.config)
            for text, values in zip(texts, scores, strict=True)
        ]


def load_complaint_model(
    directory: Path, *, mode: Mode = "nli", trusted: bool = False
) -> ComplaintModel:
    if mode not in {"nli", "linear", "rules"}:
        raise ValueError("Choose nli, linear, or rules explicitly")
    run = json.loads((directory / "run.json").read_text())
    for relative, checksum in run["files"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or file_hash(path) != checksum:
            raise ValueError("Complaint artifact checksum/path mismatch")
    config = run["configuration"]
    if run["taxonomy_version"] != TAXONOMY_VERSION or run["rule_version"] != RULE_VERSION:
        raise ValueError("Unsupported taxonomy or rule version")
    nli = ComplaintNLI(directory / "nli_checkpoint", config) if mode == "nli" else None
    pipelines: dict[str, Any] = {}
    if mode == "linear":
        if not trusted:
            raise ValueError("Joblib can execute code; explicitly trust your own local artifacts")
        import joblib

        for category, info in run["linear"].items():
            if info["available"]:
                pipelines[category] = joblib.load(directory / info["path"])
    return ComplaintModel(config, mode, nli=nli, pipelines=pipelines)
