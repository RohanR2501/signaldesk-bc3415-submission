"""Paired validation comparison, offline inference benchmarks and error candidates."""

import json
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl
import torch
from threadpoolctl import threadpool_limits

from signaldesk.config import PROJECT_ROOT
from signaldesk_data.source import file_hash, write_json
from signaldesk_ml.baselines import metrics
from signaldesk_ml.inference import load_selected, predict_texts
from signaldesk_ml.transformer_inference import LABELS, load_transformer
from signaldesk_ml.transformer_train import approved_data, seal_test


def macro_from_indices(truth: np.ndarray[Any, Any], prediction: np.ndarray[Any, Any]) -> float:
    matrix = np.bincount(truth * 3 + prediction, minlength=9).reshape(3, 3)
    denominator = matrix.sum(0) + matrix.sum(1)
    values = np.divide(
        2 * matrix.diagonal(), denominator, out=np.zeros(3, dtype=float), where=denominator != 0
    )
    return float(values.mean())


def paired_interval(truth: list[str], first: list[str], second: list[str]) -> dict[str, Any]:
    """Descriptive paired row bootstrap; not corrected for shared products/time dependence."""
    actual, a, b = [
        np.array([LABELS.index(label) for label in values]) for values in [truth, first, second]
    ]
    generator = np.random.default_rng(3415)
    differences = []
    for _ in range(1000):
        indices = generator.integers(0, len(truth), len(truth))
        differences.append(
            macro_from_indices(actual[indices], a[indices])
            - macro_from_indices(actual[indices], b[indices])
        )
    return {
        "replicates": 1000,
        "seed": 3415,
        "delta_macro_f1": macro_from_indices(actual, a) - macro_from_indices(actual, b),
        "percentile_95_interval": np.percentile(differences, [2.5, 97.5]).tolist(),
        "limitations": "Paired IID row bootstrap, no product/time clustering correction; "
        "validation reused for epoch selection; not independent test evidence",
    }


def benchmark(predict: Any, texts: list[str], cuda: bool) -> dict[str, Any]:
    predict(texts[:16])
    latencies = []
    started = perf_counter()
    for text in texts:
        if cuda:
            torch.cuda.synchronize()
        one_start = perf_counter()
        predict([text])
        if cuda:
            torch.cuda.synchronize()
        latencies.append((perf_counter() - one_start) * 1000)
    singles_seconds = perf_counter() - started
    batch_times = []
    for start in range(0, len(texts), 32):
        if cuda:
            torch.cuda.synchronize()
        one_start = perf_counter()
        predict(texts[start : start + 32])
        if cuda:
            torch.cuda.synchronize()
        batch_times.append(perf_counter() - one_start)
    return {
        "rows": len(texts),
        "warmup_rows": 16,
        "batch_size": 32,
        "single_p50_ms": float(np.percentile(latencies, 50)),
        "single_p95_ms": float(np.percentile(latencies, 95)),
        "singles_total_seconds": singles_seconds,
        "batch_amortized_ms_per_review": sum(batch_times) * 1000 / len(texts),
        "includes": "text preprocessing/tokenization, transfer if CUDA, forward, argmax; "
        "excludes artifact loading; warm sequential local calls",
    }


def analyse(config: dict[str, Any]) -> dict[str, Any]:
    train, scoring, integrity = approved_data(config)
    directory = PROJECT_ROOT / config["artifact_directory"]
    run = json.loads((directory / "run.json").read_text())
    if run["integrity"] != integrity:
        raise ValueError("Integrity changed before comparison")
    baseline_dir = PROJECT_ROOT / config["baseline_directory"]
    baseline_run = json.loads((baseline_dir / "run.json").read_text())
    baseline_predictions = pl.read_parquet(baseline_dir / "validation_predictions.parquet")
    outputs = pl.read_parquet(directory / "validation_predictions.parquet")
    if (
        file_hash(directory / "validation_predictions.parquet")
        != run["output_files"]["validation_predictions.parquet"]
    ):
        raise ValueError("Transformer predictions changed")
    if outputs["review_id"].to_list() != scoring["review_id"].to_list():
        raise ValueError("Transformer scoring IDs changed")
    truth = scoring["sentiment_proxy"].to_list()
    transformer = outputs["prediction"].to_list()
    baseline = baseline_predictions["logistic_regression_balanced"].to_list()
    baseline_metrics = metrics(truth, baseline)
    if metrics(truth, transformer) != run["metrics"]:
        raise ValueError("Transformer metrics changed")
    texts = scoring["model_text"].to_list()
    slices = {}
    for name, pattern in {
        "contrast_cues": r"\b(but|however|although|yet|though)\b",
        "negation_cues": r"\b(not|no|never|cannot|can't|don't|didn't|isn't|wasn't|won't)\b",
    }.items():
        chosen = [i for i, text in enumerate(texts) if re.search(pattern, text, re.I)]
        slices[name] = {
            "rows": len(chosen),
            "regex": pattern,
            "distilbert": metrics([truth[i] for i in chosen], [transformer[i] for i in chosen]),
            "balanced_lr": metrics([truth[i] for i in chosen], [baseline[i] for i in chosen]),
            "limitation": "Lexical cue slice, not independently labelled linguistic ability",
        }
    examples = []
    # Keep IDs already discussed in the baseline report for a traceable paired review.
    reference_ids = {
        f"amazon2023:appliances:{suffix}" for suffix in ["1513030", "126740", "2005834", "502585"]
    }
    for i, row in enumerate(scoring.iter_rows(named=True)):
        if row["review_id"] in reference_ids:
            examples.append(
                {
                    "review_id": row["review_id"],
                    "text": texts[i],
                    "proxy": truth[i],
                    "balanced_lr": baseline[i],
                    "distilbert": transformer[i],
                    "selection": "Previously discussed Phase 2A error",
                }
            )
    for actual in LABELS:
        for predicted in LABELS:
            if actual == predicted:
                continue
            indices = [
                i for i in range(len(truth)) if truth[i] == actual and transformer[i] == predicted
            ]
            indices.sort(key=lambda i: scoring["review_id"][i])
            for i in indices[:2]:
                examples.append(
                    {
                        "review_id": scoring["review_id"][i],
                        "text": texts[i],
                        "proxy": truth[i],
                        "balanced_lr": baseline[i],
                        "distilbert": transformer[i],
                        "selection": "First two sorted IDs per transformer error cell",
                    }
                )
    write_json(directory / "paired_error_examples.json", examples)
    started = perf_counter()
    predictor = load_transformer(directory)
    torch.cuda.synchronize()
    load_seconds = perf_counter() - started
    # Identical seeded 128-record validation sample for both latency measurements.
    indices = np.random.default_rng(3415).choice(scoring.height, 128, replace=False).tolist()
    benchmark_texts = [texts[i] for i in indices]
    predictor.predict_texts(train["model_text"].head(16).to_list())
    transformer_latency = benchmark(predictor.predict_texts, benchmark_texts, True)
    started = perf_counter()
    pipeline = load_selected(baseline_dir, trusted=True)
    baseline_load = perf_counter() - started
    with threadpool_limits(limits=4):
        baseline_latency = benchmark(
            lambda batch: predict_texts(pipeline, batch), benchmark_texts, False
        )
    smoke_texts = [
        "This replacement works perfectly. I am very happy with it.",
        "It broke immediately and does not work. Terrible purchase.",
        "It is okay, but nothing special.",
    ]
    smoke = predictor.predict_texts(smoke_texts)
    comparison = {
        "analysis_code_sha256": file_hash(Path(__file__)),
        "distilbert": run["metrics"],
        "balanced_lr": baseline_metrics,
        "paired_bootstrap": paired_interval(truth, transformer, baseline),
        "slices": slices,
        "paired_outcomes": {
            "both_correct": sum(
                a == y and b == y for y, a, b in zip(truth, transformer, baseline, strict=True)
            ),
            "transformer_only_correct": sum(
                a == y and b != y for y, a, b in zip(truth, transformer, baseline, strict=True)
            ),
            "baseline_only_correct": sum(
                a != y and b == y for y, a, b in zip(truth, transformer, baseline, strict=True)
            ),
        },
        "inference": {
            "distilbert_cuda_fp16": transformer_latency,
            "balanced_lr_cpu_4_threads": baseline_latency,
            "distilbert_load_seconds": load_seconds,
            "baseline_load_seconds": baseline_load,
            "load_context": "Warm OS disk cache; checksum validation included",
            "sample_seed": 3415,
            "sample_ids": [scoring["review_id"][i] for i in indices],
            "smoke_predictions": smoke,
        },
        "baseline_artifact_bytes": baseline_run["models"]["logistic_regression_balanced"][
            "artifact_bytes"
        ],
        "distilbert_checkpoint_bytes": run["checkpoint_bytes"],
        "training_seconds": run["training_seconds"],
        "validation_membership_sha256": run["integrity"]["membership_sha256"],
        "test_accessed": False,
    }
    write_json(directory / "comparison.json", comparison)
    return comparison


def main() -> None:
    sys.addaudithook(seal_test)
    config = json.loads((PROJECT_ROOT / "config/sentiment-transformer.json").read_text())
    result = analyse(config)
    print(
        json.dumps(
            {
                "status": "passed",
                "delta_macro_f1": result["paired_bootstrap"]["delta_macro_f1"],
                "inference": result["inference"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
