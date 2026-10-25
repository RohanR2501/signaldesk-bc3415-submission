"""Read-only three-model comparison; write Phase 2C evidence only to the new run."""

import gc
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import torch
from threadpoolctl import threadpool_limits

from signaldesk.config import PROJECT_ROOT
from signaldesk_data.source import file_hash, write_json
from signaldesk_ml.baselines import metrics
from signaldesk_ml.inference import load_selected, predict_texts
from signaldesk_ml.transformer_analysis import benchmark, paired_interval
from signaldesk_ml.transformer_inference import LABELS, load_transformer
from signaldesk_ml.transformer_train import approved_data, controlled_reference, seal_test

NAMES = {
    "balanced_lr": "Balanced LR",
    "unweighted_distilbert": "Unweighted DistilBERT",
    "weighted_distilbert": "Weighted DistilBERT",
}


def verify_predictions(
    directory: Path, run: dict[str, Any], scoring: pl.DataFrame, column: str = "prediction"
) -> list[str]:
    path = directory / "validation_predictions.parquet"
    files = run["output_files"] if "output_files" in run else run["files"]
    if file_hash(path) != files[path.name]:
        raise ValueError("Reference prediction checksum mismatch")
    frame = pl.read_parquet(path)
    if any(
        frame[key].to_list() != scoring[key].to_list() for key in ["review_id", "sentiment_proxy"]
    ):
        raise ValueError("Three-model scoring memberships or labels differ")
    return list(frame[column].to_list())


def compare(config: dict[str, Any]) -> dict[str, Any]:
    reference = controlled_reference(config)
    if reference is None:
        raise ValueError("Phase 2C requires the original unweighted reference")
    train, scoring, integrity = approved_data(config)
    directory = PROJECT_ROOT / config["artifact_directory"]
    run = json.loads((directory / "run.json").read_text())
    reference_dir = PROJECT_ROOT / config["reference_directory"]
    baseline_dir = PROJECT_ROOT / config["baseline_directory"]
    baseline = json.loads((baseline_dir / "run.json").read_text())
    if reference["integrity"] != integrity or run["integrity"] != integrity:
        raise ValueError("Experiment source integrity differs")
    if run["reference_run_sha256"] != file_hash(reference_dir / "run.json"):
        raise ValueError("Original transformer run changed")
    truth = scoring["sentiment_proxy"].to_list()
    predictions = {
        "balanced_lr": verify_predictions(
            baseline_dir, baseline, scoring, "logistic_regression_balanced"
        ),
        "unweighted_distilbert": verify_predictions(reference_dir, reference, scoring),
        "weighted_distilbert": verify_predictions(directory, run, scoring),
    }
    models: dict[str, Any] = {}
    for name, predicted in predictions.items():
        score = metrics(truth, predicted)
        expected = (
            baseline["models"]["logistic_regression_balanced"]["metrics"]
            if name == "balanced_lr"
            else reference["metrics"]
            if name == "unweighted_distilbert"
            else run["metrics"]
        )
        if score != expected:
            raise ValueError("Saved model metrics do not reproduce")
        models[name] = {
            "metrics": score,
            "negative_to_positive": score["confusion_matrix"][0][2],
            "false_neutral": sum(score["confusion_matrix"][i][1] for i in [0, 2]),
        }
    sample = np.random.default_rng(3415).choice(scoring.height, 128, replace=False).tolist()
    texts = [scoring["model_text"][i] for i in sample]
    # Measure serially, after training and verification; preserve every reference artifact.
    started = perf_counter()
    pipeline = load_selected(baseline_dir, trusted=True)
    models["balanced_lr"]["load_seconds"] = perf_counter() - started
    with threadpool_limits(limits=4):
        models["balanced_lr"]["latency"] = benchmark(
            lambda batch, saved_pipeline=pipeline: predict_texts(saved_pipeline, batch),
            texts,
            False,
        )
    models["balanced_lr"]["artifact_bytes"] = baseline["models"]["logistic_regression_balanced"][
        "artifact_bytes"
    ]
    del pipeline
    for name, path, metadata in [
        ("unweighted_distilbert", reference_dir, reference),
        ("weighted_distilbert", directory, run),
    ]:
        torch.cuda.empty_cache()
        started = perf_counter()
        predictor = load_transformer(path)
        torch.cuda.synchronize()
        models[name]["load_seconds"] = perf_counter() - started
        predictor.predict_texts(train["model_text"].head(16).to_list())
        models[name]["latency"] = benchmark(predictor.predict_texts, texts, True)
        models[name]["artifact_bytes"] = metadata["checkpoint_bytes"]
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
    weighted = predictions["weighted_distilbert"]
    old = predictions["unweighted_distilbert"]
    transitions = {
        "weighted_only_correct": sum(
            w == y and o != y for y, o, w in zip(truth, old, weighted, strict=True)
        ),
        "unweighted_only_correct": sum(
            w != y and o == y for y, o, w in zip(truth, old, weighted, strict=True)
        ),
        "neutral_recovered": sum(
            y == "neutral" and o != y and w == y
            for y, o, w in zip(truth, old, weighted, strict=True)
        ),
        "neutral_lost": sum(
            y == "neutral" and o == y and w != y
            for y, o, w in zip(truth, old, weighted, strict=True)
        ),
        "negative_positive_fixed": sum(
            y == "negative" and o == "positive" and w != "positive"
            for y, o, w in zip(truth, old, weighted, strict=True)
        ),
        "negative_positive_added": sum(
            y == "negative" and o != "positive" and w == "positive"
            for y, o, w in zip(truth, old, weighted, strict=True)
        ),
    }
    result = {
        "measured_at": datetime.now(UTC).isoformat(),
        "analysis_code_sha256": file_hash(Path(__file__)),
        "models": models,
        "transitions_vs_unweighted": transitions,
        "selected_research_model": max(
            models, key=lambda name: models[name]["metrics"]["macro_f1"]
        ),
        "paired_bootstrap": {
            name: paired_interval(truth, weighted, predictions[name])
            for name in ["balanced_lr", "unweighted_distilbert"]
        },
        "latency_sample_ids": [scoring["review_id"][i] for i in sample],
        "latency_sample_seed": 3415,
        "latency_context": "Serial local warm measurements, "
        "shared laptop, CPU four numerical threads or CUDA fp16; "
        "warm-cache load includes hash checking",
        "integrity": integrity,
        "test_accessed": False,
    }
    write_json(directory / "comparison.json", result)
    return result


def write_report(result: dict[str, Any], run: dict[str, Any]) -> None:
    models = result["models"]
    weighting = run["class_weighting"]
    weighted = models["weighted_distilbert"]["metrics"]
    old = models["unweighted_distilbert"]["metrics"]
    date = (
        datetime.fromisoformat(result["measured_at"]).astimezone(ZoneInfo("Asia/Singapore")).date()
    )
    selected = NAMES[result["selected_research_model"]]
    rows = []
    for key, title in [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("weighted_f1", "Weighted-F1"),
    ]:
        rows.append(
            f"| {title} | " + " | ".join(f"{m['metrics'][key]:.4f}" for m in models.values()) + " |"
        )
    for title, key in [
        ("Negative → positive errors", "negative_to_positive"),
        ("False neutral predictions", "false_neutral"),
    ]:
        rows.append(f"| {title} | " + " | ".join(str(m[key]) for m in models.values()) + " |")
    classes = []
    confusions = []
    costs = []
    for name, model in models.items():
        for index, label in enumerate(LABELS):
            metric = model["metrics"]["per_class"][label]
            classes.append(
                f"| {NAMES[name]} | {label} | {metric['support']:,} | "
                f"{metric['precision']:.4f} | {metric['recall']:.4f} | {metric['f1']:.4f} |"
            )
            confusions.append(
                f"| {NAMES[name]} | {label} | "
                + " | ".join(f"{count:,}" for count in model["metrics"]["confusion_matrix"][index])
                + " |"
            )
        latency = model["latency"]
        costs.append(
            f"| {NAMES[name]} | {model['artifact_bytes']:,} | "
            f"{latency['single_p50_ms']:.3f} | {latency['single_p95_ms']:.3f} | "
            f"{latency['batch_amortized_ms_per_review']:.3f} | {model['load_seconds']:.3f} |"
        )
    deltas = []
    for key in ["accuracy", "macro_f1", "weighted_f1"]:
        deltas.append(f"| {key} | {weighted[key] - old[key]:+.4f} |")
    for label in LABELS:
        for key in ["precision", "recall", "f1"]:
            delta = weighted["per_class"][label][key] - old["per_class"][label][key]
            deltas.append(f"| {label} {key} | {delta:+.4f} |")
    intervals = []
    for name, record in result["paired_bootstrap"].items():
        low, high = record["percentile_95_interval"]
        intervals.append(
            f"- Weighted minus {NAMES[name]} Macro-F1: "
            f"**{record['delta_macro_f1']:+.4f}**, descriptive 95% interval "
            f"**[{low:+.4f}, {high:+.4f}]**."
        )
    history = "\n".join(
        f"| {e['epoch']} | {e['training_seconds']:.3f} | "
        f"{e['validation_seconds']:.3f} | {e['metrics']['macro_f1']:.4f} |"
        for e in run["history"]
    )
    neutral = weighted["per_class"]["neutral"]
    matrix = weighted["confusion_matrix"]
    original = old["confusion_matrix"]
    change = result["transitions_vs_unweighted"]
    text = f"""# SignalDesk sentiment optimisation report

Phase 2C completed on **{date} (Asia/Singapore)**: one controlled two-epoch class-weighted
DistilBERT experiment. All metrics are **rating-proxy agreement**, not independently
verified human sentiment. The prior models and reports were preserved; Phase 3 was not started.

## Hypothesis and controlled method

Moderate square-root inverse-frequency weighting might improve minority recognition
without excessive neutral predictions. This was tested once, without a grid search.
The original pretrained encoder revision `{run["configuration"]["revision"]}` and seeded
new three-class head were used, not the already fine-tuned checkpoint. Cached encoder,
tokenizer/config files match the original experiment's checksums.

Unchanged conditions: seed 3415; review body `model_text` only; pretrained tokenizer;
right-truncation at 256 tokens; batch 16; accumulation 2 (effective 32); AdamW 2e-5,
weight decay 0.01, epsilon 1e-8; linear schedule with 10% warmup; clip norm 1;
two epochs; CUDA fp16 autocast/GradScaler with fp32 weights; validation batch 32.
Select highest validation Macro-F1, earliest exact tie. A configuration guard rejects
other training-condition changes. The intended experimental change is the loss.

Weights use training proportions f[c] = n[c]/32,230, raw[c] = 1/sqrt(f[c]), then
w[c] = raw[c]/sum(f[k]*raw[k]). The training-frequency-weighted mean is
**{weighting["training_frequency_weighted_mean"]:.16g}**.

| Class index | Label | Training count | Frequency | Exact float64 weight | CUDA float32 weight |
|---|---|---:|---:|---:|---:|
"""
    for i, label in enumerate(LABELS):
        text += (
            f"| {i} | {label} | {weighting['counts'][i]:,} | {weighting['frequencies'][i]:.12f} | "
            f"{weighting['weights'][i]:.16g} | {weighting['cuda_float32_weights'][i]:.16g} |\n"
        )
    text += f"""
Weighted cross-entropy uses PyTorch's sum-of-target-weights mean reduction. Accumulation
scales each microbatch by its target-weight mass within the effective batch, including
the remainder. A synthetic gradient test confirms equivalence to full-batch weighted
cross-entropy. Weight computation never uses validation labels/frequencies.

## Data, annotation and leakage safeguards

Exactly **32,230 training**, **6,673 scored validation**, **240 reserved annotation** IDs
match Phase 2A/2B, including row ordering and validation proxy labels. Source train and
validation checksums, the original JSON manifest, memberships and annotation provenance
are checked. No validation fitting, resampling or partition changes occurred.

The **6,921-row frozen test**, full sentiment parquet and full split manifest remained
unopened; no full-data verifier was run. The permitted loader accepts train/validation
only, and the entry point also installs the sealed-file Python open guard. Ratings,
titles, IDs, timestamps and product histories do not enter prediction features.

The 240-row blinded master and provenance remain byte-identical and human fields blank.
Minimum later manual work: one independent human labels all 240 records on a copy using
the existing guide; a second independently labels a pilot and a documented evaluation
subset. Measure agreement, adjudicate disagreements while blind to ratings/model outputs,
record provenance and freeze resolved labels/exclusions before evaluating models. Keep
unresolvable cases explicit and report class support. The proxy-stratified sample is not
a population-representative estimate. No new annotation system or labels were generated.

## Actual GPU run

GPU: **{run["gpu"]}**, device **{run["cuda_device"]}**, CUDA **{run["cuda_runtime"]}**,
PyTorch **{run["environment"]["torch"]}**, Transformers **{run["environment"]["transformers"]}**.
Compatibility forward/backward passed. The six-step weighted optimizer memory probe
took **{run["preflight"]["seconds"]:.3f} s**, peaked at
**{run["preflight"]["peak_allocated_bytes"] / 2**30:.2f} GiB allocated /
{run["preflight"]["peak_reserved_bytes"] / 2**30:.2f} GiB reserved**, and restored original
weights/RNG before fitting. Batch size remained 16; no CPU fallback occurred.

| Epoch | Training seconds | Validation seconds | Macro-F1 |
|---|---:|---:|---:|
{history}

Selected epoch: **{run["selected_epoch"]}**. Training-loop time:
**{run["training_seconds"]:.3f} s ({run["training_seconds"] / 60:.2f} min)**;
training/validation/checkpoint wall loop: **{run["training_and_evaluation_wall_seconds"]:.3f} s**.
Download/tokenization/preflight and reload verification are excluded from training time.
Training peak: **{run["peak_allocated_bytes"] / 2**30:.2f} GiB allocated /
{run["peak_reserved_bytes"] / 2**30:.2f} GiB reserved**, excluding other GPU processes.
A 1,500-second wall guard enforced the 25-minute ceiling. The preflight runtime estimate
uses the prior measured dynamic-padding run scaled by probe time; its worst-length
estimate is also retained. Seeded SDPA backward can remain nondeterministic under the
same `warn_only=True` setting used in Phase 2B; no seed-variance estimate is available.

## Three-model validation comparison

All models score the identical 6,673 reviews. Accuracy and F1 values are fractions.

| Metric | Balanced LR | Unweighted DistilBERT | Weighted DistilBERT |
|---|---:|---:|---:|
{chr(10).join(rows)}

| Model | Proxy class | Support | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
{chr(10).join(classes)}

Weighted minus unweighted, in absolute fraction units (multiply by 100 for percentage points):

| Metric | Difference |
|---|---:|
{chr(10).join(deltas)}

{chr(10).join(intervals)}

Intervals use the existing seeded 1,000-replicate paired IID row bootstrap. They do not
correct for shared-product/time correlation or repeated validation selection across
epochs/experiments and cannot establish independent generalisation or business benefit.

## Confusions and operational trade-offs

Rows are rating-derived proxies, columns predictions, ordered negative/neutral/positive.

| Model | Proxy row | Pred. negative | Pred. neutral | Pred. positive |
|---|---|---:|---:|---:|
{chr(10).join(confusions)}

Weighted neutral recall is **{neutral["recall"]:.2%}**, precision **{neutral["precision"]:.2%}**,
F1 **{neutral["f1"]:.4f}**. Correct neutrals change **{original[1][1]} → {matrix[1][1]}**;
false neutral predictions change **{models["unweighted_distilbert"]["false_neutral"]} →
{models["weighted_distilbert"]["false_neutral"]}**. Paired neutral recoveries/losses are
**{change["neutral_recovered"]} / {change["neutral_lost"]}**. These counts show the workload
trade-off without presuming an acceptable operating threshold.

Negative → positive errors change **{original[0][2]} → {matrix[0][2]}** versus unweighted
and compare with **{models["balanced_lr"]["negative_to_positive"]}** for LR, each out of
1,310 negative proxies. The weighted model fixes **{change["negative_positive_fixed"]}**
of the original model's such errors and introduces **{change["negative_positive_added"]}**
new ones. This particularly consequential error remains separate from aggregate recall.
Weighted-only versus unweighted-only correct proxy matches: **{change["weighted_only_correct"]}
/ {change["unweighted_only_correct"]}**. Positive precision/recall/F1 and their differences
above expose any cost to positive-class performance.

## Saved-model inference cost

| Model | Artifact bytes | Single p50 ms | Single p95 ms | Batch-32 ms/review | Load s |
|---|---:|---:|---:|---:|---:|
{chr(10).join(costs)}

Fresh serial measurements use the same seeded 128 validation IDs and 16-record warmup
for all models. CPU LR uses four numerical threads; transformers use CUDA fp16.
Times include tokenization/vectorization, transfer, forward and label selection, with
CUDA synchronization; loading is separate and includes hash validation with a warm OS
cache. These shared-laptop measurements are not service guarantees. Prior benchmark
files were not rewritten. Weighted inference reuses the same local-only safetensors
loader; class weighting adds no inference-time computation.

## Recommendation, limitations and Phase 3

Provisional research choice by validation Macro-F1: **{selected}**. Accuracy alone does
not select the model. No operational model is endorsed without independent human-labelled
evaluation and an explicit acceptable error-cost policy. Carry forward the neutral,
negative → positive and positive-class trade-offs above rather than interpreting the
research ranking as verified problem detection. Positive sentiment can coexist with
a serious customer-reported complaint; sentiment must not suppress a separate complaint task.

Retain balanced LR as a small, fast CPU fallback candidate and reproducible reference.
Selecting that fallback later must be explicit; transformer code never silently falls
back to CPU. Preserve the unweighted transformer regardless of the new model's result.
Further training is not launched; independent annotation and error adjudication are a
higher-priority next step than automatic repeated tuning of the same proxy validation set.

Technical integration is feasible through the saved text-only inference interface after
explicit Phase 3 authorization. Operational readiness is not established. Carry forward
rating-proxy ambiguity, positive-skewed training, the small 279-neutral validation support,
repeated validation selection, English/domain limits, 256-token truncation, uncalibrated
outputs, known-product chronological evaluation and duplicate/repeated-author limits.
Complaints require separate labels/evaluation. Keep test sealed until authorised final evaluation.

## Artifacts and verification

New run: `artifacts/sentiment/distilbert-weighted-v1/`:
`checkpoint/`, `run.json`, `started.json`, `epoch_history.json`,
`validation_predictions.parquet`, `verification.json`, `comparison.json`.
The original inference loader loads this checkpoint without network access.

Verification commands/results are recorded below after final checks. All commands use
the frozen environment with `--extra data --extra gpu --extra baselines`; generated
weights, predictions and memberships remain ignored by Git.
"""
    (PROJECT_ROOT / "docs/SENTIMENT_OPTIMISATION_REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    sys.addaudithook(seal_test)
    config = json.loads((PROJECT_ROOT / "config/sentiment-transformer-weighted.json").read_text())
    result = compare(config)
    run = json.loads((PROJECT_ROOT / config["artifact_directory"] / "run.json").read_text())
    write_report(result, run)
    print(
        json.dumps(
            {
                "selected_research_model": result["selected_research_model"],
                "metrics": {
                    name: model["metrics"]["macro_f1"] for name, model in result["models"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
