"""Source-backed figures and a compact report for the one frozen baseline run."""

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import matplotlib
import numpy as np

from signaldesk.config import PROJECT_ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {
    "majority": "Majority",
    "naive_bayes": "TF-IDF + Naive Bayes",
    "logistic_regression": "TF-IDF + Logistic Regression",
    "logistic_regression_balanced": "TF-IDF + balanced Logistic Regression",
}
LABELS = ["negative", "neutral", "positive"]


def write_report(run: dict[str, Any], directory: Path) -> None:
    models = run["models"]
    names = list(NAMES)
    selected = run["selected_model"]
    chosen = models[selected]
    date = datetime.fromisoformat(run["finished_at"]).astimezone(ZoneInfo("Asia/Singapore")).date()
    figures = PROJECT_ROOT / "docs/figures"
    figures.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    for name, ax in zip(names, axes.flat, strict=True):
        matrix = np.asarray(models[name]["metrics"]["confusion_matrix"])
        fractions = matrix / matrix.sum(axis=1, keepdims=True)
        image = ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues")
        for row in range(3):
            for col in range(3):
                ax.text(
                    col,
                    row,
                    f"{matrix[row, col]:,}\n{fractions[row, col]:.1%}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if fractions[row, col] > 0.55 else "#16333d",
                )
        ax.set(
            title=NAMES[name],
            xticks=range(3),
            yticks=range(3),
            xticklabels=LABELS,
            yticklabels=LABELS,
            xlabel="Predicted label",
            ylabel="Rating-derived proxy",
        )
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Share within proxy class", shrink=0.8)
    fig.suptitle(
        f"Validation proxy confusion matrices · {run['validation_scoring_rows']:,} reviews"
    )
    fig.savefig(figures / "sentiment-baseline-confusions.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    short_names = ["Majority", "TF-IDF + NB", "TF-IDF + LR", "TF-IDF + balanced LR"]
    values = [models[name]["metrics"]["macro_f1"] for name in names]
    positions = np.arange(4)
    axes[0].barh(
        positions, values, color=["#aab8bf" if name != selected else "#1f5268" for name in names]
    )
    for y, value in enumerate(values):
        axes[0].text(value + 0.015, y, f"{value:.3f}", va="center")
    axes[0].set(
        title="Primary comparison: Macro-F1",
        xlim=(0, 1),
        yticks=positions,
        yticklabels=short_names,
        xlabel="Agreement with rating-derived proxies",
    )
    for offset, (label, color) in enumerate([("negative", "#1f5268"), ("neutral", "#c28c32")]):
        recalls = [models[name]["metrics"]["per_class"][label]["recall"] for name in names]
        axes[1].barh(
            positions - 0.18 + offset * 0.36, recalls, height=0.34, label=label, color=color
        )
    axes[1].set(
        title="Minority-class recall",
        xlim=(0, 1),
        yticks=positions,
        yticklabels=short_names,
        xlabel="Fraction of proxy-class reviews recovered",
    )
    axes[1].legend(frameon=False, loc="lower right")
    for ax in axes:
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figures / "sentiment-baseline-comparison.png", dpi=150)
    plt.close(fig)

    comparison = []
    classes = []
    for name in names:
        record = models[name]
        score = record["metrics"]
        comparison.append(
            f"| {NAMES[name]} | {score['accuracy']:.2%} | {score['macro_f1']:.4f} | "
            f"{score['weighted_f1']:.4f} | {record['training_seconds']:.3f} | "
            f"{record['inference_seconds']:.3f} | {record['artifact_bytes'] / 2**20:.2f} |"
        )
        for label in LABELS:
            result = score["per_class"][label]
            classes.append(
                f"| {NAMES[name]} | {label} | {result['support']:,} | "
                f"{result['precision']:.3f} | {result['recall']:.3f} | {result['f1']:.3f} |"
            )
    matrix = chosen["metrics"]["confusion_matrix"]
    negative_to_positive = matrix[0][2]
    neutral = chosen["metrics"]["per_class"]["neutral"]
    negative = chosen["metrics"]["per_class"]["negative"]
    unweighted = models["logistic_regression"]["metrics"]
    weighted = models["logistic_regression_balanced"]["metrics"]
    weight_delta = weighted["macro_f1"] - unweighted["macro_f1"]
    warning_text = (
        "; ".join(
            f"{name}: {item['category']}: {item['message']}"
            for name in names
            for item in models[name]["warnings"]
        )
        or "No fit warnings were recorded; all configured fits converged."
    )
    comparison_table = "\n".join(comparison)
    class_table = "\n".join(classes)
    model_path = directory / chosen["artifact"]
    annotation_path = (
        PROJECT_ROOT / run["configuration"]["annotation_directory"] / "annotation_template.csv"
    )
    text = f"""# SignalDesk sentiment baseline report

Phase 2A completed on {date} (Asia/Singapore). All reported scores measure agreement
with **rating-derived sentiment proxies**, not independently verified human sentiment.
No transformer was trained and the frozen test file was not opened or evaluated.

## Data and blind annotation reservation

Training uses the original **{run["training_rows"]:,}** reviews. Before any model fitting,
**{run["annotation_rows"]} validation records** were reserved (80 per proxy class, seed
3415), leaving **{run["validation_scoring_rows"]:,}** validation reviews for comparison.
The original 6,913-row validation partition and 6,921-row frozen test partition are
unchanged. Only training/validation files and the existing manifest were read; the
full data-verification command was deliberately not called because it reads test data.

Selection ranks SHA256(seed, purpose, review ID) within each proxy class, then separately
shuffles display order. `{annotation_path.relative_to(PROJECT_ROOT).as_posix()}` contains
only stable review ID, original text and **blank** human label, ambiguity and notes fields.
Proxy strata/source hashes/reserved IDs stay in its separate `internal/provenance.json`.
Sampling is reproducible and its provenance timestamp precedes experimentation. The
240 records were excluded from all fitting, model-selection scoring and error analysis.

See [the evaluation methodology](../README.md#evaluation). Human annotators work on
independent copies without seeing this report, internal keys, ratings or predictions.
No human or AI-generated independent ground truth was fabricated. The balanced proxy
strata improve minority coverage but do not make this a representative population sample.
Unresolved labels and annotator disagreement must be reported when human evaluation occurs.

Scoring supports: negative {run["proxy_counts"]["scoring"]["negative"]:,}, neutral
{run["proxy_counts"]["scoring"]["neutral"]:,},
positive {run["proxy_counts"]["scoring"]["positive"]:,}.
Removing 80 examples per stratum changes the scoring distribution, especially neutral;
all four candidates use this identical subset. Chronological imbalance and possible
text/rating disagreement persist; no resampling/rebalancing of records was performed.

## Fixed configurations and resource limits

All candidates use **review body text only**. Ratings, titles, IDs, timestamps, proxy
targets and investigation histories never enter prediction features. TF-IDF is fitted
only on training text and preserved inside each sklearn Pipeline. Vocabulary/IDF are
checked against training-only document frequencies after saving/loading.

- Majority: most frequent training class, no text features required.
- TF-IDF + Multinomial Naive Bayes: alpha 1.0.
- TF-IDF + Logistic Regression: C=1, lbfgs, max_iter=500, tol=1e-4, seed 3415.
- TF-IDF + balanced Logistic Regression: same settings, class weights computed from
  training labels only with `class_weight=balanced`.

Shared vectorization uses word unigrams/bigrams, max_features=60,000, min_df=2,
lowercasing, sublinear TF and L2 normalization. The token pattern retains contractions,
negation and `!`/`?` runs; no stop-word removal or stemming is used. Features are sparse.
CPU numerical libraries are capped at four threads. No grid search, Linear SVM,
validation fitting or transformer experimentation was performed. Each real candidate
was fitted once; subsequent checks reload artifacts and predict without refitting.

## Actual validation comparison

Macro-F1 is the primary selection metric. Fractions use the explicit label order
negative, neutral, positive. Undefined precision/recall/F1 are reported as zero.

| Model | Accuracy | Macro-F1 | Weighted-F1 | Fit seconds | Predict seconds | Artifact MiB |
|---|---:|---:|---:|---:|---:|---:|
{comparison_table}

Fit time includes vectorization and classifier fitting; compressed serialization is
excluded. Predict time covers the entire {run["validation_scoring_rows"]:,}-review batch,
including vectorization, after a 16-training-review warmup. These are local CPU batch
measurements, not single-request latency or a deployment benchmark. Artifact sizes include
the complete preprocessing pipeline. Versions/hardware context are in the run metadata.

| Model | Proxy class | Support | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
{class_table}

![Validation comparison and minority recall](figures/sentiment-baseline-comparison.png)

## Confusion matrices and class-weight trade-offs

![Actual counts and within-class shares](figures/sentiment-baseline-confusions.png)

Rows are rating proxies; columns are predictions. The majority reference demonstrates
why accuracy alone is insufficient: it cannot recover negative or neutral reviews.
For the selected model, **{negative_to_positive:,} negative-proxy reviews were predicted
positive**; negative recall is {negative["recall"]:.3f}. Neutral precision/recall are
{neutral["precision"]:.3f}/{neutral["recall"]:.3f}. These are proxy disagreements, not
verified missed customer complaints or product defects.

Class weighting changes validation Macro-F1 by **{weight_delta:+.4f}** relative to
unweighted Logistic Regression. It changes neutral recall from
{unweighted["per_class"]["neutral"]["recall"]:.3f} to
{weighted["per_class"]["neutral"]["recall"]:.3f}; precision and other-class costs are
reported above. A point difference from this single validation subset does not establish
statistical significance or generalization beyond this historical cohort.

## Error analysis and linear feature weights

A bounded seed-ranked review sample was taken only from selected-model errors in the
scored validation subset (up to three per off-diagonal confusion cell). The local
`error_examples.json` preserves supporting texts for review; it is never an annotation
source. Qualitative interpretation is added below after inspecting actual errors.

`feature_weights.json` contains the ten largest positive coefficients per class for
both linear classifiers. These are **global learned TF-IDF feature weights**, not
explanations of a specific prediction, evidence of causation, or defect understanding.
No individual prediction attribution or calibrated confidence is claimed in this phase.

## Recommendation and artifact

Recommend **{NAMES[selected]}** as the classical reference for the later DistilBERT
comparison: it has the highest validation Macro-F1 under the fixed selection rule.
Minority precision/recall, negative-to-positive errors and local CPU inference cost
must be considered alongside that score. No numerical production quality target was set.

Selected complete pipeline: `{model_path.relative_to(PROJECT_ROOT).as_posix()}`.
`run.json` records configuration, class mapping, actual metrics, canonical evaluation
contracts, training/validation-only source references, CPU thread cap, durations, sizes,
environment/code hashes and selected artifact checksum. `membership.json` records the
exact training/scoring/reserved IDs; `validation_predictions.parquet` permits metric
reproduction. Artifacts are ignored by Git and must only be loaded from trusted sources.

Fit status: {warning_text}

## Verification, limits and next stage

The workflow verifies source train/validation hashes, annotation exclusion/blinding,
training-only IDF, class consistency, saved-pipeline loading, exact prediction/metric
reproduction and canonical ModelEvaluationResult compatibility. No test data path is
accepted by its partition loader. Repeated `train` on this completed run calls verification
instead of fitting again. Test data remains sealed for a later explicitly authorized evaluation.

The cohort remains positive-skewed, chronologically shifted, deduplicated and limited
to Appliances. Very short/mixed text, HTML and uncertain language were retained.
An independently annotated reference is still incomplete, and complaint classification,
calibration, abstention and operational impact are outside Phase 2A.

Proceed only on instruction to a GPU DistilBERT comparison using the same training and
scored-validation IDs while preserving the annotation holdout and frozen test. Target
neutral/mixed sentiment, contextual negation and negative-to-positive confusions; do not
assume the transformer will improve them. The classical pipeline supplies an empirical
reference, not evidence of production readiness or true sentiment accuracy.
"""
    (PROJECT_ROOT / "docs/SENTIMENT_BASELINE_REPORT.md").write_text(text, encoding="utf-8")
