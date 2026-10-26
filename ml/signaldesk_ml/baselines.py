"""CPU-only sentiment baselines with a pre-experiment blind annotation reservation.

Only train.parquet, validation.parquet and the dataset manifest are read. No full-data
verification helper is called because Phase 2A explicitly prohibits opening test data.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import warnings
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from signaldesk.config import PROJECT_ROOT
from signaldesk.contracts import EvaluationMetric, ModelEvaluationResult
from signaldesk_data.source import file_hash, write_json
from signaldesk_ml.inference import load_selected, predict_texts

LABELS = ["negative", "neutral", "positive"]
INPUT_COLUMNS = [
    "review_id",
    "text",
    "model_text",
    "sentiment_proxy",
    "split",
    "duplicate_group_id",
]
ANNOTATION_COLUMNS = ["review_id", "text", "human_sentiment", "ambiguity_flag", "annotator_notes"]
MODEL_NAMES = ["majority", "naive_bayes", "logistic_regression", "logistic_regression_balanced"]


def id_hash(identifiers: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(identifiers)).encode()).hexdigest()


def load_partition(directory: Path, name: str, manifest: dict[str, Any]) -> pl.DataFrame:
    if name not in {"train", "validation"}:
        raise ValueError("Phase 2A permits training and validation only; test access is forbidden")
    path = directory / f"{name}.parquet"
    if file_hash(path) != manifest["files"][path.name]["sha256"]:
        raise ValueError(f"Frozen {name} checksum mismatch")
    frame = pl.read_parquet(path, columns=INPUT_COLUMNS)
    if set(frame["split"].unique()) != {name}:
        raise ValueError("Unexpected partition membership")
    if set(frame["sentiment_proxy"].unique()) != set(LABELS):
        raise ValueError("Expected all three rating-derived sentiment proxies")
    if any(not text.strip() for text in frame["model_text"]) or not frame["model_text"].equals(
        frame["text"]
    ):
        raise ValueError("Model text must be nonblank original review text")
    if frame["review_id"].n_unique() != frame.height:
        raise ValueError("Duplicate review identifier")
    return frame


def load_data(config: dict[str, Any]) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    directory = PROJECT_ROOT / config["dataset_directory"]
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = load_partition(directory, "train", manifest)
    validation = load_partition(directory, "validation", manifest)
    if (
        train.height != config["expected_training_rows"]
        or validation.height != config["expected_validation_rows"]
    ):
        raise ValueError("Frozen partition row counts differ from the approved configuration")
    for column in ["review_id", "duplicate_group_id"]:
        if set(train[column]) & set(validation[column]):
            raise ValueError(f"Cross-partition overlap in {column}")
    return (
        train,
        validation,
        {
            "manifest_sha256": file_hash(manifest_path),
            "source_sha256": manifest["source_provenance"]["sha256"],
            "train_sha256": manifest["files"]["train.parquet"]["sha256"],
            "validation_sha256": manifest["files"]["validation.parquet"]["sha256"],
            "original_split_membership_sha256": manifest["split_membership_sha256"],
            "files_opened": ["manifest.json", "train.parquet", "validation.parquet"],
            "test_accessed": False,
        },
    )


def annotation_rows(validation: pl.DataFrame, per_class: int, seed: int) -> list[dict[str, Any]]:
    """Select within proxy strata, then shuffle display order; this does not label text."""
    selected = []
    for label in LABELS:
        candidates = validation.filter(pl.col("sentiment_proxy") == label).to_dicts()
        if len(candidates) < per_class:
            raise ValueError(f"Insufficient validation support for annotation stratum {label}")
        candidates.sort(
            key=lambda row: hashlib.sha256(f"{seed}:reserve:{row['review_id']}".encode()).digest()
        )
        selected.extend(candidates[:per_class])
    selected.sort(
        key=lambda row: hashlib.sha256(f"{seed}:display:{row['review_id']}".encode()).digest()
    )
    return selected


def prepare_annotations(
    validation: pl.DataFrame, config: dict[str, Any], dataset: dict[str, Any]
) -> dict[str, Any]:
    directory = PROJECT_ROOT / config["annotation_directory"]
    template = directory / "annotation_template.csv"
    provenance_path = directory / "internal/provenance.json"
    selected = annotation_rows(validation, config["annotation_per_class"], config["seed"])
    reserved = [
        {"review_id": row["review_id"], "sentiment_proxy": row["sentiment_proxy"]}
        for row in selected
    ]
    if provenance_path.exists():
        existing: dict[str, Any] = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (
            existing["reserved"] != reserved
            or existing["validation_sha256"] != dataset["validation_sha256"]
        ):
            raise ValueError("Annotation reservation changed; refusing to replace the holdout")
        if file_hash(template) != existing["blank_template_sha256"]:
            raise ValueError("Blank annotation master changed; use separate annotator copies")
        return existing
    if template.exists():
        raise ValueError("Annotation template exists without provenance; refusing overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    with template.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=ANNOTATION_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "review_id": row["review_id"],
                    "text": row["text"],
                    "human_sentiment": "",
                    "ambiguity_flag": "",
                    "annotator_notes": "",
                }
            )
    provenance = {
        "created_at": datetime.now(UTC).isoformat(),
        "seed": config["seed"],
        "sampling": f"{config['annotation_per_class']} per rating-proxy stratum; "
        "seeded SHA256 ordering; shuffled display",
        "per_proxy_class": config["annotation_per_class"],
        "reserved": reserved,
        "validation_sha256": dataset["validation_sha256"],
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "reserved_ids_sha256": id_hash([row["review_id"] for row in selected]),
        "blank_template_sha256": file_hash(template),
        "human_labels_created": False,
    }
    write_json(provenance_path, provenance)
    return provenance


def make_pipeline(name: str, config: dict[str, Any]) -> Any:
    # Keeping TF-IDF inside the pipeline fits vocabulary/IDF on training text only.
    if name == "majority":
        return Pipeline([("classifier", DummyClassifier(strategy="most_frequent"))])
    settings = {**config["tfidf"], "ngram_range": tuple(config["tfidf"]["ngram_range"])}
    vectorizer = TfidfVectorizer(**settings)
    if name == "naive_bayes":
        classifier = MultinomialNB(alpha=config["naive_bayes_alpha"])
    elif name in {"logistic_regression", "logistic_regression_balanced"}:
        classifier = LogisticRegression(
            **config["logistic_regression"],
            random_state=config["seed"],
            class_weight="balanced" if name.endswith("balanced") else None,
        )
    else:
        raise ValueError(f"Unknown baseline: {name}")
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def metrics(truth: list[str], predictions: list[str]) -> dict[str, Any]:
    # Selection uses Macro-F1 so the large positive class cannot dominate the average.
    # Proxy-label agreement measures reproduction of ratings, not verified sentiment accuracy.
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=LABELS, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(
            f1_score(truth, predictions, labels=LABELS, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(truth, predictions, labels=LABELS, average="weighted", zero_division=0)
        ),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(truth, predictions, labels=LABELS).tolist(),
        "confusion_label_order": LABELS,
        "predicted_counts": {label: predictions.count(label) for label in LABELS},
        "zero_division_policy": "undefined precision/recall/F1 reported as zero",
    }


def fit_and_score(
    pipeline: Any, train: pl.DataFrame, scoring: pl.DataFrame
) -> tuple[list[str], dict[str, Any]]:
    """Only this boundary fits estimators: metadata frames never reach sklearn."""
    inputs: list[str] = train["model_text"].to_list()
    labels: list[str] = train["sentiment_proxy"].to_list()
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(inputs, labels)
    fit_seconds = perf_counter() - started
    predict_texts(pipeline, inputs[:16])  # Warm numerical paths with training text only.
    started = perf_counter()
    predictions = predict_texts(pipeline, scoring["model_text"].to_list())
    inference_seconds = perf_counter() - started
    return predictions, {
        "training_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "inference_rows": scoring.height,
        "warmup_rows": 16,
        "amortized_ms_per_review": 1000 * inference_seconds / scoring.height,
        "warnings": [
            {"category": type(w.message).__name__, "message": str(w.message)} for w in caught
        ],
        "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
    }


def verify_idf(pipeline: Any, train_text: list[str]) -> bool:
    if "tfidf" not in pipeline.named_steps:
        return True
    vectorizer = pipeline.named_steps["tfidf"]
    counter = CountVectorizer(
        vocabulary=vectorizer.vocabulary_, analyzer=vectorizer.build_analyzer(), token_pattern=None
    )
    counts = counter.transform(train_text)
    frequencies = np.asarray((counts > 0).sum(axis=0)).ravel()
    expected = np.log((1 + len(train_text)) / (1 + frequencies)) + 1
    if not np.allclose(vectorizer.idf_, expected, rtol=1e-12, atol=1e-12):
        raise ValueError("TF-IDF statistics do not match training-only document frequencies")
    return True


def canonical_result(
    name: str, score: dict[str, Any], run: dict[str, Any], config: dict[str, Any], size: int
) -> dict[str, Any]:
    values = [
        EvaluationMetric(
            name=key,
            value=score[key],
            averaging="macro"
            if key == "macro_f1"
            else "weighted"
            if key == "weighted_f1"
            else "none",
            unit="fraction",
        )
        for key in ["accuracy", "macro_f1", "weighted_f1"]
    ]
    for label in LABELS:
        values.extend(
            EvaluationMetric(
                name=key,
                value=score["per_class"][label][key],
                averaging="per_class",
                label=label,
                unit="fraction",
            )
            for key in ["precision", "recall", "f1"]
        )
    result = ModelEvaluationResult(
        evaluation_id=f"{config['version']}:{name}:validation",
        task="sentiment",
        model_version=f"{config['version']}:{name}",
        dataset_revision=run["dataset"]["source_sha256"],
        split_manifest_hash=run["membership_sha256"],
        annotation_version="rating_proxy_v1",
        code_revision=run["code_sha256"]["ml/signaldesk_ml/baselines.py"],
        environment_lock_hash=run["uv_lock_sha256"],
        evaluated_at=datetime.now(UTC),
        sample_count=size,
        split="validation",
        metrics=values,
        confusion_matrix_artifact="run.json",
        limitations=[
            "Agreement with rating-derived proxies, not verified human sentiment",
            "Annotation holdout excluded; frozen test unopened",
            "Chronological imbalance/shift and duplicate detector limitations",
        ],
    )
    return result.model_dump(mode="json")


def feature_weights(pipeline: Any) -> dict[str, Any]:
    if "tfidf" not in pipeline.named_steps or not hasattr(
        pipeline.named_steps["classifier"], "coef_"
    ):
        return {}
    classifier = pipeline.named_steps["classifier"]
    names = pipeline.named_steps["tfidf"].get_feature_names_out()
    return {
        str(label): [
            {"feature": str(names[index]), "coefficient": float(row[index])}
            for index in np.argsort(row)[-10:][::-1]
        ]
        for label, row in zip(classifier.classes_, classifier.coef_, strict=True)
    }


def verify_run(config: dict[str, Any]) -> dict[str, Any]:
    train, validation, dataset = load_data(config)
    directory = PROJECT_ROOT / config["artifact_directory"]
    run: dict[str, Any] = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if run["configuration"] != config or run["dataset"] != dataset:
        raise ValueError("Run configuration or frozen train/validation reference changed")
    annotation = prepare_annotations(validation, config, dataset)
    reserved = {record["review_id"] for record in annotation["reserved"]}
    scoring = validation.filter(~pl.col("review_id").is_in(sorted(reserved)))
    membership = json.loads((directory / "membership.json").read_text())
    if (
        membership["training_ids"] != train["review_id"].to_list()
        or membership["scoring_ids"] != scoring["review_id"].to_list()
        or set(membership["annotation_ids"]) != reserved
    ):
        raise ValueError("Training/scoring/annotation membership differs from the recorded run")
    if (
        set(membership["training_ids"]) & (set(membership["scoring_ids"]) | reserved)
        or set(membership["scoring_ids"]) & reserved
    ):
        raise ValueError("Annotation holdout or validation training contamination")
    if file_hash(directory / "membership.json") != run["membership_sha256"]:
        raise ValueError("Run membership checksum changed")
    if datetime.fromisoformat(annotation["created_at"]) >= datetime.fromisoformat(
        run["started_at"]
    ):
        raise ValueError("Annotation sample was not reserved before experimentation")
    for name, checksum in run["files"].items():
        if file_hash(directory / name) != checksum:
            raise ValueError(f"Run artifact checksum mismatch: {name}")
    predictions_frame = pl.read_parquet(directory / "validation_predictions.parquet")
    with threadpool_limits(limits=config["cpu_threads"]):
        for name, record in run["models"].items():
            pipeline = joblib.load(directory / record["artifact"])
            predictions = predict_texts(pipeline, scoring["model_text"].to_list())
            if predictions != predictions_frame[name].to_list():
                raise ValueError("Saved-model validation predictions do not reproduce")
            if metrics(scoring["sentiment_proxy"].to_list(), predictions) != record["metrics"]:
                raise ValueError("Validation metrics do not reproduce")
            if set(pipeline.classes_) != set(LABELS):
                raise ValueError("Saved class mapping differs from canonical sentiment labels")
            verify_idf(pipeline, train["model_text"].to_list())
            ModelEvaluationResult.model_validate(record["evaluation"])
        selected = load_selected(directory, trusted=True)
        predict_texts(selected, scoring["model_text"].head(3).to_list())
    if annotation["reserved_ids_sha256"] != id_hash(sorted(reserved)):
        raise ValueError("Reserved annotation membership changed")
    return {
        "status": "passed",
        "models": len(run["models"]),
        "scored_validation_rows": scoring.height,
        "annotation_rows": len(reserved),
        "test_accessed": False,
        "retraining_performed": False,
    }


def train_baselines(config: dict[str, Any]) -> dict[str, Any]:
    directory = PROJECT_ROOT / config["artifact_directory"]
    if (directory / "run.json").exists():
        return verify_run(config)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(
            "Run directory is nonempty; refusing to overwrite partial or existing experiments"
        )
    train, validation, dataset = load_data(config)
    annotation = prepare_annotations(validation, config, dataset)
    reserved = {row["review_id"] for row in annotation["reserved"]}
    scoring = validation.filter(~pl.col("review_id").is_in(sorted(reserved)))
    if len(reserved) != 3 * config["annotation_per_class"]:
        raise ValueError("Annotation reservation is incomplete")
    directory.mkdir(parents=True, exist_ok=True)
    membership = {
        "training_ids": train["review_id"].to_list(),
        "scoring_ids": scoring["review_id"].to_list(),
        "annotation_ids": sorted(reserved),
    }
    write_json(directory / "membership.json", membership)
    run: dict[str, Any] = {
        "version": config["version"],
        "configuration": config,
        "dataset": dataset,
        "started_at": datetime.now(UTC).isoformat(),
        "model_input_columns": ["model_text"],
        "models": {},
        "membership_sha256": file_hash(directory / "membership.json"),
        "annotation_provenance_sha256": file_hash(
            PROJECT_ROOT / config["annotation_directory"] / "internal/provenance.json"
        ),
        "uv_lock_sha256": file_hash(PROJECT_ROOT / "uv.lock"),
        "code_sha256": {
            p.relative_to(PROJECT_ROOT).as_posix(): file_hash(p)
            for p in sorted((PROJECT_ROOT / "ml/signaldesk_ml").glob("*.py"))
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in [
                    "scikit-learn",
                    "numpy",
                    "scipy",
                    "joblib",
                    "threadpoolctl",
                    "torch",
                    "transformers",
                    "accelerate",
                ]
            },
        },
        "training_rows": train.height,
        "validation_scoring_rows": scoring.height,
        "annotation_rows": len(reserved),
        "test_accessed": False,
        "proxy_counts": {
            "training": dict(train.group_by("sentiment_proxy").len().iter_rows()),
            "scoring": dict(scoring.group_by("sentiment_proxy").len().iter_rows()),
        },
    }
    outputs = scoring.select("review_id", "sentiment_proxy")
    weights: dict[str, Any] = {}
    with threadpool_limits(limits=config["cpu_threads"]):
        for name in MODEL_NAMES:
            print(
                f"Fitting {name} on {train.height:,} training texts; "
                f"scoring {scoring.height:,} validation texts...",
                flush=True,
            )
            pipeline = make_pipeline(name, config)
            predictions, costs = fit_and_score(pipeline, train, scoring)
            path = directory / f"{name}.joblib"
            joblib.dump(pipeline, path, compress=3)
            score = metrics(scoring["sentiment_proxy"].to_list(), predictions)
            vectorizer = pipeline.named_steps.get("tfidf")
            record = {
                "metrics": score,
                **costs,
                "artifact": path.name,
                "artifact_bytes": path.stat().st_size,
                "artifact_sha256": file_hash(path),
                "class_mapping": [str(label) for label in pipeline.classes_],
                "feature_count": len(vectorizer.vocabulary_) if vectorizer is not None else 0,
                "tfidf_training_only_verified": verify_idf(pipeline, train["model_text"].to_list()),
                "evaluation": canonical_result(name, score, run, config, scoring.height),
            }
            run["models"][name] = record
            outputs = outputs.with_columns(pl.Series(name, predictions))
            if name.startswith("logistic_regression"):
                weights[name] = feature_weights(pipeline)
            print(
                json.dumps(
                    {
                        "model": name,
                        "macro_f1": score["macro_f1"],
                        "training_seconds": costs["training_seconds"],
                        "converged": costs["converged"],
                    }
                ),
                flush=True,
            )
    run["selected_model"] = max(
        MODEL_NAMES,
        key=lambda name: (
            run["models"][name]["metrics"]["macro_f1"],
            -run["models"][name]["inference_seconds"],
        ),
    )
    run["selection_rule"] = (
        "Highest validation Macro-F1, exact ties broken by lower measured batch inference time"
    )
    outputs.write_parquet(directory / "validation_predictions.parquet", compression="zstd")
    write_json(directory / "feature_weights.json", weights)
    selected = run["selected_model"]
    errors = scoring.with_columns(outputs[selected].alias("prediction")).filter(
        pl.col("prediction") != pl.col("sentiment_proxy")
    )
    examples: list[dict[str, Any]] = []
    for truth in LABELS:
        for predicted in LABELS:
            if truth == predicted:
                continue
            candidates = errors.filter(
                (pl.col("sentiment_proxy") == truth) & (pl.col("prediction") == predicted)
            ).to_dicts()
            candidates.sort(
                key=lambda row: hashlib.sha256(
                    f"{config['seed']}:error:{row['review_id']}".encode()
                ).digest()
            )
            examples.extend(
                {key: row[key] for key in ["review_id", "text", "sentiment_proxy", "prediction"]}
                for row in candidates[:3]
            )
    write_json(directory / "error_examples.json", examples)
    run["files"] = {path.name: file_hash(path) for path in directory.iterdir() if path.is_file()}
    run["finished_at"] = datetime.now(UTC).isoformat()
    write_json(directory / "run.json", run)
    result = verify_run(config)
    from signaldesk_ml.reporting import write_report

    write_report(run, directory)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare-annotations", "train", "verify"])
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config/sentiment-baselines.json"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        if args.command == "prepare-annotations":
            _, validation, dataset = load_data(config)
            reserved = prepare_annotations(validation, config, dataset)
            result = {"reserved": len(reserved["reserved"]), "human_labels_created": False}
        elif args.command == "train":
            result = train_baselines(config)
        else:
            result = verify_run(config)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Baseline workflow stopped: {exc}\n")
    print(json.dumps(result, indent=2))
