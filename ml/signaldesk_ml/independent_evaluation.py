"""Blinded CSV setup and later human evaluation; no fitting and no test reader."""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from signaldesk.config import PROJECT_ROOT
from signaldesk_data.source import file_hash, write_json
from signaldesk_ml.complaint_preparation import FIELDS, export_working
from signaldesk_ml.complaint_rules import RULE_VERSION, TAXONOMY_VERSION, ComplaintCategory

CONFIG = Path("config/independent-evaluation.json")
LABELS = tuple(c.value for c in ComplaintCategory)
SENTIMENT = ("negative", "neutral", "positive")
CODE = [
    "ml/signaldesk_ml/independent_evaluation.py",
    "ml/signaldesk_ml/complaint_preparation.py",
    "ml/signaldesk_ml/complaint_rules.py",
    "ml/signaldesk_ml/complaint_inference.py",
    "ml/signaldesk_ml/inference.py",
    "ml/signaldesk_ml/transformer_inference.py",
    "ml/signaldesk_ml/baselines.py",
    "docs/COMPLAINT_TAXONOMY.md",
    "uv.lock",
]
STATES = ("positive", "negative", "no_detected_evidence", "abstain", "unavailable")


def deny_test_files(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        name = os.fsdecode(args[0]).replace("\\", "/").rsplit("/", 1)[-1].lower()
        if name in {
            "test.parquet",
            "sentiment.parquet",
            "split_manifest.parquet",
            "investigation.parquet",
        }:
            raise PermissionError("Frozen test/full partitions and investigation are forbidden")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_csv(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError("CSV headers/order must exactly match the supplied template")
        rows = []
        for number, row in enumerate(reader, 2):
            if set(row) != set(fields) or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row {number}; check delimiters and missing cells")
            rows.append(dict(row))
    if len({r["review_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate review IDs; do not duplicate or substitute records")
    return rows


def select_primary(
    reserved: list[dict[str, str]], seed: int, per_stratum: int = 50
) -> tuple[list[str], list[str]]:
    """Uses only original reservation strata/IDs; no text, rules or model outputs."""

    def rank(identifier: str, purpose: str) -> bytes:
        return hashlib.sha256(f"{seed}:independent-v1:{purpose}:{identifier}".encode()).digest()

    if len({r["review_id"] for r in reserved}) != len(reserved):
        raise ValueError("Duplicate IDs in original reservation provenance")
    if {r["sentiment_proxy"] for r in reserved} != set(SENTIMENT):
        raise ValueError("Original reservation strata changed")
    chosen = []
    for label in SENTIMENT:
        candidates = [r["review_id"] for r in reserved if r["sentiment_proxy"] == label]
        if len(candidates) < per_stratum:
            raise ValueError("Insufficient original reservation stratum support")
        chosen.extend(
            sorted(candidates, key=lambda identifier: rank(identifier, "select"))[:per_stratum]
        )
    chosen.sort(key=lambda identifier: rank(identifier, "display"))
    unused = sorted({r["review_id"] for r in reserved} - set(chosen))
    return chosen, unused


def identity(rows: list[dict[str, str]]) -> dict[str, str]:
    return {r["review_id"]: text_hash(r["text"]) for r in rows}


def model_files(root: Path, config: dict[str, Any]) -> dict[str, str]:
    """Freeze saved model/config bytes from existing manifests, without loading models."""
    files = {}
    for name, relative in config["sentiment_models"].items():
        directory = root / relative
        run = json.loads((directory / "run.json").read_text())
        entries = (
            {
                run["models"]["logistic_regression_balanced"]["artifact"]: run["models"][
                    "logistic_regression_balanced"
                ]["artifact_sha256"]
            }
            if name == "balanced_logistic_regression"
            else run["checkpoint_files"]
        )
        files[f"{relative}/run.json"] = file_hash(directory / "run.json")
        for item, checksum in entries.items():
            path = (directory / item).resolve()
            if not path.is_relative_to(directory.resolve()) or file_hash(path) != checksum:
                raise ValueError("Existing sentiment artifact checksum/path mismatch")
            files[path.relative_to(root).as_posix()] = checksum
    relative = config["complaint_directory"]
    directory = root / relative
    run = json.loads((directory / "run.json").read_text())
    if run["taxonomy_version"] != TAXONOMY_VERSION or run["rule_version"] != RULE_VERSION:
        raise ValueError("Complaint versions do not match the frozen taxonomy/rules")
    files[f"{relative}/run.json"] = file_hash(directory / "run.json")
    for item, checksum in run["files"].items():
        if item.startswith(("nli_checkpoint/", "linear/")):
            path = (directory / item).resolve()
            if not path.is_relative_to(directory.resolve()) or file_hash(path) != checksum:
                raise ValueError("Existing complaint artifact checksum/path mismatch")
            files[path.relative_to(root).as_posix()] = checksum
    return files


def load_protocol(root: Path = PROJECT_ROOT, *, check_models: bool = False) -> dict[str, Any]:
    config = json.loads((root / CONFIG).read_text())
    internal = root / config["annotation_directory"] / "internal"
    path = internal / "protocol.json"
    if file_hash(path) != (internal / "protocol.sha256").read_text().strip():
        raise ValueError(
            "Frozen evaluation protocol changed; do not regenerate after seeing labels"
        )
    protocol: dict[str, Any] = json.loads(path.read_text())
    expected = {CONFIG.as_posix(): protocol["config_sha256"], **protocol["frozen_files"]}
    if check_models:
        expected.update(protocol["model_files"])
    for relative, checksum in expected.items():
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root.resolve()) or file_hash(candidate) != checksum:
            raise ValueError(f"Frozen evaluation dependency changed: {relative}")
    return protocol


def prepare(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    config = json.loads((root / CONFIG).read_text())
    internal = root / config["annotation_directory"] / "internal"
    if (internal / "protocol.json").exists():
        protocol = load_protocol(root, check_models=True)
        read_annotations(root / config["primary_csv"], protocol, "primary", require_complete=False)
        return protocol
    if internal.parent.exists():
        raise ValueError("Partial annotation setup exists; refusing overwrite")
    if config["complaint_categories"] != list(LABELS):
        raise ValueError("Evaluation config must use the exact five-category taxonomy")
    provenance_path = root / "data/annotations/sentiment-v1/internal/provenance.json"
    old = json.loads(provenance_path.read_text())
    if len(old["reserved"]) != 240 or Counter(r["sentiment_proxy"] for r in old["reserved"]) != {
        label: 80 for label in SENTIMENT
    }:
        raise ValueError("Expected the original 80/80/80 reservation")
    selected, unused = select_primary(old["reserved"], config["seed"], config["per_stratum"])
    if len(selected) != 150 or len(unused) != 90:
        raise ValueError("Task 08 requires exactly 150 selected and 90 preserved")
    source = root / "data/annotations/sentiment-v1/annotation_template.csv"
    if file_hash(source) != old["blank_template_sha256"]:
        raise ValueError("Original blank annotation master changed; refusing to replace it")
    original = read_csv(
        source, ["review_id", "text", "human_sentiment", "ambiguity_flag", "annotator_notes"]
    )
    if set(identity(original)) != {r["review_id"] for r in old["reserved"]}:
        raise ValueError("Original annotation identities differ from reservation")
    optional_path = root / config["targeted_csv"]
    optional = read_csv(optional_path, FIELDS)
    complaint_provenance = (
        root / "data/annotations/sentiment-v1/complaints-v1/internal/provenance.json"
    )
    complaint = json.loads(complaint_provenance.read_text())
    if set(identity(optional)) != set(complaint["optional_display_ids"]) or len(optional) != 20:
        raise ValueError("Original optional supplement changed")
    if set(identity(original)) & set(identity(optional)):
        raise ValueError("Primary and targeted reservations overlap")
    # Existing memberships demonstrate the reservations stay excluded. No training data is opened.
    baseline = root / "artifacts/sentiment/baselines-v1"
    baseline_run = json.loads((baseline / "run.json").read_text())
    if file_hash(baseline / "membership.json") != baseline_run["membership_sha256"]:
        raise ValueError("Original sentiment membership changed")
    members = json.loads((baseline / "membership.json").read_text())
    if set(members["annotation_ids"]) != set(identity(original)):
        raise ValueError("Sentiment annotation exclusions changed")
    reserved_ids = set(identity(original)) | set(identity(optional))
    if reserved_ids & set(members["training_ids"]):
        raise ValueError("Reserved IDs overlap sentiment fitting")
    complaint_dir = root / config["complaint_directory"]
    selection_path = complaint_dir / "selection.json"
    selection = json.loads(selection_path.read_text())
    complaint_run = json.loads((complaint_dir / "run.json").read_text())
    if file_hash(selection_path) != complaint_run["source_selection_sha256"]:
        raise ValueError("Original complaint development/exclusion provenance changed")
    reserved_families = set(selection["excluded_families"])
    optional_families = {r["duplicate_group_id"] for r in complaint["optional_reservation"]}
    if (
        not reserved_ids <= set(selection["excluded_ids"])
        or not optional_families <= reserved_families
    ):
        raise ValueError("Complaint reservation exclusions changed")
    if reserved_ids & (set(selection["fit_ids"]) | set(selection["holdout_ids"])):
        raise ValueError("Reserved IDs overlap complaint development")
    models = model_files(root, config)
    by_id = {r["review_id"]: r for r in original}
    primary = [by_id[identifier] for identifier in selected]
    frozen = {name: file_hash(root / name) for name in CODE}
    for path in [
        source,
        provenance_path,
        complaint_provenance,
        baseline / "membership.json",
        selection_path,
    ]:
        frozen[path.relative_to(root).as_posix()] = file_hash(path)
    protocol = {
        "version": config["version"],
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": config,
        "config_sha256": file_hash(root / CONFIG),
        "frozen_files": frozen,
        "model_files": models,
        "primary_identity": identity(primary),
        "targeted_identity": identity(optional),
        "primary_ids": selected,
        "unused_primary_ids": unused,
        "primary_counts": {label: 50 for label in SENTIMENT},
        "selection_keys": [r for r in old["reserved"] if r["review_id"] in selected],
        "all_reserved_ids": sorted(reserved_ids),
        "reserved_families": sorted(reserved_families),
        "primary_remaining_reserved": 90,
        "targeted_rows": 20,
        "annotation_fields": FIELDS,
        "human_labels_created": False,
        "complaint_configuration": json.loads((complaint_dir / "run.json").read_text())[
            "configuration"
        ],
        "primary_max_text_characters": max(len(r["text"]) for r in primary),
        "original_annotation_hashes_at_setup": {
            p.relative_to(root).as_posix(): file_hash(p)
            for p in [source, optional_path, source.parent / "complaints-v1/annotation_working.csv"]
        },
    }
    # Freeze before exposing a human-editable subset. Existing human files are never overwritten.
    internal.mkdir(parents=True)
    write_json(internal / "protocol.json", protocol)
    (internal / "protocol.sha256").write_text(file_hash(internal / "protocol.json") + "\n")
    export_working(root / config["primary_csv"], primary)
    return protocol


def read_annotations(
    path: Path, protocol: dict[str, Any], cohort: str, *, require_complete: bool = True
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if cohort not in {"primary", "targeted"}:
        raise ValueError("Choose primary or targeted explicitly; cohorts cannot be pooled")
    expected = protocol[f"{cohort}_identity"]
    rows = read_csv(path, FIELDS)
    if identity(rows) != expected:
        raise ValueError(
            "Missing/extra IDs, wrong cohort, or modified review text; restore original ID/text"
        )
    config = protocol["configuration"]
    missing_cells, incomplete_rows = 0, 0
    for number, row in enumerate(rows, 2):
        missing = False
        for field in ["human_sentiment", *LABELS, "ambiguity_flag"]:
            row[field] = row[field].strip().lower()
            allowed = config[
                "sentiment_values"
                if field == "human_sentiment"
                else "ambiguity_values"
                if field == "ambiguity_flag"
                else "complaint_values"
            ]
            if row[field] and row[field] not in allowed:
                raise ValueError(f"Invalid annotation value at CSV row {number}, field {field}")
            if not row[field]:
                missing_cells += 1
                missing = True
        if missing:
            incomplete_rows += 1
        uncertain = any(row[field] == "uncertain" for field in ["human_sentiment", *LABELS])
        if uncertain and (row["ambiguity_flag"] != "true" or not row["annotator_notes"].strip()):
            raise ValueError(
                f"Uncertain judgments need ambiguity=true and notes at CSV row {number}"
            )
        if row["ambiguity_flag"] == "true" and not row["annotator_notes"].strip():
            raise ValueError(f"Ambiguity needs explanatory notes at CSV row {number}")
        evidenced = set()
        if row["evidence"].strip():
            for entry in row["evidence"].split(" | "):
                category, separator, excerpt = entry.partition(":")
                category, excerpt = category.strip(), excerpt.strip()
                if (
                    not separator
                    or category not in LABELS
                    or not excerpt
                    or excerpt not in row["text"]
                ):
                    raise ValueError(
                        f"Evidence must name a category and quote exact review text at row {number}"
                    )
                evidenced.add(category)
        if any(row[label] == "yes" and label not in evidenced for label in LABELS):
            raise ValueError(
                f"Every yes complaint needs a category-prefixed exact excerpt at row {number}"
            )
    summary = {
        "cohort": cohort,
        "rows": len(rows),
        "complete": incomplete_rows == 0,
        "incomplete_rows": incomplete_rows,
        "missing_label_or_ambiguity_cells": missing_cells,
        "human_uncertain_by_field": {
            field: sum(r[field] == "uncertain" for r in rows)
            for field in ["human_sentiment", *LABELS]
        },
    }
    if require_complete and incomplete_rows:
        raise ValueError(
            f"Annotations incomplete: {incomplete_rows} rows, "
            f"{missing_cells} required cells blank. "
            "No models loaded and no evaluation performed."
        )
    order = list(expected)
    rows.sort(key=lambda row: order.index(row["review_id"]))
    return rows, summary


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def complaint_metrics(
    truth: list[str], states: list[str], config: dict[str, Any]
) -> dict[str, Any]:
    if (
        len(truth) != len(states)
        or not set(truth) <= {"yes", "no", "uncertain"}
        or not set(states) <= set(STATES)
    ):
        raise ValueError("Complaint truth/state arrays must align and contain only explicit states")
    known = [(t, p) for t, p in zip(truth, states, strict=True) if t != "uncertain"]
    positive = sum(t == "yes" for t, _ in known)
    negative = len(known) - positive
    tp = sum(t == "yes" and p == "positive" for t, p in known)
    fp = sum(t == "no" and p == "positive" for t, p in known)
    fn_decided = sum(t == "yes" and p == "negative" for t, p in known)
    tn = sum(t == "no" and p == "negative" for t, p in known)
    missed = positive - tp
    unavailable = bool(states) and all(p == "unavailable" for p in states)
    status = (
        "unavailable_classifier"
        if unavailable
        else "insufficient_human_positive_support"
        if positive < config["minimum_human_positive_support"]
        else "insufficient_human_negative_support"
        if negative < config["minimum_human_negative_support"]
        else "descriptive_only_no_reliability_guarantee"
    )
    return {
        "human_positive_support": positive,
        "human_negative_support": negative,
        "human_uncertain_excluded": truth.count("uncertain"),
        "state_counts_all_rows": {state: states.count(state) for state in STATES},
        "state_counts_known_human_cells": {
            state: sum(p == state for _, p in known) for state in STATES
        },
        "prediction_coverage_known": ratio(tp + fp + fn_decided + tn, len(known)),
        "selected_coverage_all_rows": ratio(states.count("positive"), len(states)),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative_explicit_negative": fn_decided,
        "true_negative_explicit_negative": tn,
        "undetected_human_positive": missed,
        "positive_support_by_prediction_state": {
            state: sum(t == "yes" and p == state for t, p in known) for state in STATES
        },
        "all_known_detection": None
        if unavailable
        else {
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, positive),
            "f1": ratio(2 * tp, 2 * tp + fp + missed),
            "false_negatives_detection": missed,
            "policy": (
                "Unselected positive human cases are detection misses, "
                "not asserted negative predictions"
            ),
        },
        "decided_only": None
        if unavailable
        else {
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn_decided),
            "f1": ratio(2 * tp, 2 * tp + fp + fn_decided),
            "policy": (
                "Conditional on explicit positive/negative model decisions; read with coverage"
            ),
        },
        "support_status": status,
        "reliable_performance_established": False,
    }


def evaluate(
    path: Path, cohort: str, *, trusted_local_models: bool = False, root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    # Configuration/model hashes are checked BEFORE completed labels are loaded.
    protocol = load_protocol(root, check_models=True)
    annotation_hash = file_hash(path)
    rows, validation = read_annotations(path, protocol, cohort)
    if file_hash(path) != annotation_hash:
        raise ValueError("Annotation file changed during validation; close Excel and retry")
    if not trusted_local_models:
        raise ValueError(
            "Use --trusted-local-models only for these locally produced joblib artifacts"
        )
    config = protocol["configuration"]
    target = root / config["result_directory"] / cohort
    if target.exists():
        raise ValueError("Evaluation output already exists; refusing to overwrite previous results")
    # Imports and model loads occur only after the completeness gate. No fitting functions called.
    import joblib
    import torch
    from threadpoolctl import threadpool_limits

    from signaldesk_ml.baselines import metrics as sentiment_metrics
    from signaldesk_ml.complaint_inference import load_complaint_model
    from signaldesk_ml.inference import predict_texts
    from signaldesk_ml.transformer_inference import load_transformer, require_cuda

    require_cuda()
    torch.set_num_threads(config["cpu_threads"])
    texts = [r["text"] for r in rows]
    sentiment_results, complaint_results = {}, {}
    predictions: dict[str, Any] = {
        "review_ids": [r["review_id"] for r in rows],
        "sentiment": {},
        "complaints": {},
    }
    for name, relative in config["sentiment_models"].items():
        directory = root / relative
        if name == "balanced_logistic_regression":
            run = json.loads((directory / "run.json").read_text())
            model = joblib.load(
                directory / run["models"]["logistic_regression_balanced"]["artifact"]
            )
            with threadpool_limits(limits=config["cpu_threads"]):
                labels = predict_texts(model, texts)
        else:
            model = load_transformer(directory)
            labels = model.predict_texts(texts)
        known = [i for i, r in enumerate(rows) if r["human_sentiment"] != "uncertain"]
        sentiment_results[name] = {
            "known_human_support": len(known),
            "human_uncertain_excluded": len(rows) - len(known),
            "metrics": sentiment_metrics(
                [rows[i]["human_sentiment"] for i in known], [labels[i] for i in known]
            )
            if known
            else None,
        }
        predictions["sentiment"][name] = labels
        del model
        torch.cuda.empty_cache()
    for mode in ("rules", "nli", "linear"):
        model = load_complaint_model(
            root / config["complaint_directory"], mode=mode, trusted=mode == "linear"
        )
        outputs = model.predict_texts(texts)
        per_label = {}
        saved_predictions = []
        for j, category in enumerate(LABELS):
            states = []
            for output in outputs:
                state = output.predictions[j].status
                states.append(
                    "no_detected_evidence"
                    if mode == "rules" and state == "abstain"
                    else {"selected": "positive", "not_selected": "negative"}.get(state, state)
                )
            per_label[category] = complaint_metrics([r[category] for r in rows], states, config)
            for row, saved_state, output in zip(rows, states, outputs, strict=True):
                saved_predictions.append(
                    {
                        "review_id": row["review_id"],
                        "category": category,
                        "state": saved_state,
                        "score": output.predictions[j].score,
                        "source": mode,
                        "human_value": row[category],
                    }
                )
        complaint_results[mode] = per_label
        predictions["complaints"][mode] = saved_predictions
        del model
        torch.cuda.empty_cache()
    if file_hash(path) != annotation_hash:
        raise ValueError("Annotation file changed during evaluation; no results saved")
    result = {
        "version": config["version"],
        "cohort": cohort,
        "validation": validation,
        "annotation_file_sha256": annotation_hash,
        "protocol_sha256": file_hash(
            root / config["annotation_directory"] / "internal/protocol.json"
        ),
        "evaluated_at": datetime.now(UTC).isoformat(),
        "ground_truth": "independent_human_annotations",
        "no_population_prevalence_claim": True,
        "no_tuning_performed": True,
        "sentiment": sentiment_results,
        "complaints": complaint_results,
    }
    target.mkdir(parents=True)
    write_json(target / "results.json", result)
    write_json(target / "predictions.json", predictions)
    return result


def main() -> None:
    sys.addaudithook(deny_test_files)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "validate", "evaluate"])
    parser.add_argument("--cohort", choices=["primary", "targeted"], default="primary")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--trusted-local-models", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            protocol = prepare()
            print(
                json.dumps(
                    {
                        "primary_rows": len(protocol["primary_identity"]),
                        "preserved_unused": 90,
                        "targeted_separate": 20,
                        "human_labels_created": False,
                    }
                )
            )
            return
        protocol = load_protocol()
        path = args.annotations or PROJECT_ROOT / protocol["configuration"][f"{args.cohort}_csv"]
        if args.action == "validate":
            _, summary = read_annotations(path, protocol, args.cohort, require_complete=False)
            print(json.dumps(summary))
            if not summary["complete"]:
                raise SystemExit(2)
        else:
            result = evaluate(path, args.cohort, trusted_local_models=args.trusted_local_models)
            print(
                json.dumps(
                    {
                        "cohort": result["cohort"],
                        "status": "evaluated",
                        "rows": result["validation"]["rows"],
                    }
                )
            )
    except (ValueError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
