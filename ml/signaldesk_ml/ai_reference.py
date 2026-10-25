"""Serialize explicit Codex decisions, then compare frozen models to an LLM reference.

No classifier generates reference labels. The original human study remains unchanged.
The legacy human_sentiment column in the separate AI CSV is NOT human ground truth.
"""

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from signaldesk.config import PROJECT_ROOT
from signaldesk_data.source import file_hash, write_json
from signaldesk_ml.complaint_preparation import FIELDS
from signaldesk_ml.independent_evaluation import (
    LABELS,
    complaint_metrics,
    deny_test_files,
    load_protocol,
    read_annotations,
)

REFERENCE = Path("data/annotations/evaluation-v1/annotation_ai_150.csv")
AUDIT = Path("artifacts/annotations/ai-primary-v1")
RESULTS = Path("artifacts/evaluation/ai-reference-v1/primary")
PREFIX = "LLM-generated; not human-verified. "


def assemble_decisions(
    source: list[dict[str, str]], decisions: list[list[Any]], second: dict[str, Any]
) -> list[dict[str, str]]:
    if len(source) != 150 or [r[0] for r in decisions] != list(range(1, 151)):
        raise ValueError("Require exactly 150 explicit, ordered row decisions, without omissions")
    reviewed = set(second["reviewed_rows"]) | set(second["additional_boundary_rows"])
    if not {r[0] for r in decisions if r[3] or "U" in r[2] or r[1] == "uncertain"} <= reviewed:
        raise ValueError("All ambiguous/uncertain first-pass decisions require second-pass review")
    final = {row[0]: row for row in decisions}
    for override in second["overrides"]:
        if override[0] not in reviewed:
            raise ValueError("Unreviewed override")
        final[override[0]] = override
    result = []
    for number in range(1, 151):
        _, sentiment, codes, ambiguity, evidence, notes = final[number]
        if sentiment not in {"positive", "neutral", "negative", "uncertain"}:
            raise ValueError("Invalid explicit sentiment decision")
        if len(codes) != 5 or set(codes) - set("YNU") or not isinstance(ambiguity, bool):
            raise ValueError("Every category needs its own explicit yes/no/uncertain decision")
        if ("U" in codes or sentiment == "uncertain") and not ambiguity:
            raise ValueError("Uncertainty requires ambiguity flag")
        if not isinstance(notes, str) or not notes.strip() or set(evidence) - set(LABELS):
            raise ValueError("Require notes and canonical evidence categories")
        excerpts = []
        for category, quotes in evidence.items():
            for quote in quotes:
                if not quote or quote not in source[number - 1]["text"] or " | " in quote:
                    raise ValueError(f"Nonexact or unsupported evidence at row {number}")
                excerpts.append(f"{category}: {quote}")
        values = {
            label: {"Y": "yes", "N": "no", "U": "uncertain"}[code]
            for label, code in zip(LABELS, codes, strict=True)
        }
        if any(values[label] == "yes" and not evidence.get(label) for label in LABELS):
            raise ValueError("Each affirmative complaint requires exact evidence")
        result.append(
            {
                "review_id": source[number - 1]["review_id"],
                "text": source[number - 1]["text"],
                "human_sentiment": sentiment,
                **values,
                "ambiguity_flag": str(ambiguity).lower(),
                "evidence": " | ".join(excerpts),
                "annotator_notes": PREFIX + notes,
            }
        )
    if len({r["review_id"] for r in result}) != 150:
        raise ValueError("Duplicate source ID")
    return result


def freeze(decisions_directory: Path, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    if (root / REFERENCE).exists() or (root / AUDIT).exists():
        raise ValueError("AI outputs already exist; refusing to overwrite frozen decisions")
    source = json.loads((decisions_directory / "source.json").read_text(encoding="utf-8"))
    batches = sorted(decisions_directory.glob("batch*.json"))
    decisions = [row for p in batches for row in json.loads(p.read_text(encoding="utf-8"))]
    second = json.loads((decisions_directory / "second_pass.json").read_text(encoding="utf-8"))
    rows = assemble_decisions(source, decisions, second)
    # All reasoning is complete before loading metadata for identity/model-integrity validation.
    protocol = load_protocol(root, check_models=True)
    output = root / REFERENCE
    with output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    _, validation = read_annotations(output, protocol, "primary")
    directory = root / AUDIT
    directory.mkdir(parents=True)
    archive = directory / "decisions"
    archive.mkdir()
    for p in [
        *batches,
        decisions_directory / "source.json",
        decisions_directory / "second_pass.json",
    ]:
        shutil.copy2(p, archive / p.name)
    fields = ["human_sentiment", *LABELS]
    uncertain = [
        {
            "row": i,
            "review_id": row["review_id"],
            "fields": [field for field in fields if row[field] == "uncertain"],
        }
        for i, row in enumerate(rows, 1)
        if any(row[field] == "uncertain" for field in fields)
    ]
    evidence_count = sum(len(row["evidence"].split(" | ")) for row in rows if row["evidence"])
    audit = {
        "reference_kind": "LLM_generated_not_human_verified",
        "created_at": datetime.now(UTC).isoformat(),
        "annotator": (
            "Codex assistant in this conversation; exact serving checkpoint/revision not exposed"
        ),
        "external_llm_api_used": False,
        "distinct_llm_used": False,
        "method": (
            "Direct text-only reasoning in ten batches of 15; explicit per-category decisions"
        ),
        "second_pass": {
            "same_assistant": True,
            "independent_annotator": False,
            "ambiguous_mixed_rows_reviewed": second["reviewed_rows"],
            "additional_boundary_rows_reviewed": second["additional_boundary_rows"],
            "overridden_rows": [r[0] for r in second["overrides"]],
            "inter_annotator_agreement": "not measured; no independent annotator",
        },
        "metadata_predictions_and_rules_used_for_label_generation": False,
        "original_input_note": (
            "Excel added a Column1..Column11 header; "
            "only the separate AI output restores the canonical header"
        ),
        "original_input_sha256": file_hash(root / protocol["configuration"]["primary_csv"]),
        "reference_csv": REFERENCE.as_posix(),
        "reference_sha256": file_hash(output),
        "legacy_column_notice": (
            "human_sentiment holds AI labels in this file; no human verification"
        ),
        "counts": {field: dict(Counter(row[field] for row in rows)) for field in fields},
        "ambiguous_rows": sum(row["ambiguity_flag"] == "true" for row in rows),
        "uncertain_rows": uncertain,
        "uncertain_row_count": len(uncertain),
        "validation": validation,
        "evidence_excerpts_checked": evidence_count,
        "evidence_exact_substring_failures": 0,
        "decisions_sha256": {p.name: file_hash(p) for p in archive.iterdir()},
        "comparison_frozen_before_model_predictions": {
            "original_protocol_sha256": file_hash(
                root / protocol["configuration"]["annotation_directory"] / "internal/protocol.json"
            ),
            "ai_comparison_code_sha256": file_hash(Path(__file__)),
            "threshold_policy": (
                "unchanged saved model settings; inherited metrics and state handling"
            ),
            "output_directory": RESULTS.as_posix(),
            "models": [*protocol["configuration"]["sentiment_models"], "rules", "nli", "linear"],
        },
    }
    write_json(directory / "audit.json", audit)
    (directory / "audit.sha256").write_text(file_hash(directory / "audit.json") + "\n")
    write_json(
        output.with_suffix(".provenance.json"),
        {
            "annotation_source": audit["reference_kind"],
            "human_verified": False,
            "reference_sha256": audit["reference_sha256"],
            "audit": AUDIT.as_posix() + "/audit.json",
            "audit_sha256": file_hash(directory / "audit.json"),
        },
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "reference_kind": audit["reference_kind"],
                "counts": audit["counts"],
                "uncertain_rows": len(uncertain),
                "evidence_excerpts_checked": evidence_count,
            }
        ),
        flush=True,
    )
    return audit


def reference_names(value: Any) -> Any:
    """Retain established metric maths while making reference provenance explicit."""
    if isinstance(value, dict):
        return {
            key.replace("human", "llm_reference").replace(
                "accuracy", "label_agreement"
            ): reference_names(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [reference_names(item) for item in value]
    if isinstance(value, str):
        return value.replace("human_", "llm_reference_").replace("human ", "LLM-reference ")
    return value


def compare(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    directory = root / AUDIT
    audit_path = directory / "audit.json"
    if file_hash(audit_path) != (directory / "audit.sha256").read_text().strip():
        raise ValueError("Frozen AI annotation audit changed")
    audit = json.loads(audit_path.read_text())
    if (
        file_hash(Path(__file__))
        != audit["comparison_frozen_before_model_predictions"]["ai_comparison_code_sha256"]
    ):
        raise ValueError("Frozen AI comparison code changed")
    protocol = load_protocol(root, check_models=True)
    protocol_path = (
        root / protocol["configuration"]["annotation_directory"] / "internal/protocol.json"
    )
    if (
        file_hash(protocol_path)
        != audit["comparison_frozen_before_model_predictions"]["original_protocol_sha256"]
    ):
        raise ValueError("Original frozen model/metric protocol changed")
    path = root / REFERENCE
    if file_hash(path) != audit["reference_sha256"]:
        raise ValueError("AI reference changed after freezing")
    rows, validation = read_annotations(path, protocol, "primary")
    if any(not row["annotator_notes"].startswith(PREFIX) for row in rows):
        raise ValueError("Every row must declare LLM generation")
    output = root / RESULTS
    if output.exists():
        raise ValueError("AI comparison already exists; refusing overwrite or another experiment")
    import joblib
    import torch
    from threadpoolctl import threadpool_limits

    from signaldesk_ml.baselines import metrics
    from signaldesk_ml.complaint_inference import load_complaint_model
    from signaldesk_ml.inference import predict_texts
    from signaldesk_ml.transformer_inference import load_transformer, require_cuda

    require_cuda()
    config = protocol["configuration"]
    torch.set_num_threads(config["cpu_threads"])
    texts = [r["text"] for r in rows]
    known = [i for i, r in enumerate(rows) if r["human_sentiment"] != "uncertain"]
    sentiment: dict[str, Any] = {}
    complaints: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    timing: dict[str, float] = {}
    for name, relative in config["sentiment_models"].items():
        start = time.perf_counter()
        if name == "balanced_logistic_regression":
            run = json.loads((root / relative / "run.json").read_text())
            model = joblib.load(
                root / relative / run["models"]["logistic_regression_balanced"]["artifact"]
            )
            with threadpool_limits(limits=config["cpu_threads"]):
                labels = predict_texts(model, texts)
        else:
            model = load_transformer(root / relative)
            labels = model.predict_texts(texts)
        sentiment[name] = {
            "resolved_reference_rows": len(known),
            "reference_uncertain_excluded": len(rows) - len(known),
            "agreement_metrics": reference_names(
                metrics([rows[i]["human_sentiment"] for i in known], [labels[i] for i in known])
            )
            if known
            else None,
        }
        predictions[name] = labels
        timing[name] = time.perf_counter() - start
        print(json.dumps({"completed_comparison": name}), flush=True)
        del model
        torch.cuda.empty_cache()
    for mode in ("rules", "nli", "linear"):
        start = time.perf_counter()
        model = load_complaint_model(
            root / config["complaint_directory"], mode=mode, trusted=mode == "linear"
        )
        outputs = model.predict_texts(texts)
        category_results = {}
        raw: list[dict[str, Any]] = []
        for j, category in enumerate(LABELS):
            states = [
                "no_detected_evidence"
                if mode == "rules" and p.predictions[j].status == "abstain"
                else {"selected": "positive", "not_selected": "negative"}.get(
                    p.predictions[j].status, p.predictions[j].status
                )
                for p in outputs
            ]
            category_results[category] = reference_names(
                complaint_metrics([r[category] for r in rows], states, config)
            )
            raw.extend(
                {
                    "review_id": row["review_id"],
                    "category": category,
                    "state": state,
                    "uncalibrated_score": p.predictions[j].score,
                    "llm_reference": row[category],
                }
                for row, state, p in zip(rows, states, outputs, strict=True)
            )
        complaints[mode], predictions[mode] = category_results, raw
        timing[mode] = time.perf_counter() - start
        print(json.dumps({"completed_comparison": mode}), flush=True)
        del model
        torch.cuda.empty_cache()
    if file_hash(path) != audit["reference_sha256"]:
        raise ValueError("AI reference changed during comparison; no results saved")
    result = {
        "reference_kind": "LLM_generated_not_human_verified",
        "human_verified": False,
        "metric_interpretation": (
            "Agreement against an LLM-generated reference; "
            "NOT independently verified human accuracy"
        ),
        "validation": reference_names(validation),
        "reference_sha256": audit["reference_sha256"],
        "audit_sha256": file_hash(audit_path),
        "frozen_protocol_sha256": file_hash(protocol_path),
        "sentiment": sentiment,
        "complaints": complaints,
        "load_and_prediction_seconds": timing,
        "no_training_or_threshold_changes": True,
        "no_population_prevalence_claim": True,
    }
    output.mkdir(parents=True)
    write_json(output / "results.json", result)
    write_json(
        output / "predictions.json",
        {
            "reference_kind": result["reference_kind"],
            "review_ids": [r["review_id"] for r in rows],
            "outputs": predictions,
        },
    )
    print(
        json.dumps(
            {"comparison_saved": RESULTS.as_posix(), "reference_kind": result["reference_kind"]}
        ),
        flush=True,
    )
    return result


def main() -> None:
    sys.addaudithook(deny_test_files)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "compare"])
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        if args.decisions is None:
            parser.error("--decisions is required to serialize the explicit reasoning ledger")
        freeze(args.decisions)
    else:
        compare()


if __name__ == "__main__":
    main()
