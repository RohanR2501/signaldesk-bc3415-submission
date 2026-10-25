"""Consolidate three immutable annotation sources with verified frozen predictions."""

import csv
import json
import math
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from signaldesk.config import PROJECT_ROOT
from signaldesk_data.source import file_hash, write_json
from signaldesk_ml.ai_reference import reference_names
from signaldesk_ml.complaint_preparation import FIELDS
from signaldesk_ml.independent_evaluation import (
    LABELS,
    SENTIMENT,
    STATES,
    complaint_metrics,
    deny_test_files,
    identity,
    load_protocol,
    read_csv,
)

ANNOTATIONS = Path("data/annotations/evaluation-v1")
SOURCES = {
    "llm_a": ANNOTATIONS / "annotation_ai_150.csv",
    "llm_b": ANNOTATIONS / "annotation_independent_llm_150.csv",
    "human20": ANNOTATIONS / "signaldesk_human_spotcheck_20_ready.csv",
}
CACHE = Path("artifacts/evaluation/ai-reference-v1/primary")
OUTPUT = Path("artifacts/evaluation/final-consolidation-v1")
FIELDS_TO_COMPARE = ("human_sentiment", *LABELS)
UNKNOWN = {"", "uncertain"}
SOURCE_KINDS = {
    "llm_a": "Codex-generated reference; not human-verified",
    "llm_b": "User-supplied independent LLM reference; model/version not provided",
    "human20": "User-declared independent human judgments; deliberately selected challenge cases",
}


def read_reference(path: Path, expected: dict[str, str], size: int) -> list[dict[str, str]]:
    """Labels are validated separately from evidence; source bytes are never rewritten."""
    rows = read_csv(path, FIELDS)
    actual = identity(rows)
    if len(rows) != size or not set(actual) <= set(expected):
        raise ValueError("Unexpected source size or ID membership")
    if any(expected[identifier] != checksum for identifier, checksum in actual.items()):
        raise ValueError("Annotation review text differs from frozen original")
    for row in rows:
        for field in FIELDS_TO_COMPARE:
            row[field] = row[field].strip().lower()
            allowed = (
                {*SENTIMENT, "uncertain", ""}
                if field == "human_sentiment"
                else {"yes", "no", "uncertain", ""}
            )
            if row[field] not in allowed:
                raise ValueError("Unknown annotation value; no judgment is inferred")
        if row["ambiguity_flag"].strip().lower() not in {"true", "false", ""}:
            raise ValueError("Unknown ambiguity value")
    return rows


def compare_annotations(
    first: list[dict[str, str]], second: list[dict[str, str]], first_name: str, second_name: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {row["review_id"]: row for row in first}
    if not {r["review_id"] for r in second} <= set(by_id):
        raise ValueError("Comparison requires exact ID alignment; no row-order join")
    ledger: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"reviews": len(second), "fields": {}}
    complete_match = 0
    for row in second:
        other = by_id[row["review_id"]]
        all_resolved_equal = True
        for field in FIELDS_TO_COMPARE:
            left, right = other[field], row[field]
            unresolved = left in UNKNOWN or right in UNKNOWN
            status = "unresolved" if unresolved else "agree" if left == right else "disagree"
            ledger.append(
                {
                    "review_id": row["review_id"],
                    "field": field,
                    "first_source": first_name,
                    "second_source": second_name,
                    "first_value": left,
                    "second_value": right,
                    "status": status,
                    "literal_equal": left == right,
                    "unresolved_reason": (
                        "both"
                        if left in UNKNOWN and right in UNKNOWN
                        else first_name
                        if left in UNKNOWN
                        else second_name
                    )
                    if unresolved
                    else "",
                }
            )
            all_resolved_equal &= status == "agree"
        complete_match += int(all_resolved_equal)
    for field in FIELDS_TO_COMPARE:
        cells = [cell for cell in ledger if cell["field"] == field]
        counts = Counter(cell["status"] for cell in cells)
        resolved = counts["agree"] + counts["disagree"]
        summary["fields"][field] = {
            "resolved_agree": counts["agree"],
            "resolved_disagree": counts["disagree"],
            "unresolved_any": counts["unresolved"],
            "literal_matches_including_uncertainty": sum(c["literal_equal"] for c in cells),
            "both_uncertain": sum(
                c["first_value"] == c["second_value"] == "uncertain" for c in cells
            ),
            "resolved_agreement_fraction": counts["agree"] / resolved if resolved else None,
            "agreement_ids": [c["review_id"] for c in cells if c["status"] == "agree"],
            "disagreements": [c for c in cells if c["status"] == "disagree"],
            "unresolved": [c for c in cells if c["status"] == "unresolved"],
        }
    summary["all_six_fields_resolved_and_equal_reviews"] = complete_match
    summary["ambiguity_flag_literal_agreement"] = sum(
        by_id[r["review_id"]]["ambiguity_flag"].strip().lower()
        == r["ambiguity_flag"].strip().lower()
        for r in second
    )
    return summary, ledger


def audit_evidence(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Flag source formatting and literal quotes only; never adjudicate classifications."""
    issues = []
    entries = []
    category_pattern = "(?:" + "|".join(label.replace("_", "[_ ]") for label in LABELS) + ")"
    delimiter = re.compile(r"\s*(?:\||;)\s*(?=" + category_pattern + r"\s*:)", re.I)
    for row in rows:
        row_issues = []
        covered = set()
        evidence = row["evidence"].strip()
        if re.search(r";\s*" + category_pattern + r"\s*:", evidence, re.I):
            row_issues.append("noncanonical_semicolon_separator")
        for entry in delimiter.split(evidence) if evidence else []:
            prefix, separator, quote = entry.partition(":")
            category = re.sub(r"\s+", "_", prefix.strip().lower()) if separator else None
            if category not in LABELS:
                category, quote = None, entry.strip()
                row_issues.append("missing_or_unknown_category_prefix")
            else:
                if prefix.strip() != category:
                    row_issues.append("noncanonical_category_prefix")
                quote = quote.strip()
                if row[category] == "no":
                    row_issues.append("evidence_category_has_negative_judgment")
            exact = bool(quote) and quote in row["text"]
            if not exact:
                row_issues.append("nonexact_or_concatenated_quote")
            if exact and category is not None:
                covered.add(category)
            entries.append(
                {
                    "review_id": row["review_id"],
                    "raw_entry": entry,
                    "audit_category": category,
                    "excerpt": quote,
                    "exact_substring": exact,
                }
            )
        missing = [label for label in LABELS if row[label] == "yes" and label not in covered]
        if missing:
            row_issues.append("affirmative_without_category_linked_exact_quote")
        if (
            any(row[field] == "uncertain" for field in FIELDS_TO_COMPARE)
            and not row["annotator_notes"].strip()
        ):
            row_issues.append("uncertain_without_explanatory_notes")
        if row["ambiguity_flag"].strip().lower() == "true" and not row["annotator_notes"].strip():
            row_issues.append("ambiguity_without_explanatory_notes")
        if row_issues:
            issues.append(
                {
                    "review_id": row["review_id"],
                    "flags": sorted(set(row_issues)),
                    "yes_categories_without_linked_exact_quote": missing,
                    "raw_evidence": row["evidence"],
                    "judgments_preserved": True,
                }
            )
    return {
        "rows": len(rows),
        "flagged_rows": len(issues),
        "issues": issues,
        "parsed_excerpt_entries": len(entries),
        "exact_substring_entries": sum(e["exact_substring"] for e in entries),
        "nonexact_substring_entries": sum(not e["exact_substring"] for e in entries),
        "entries": entries,
        "policy": (
            "Alias recognition and delimiter parsing are audit-only; no source or judgment edits"
        ),
    }


def checked_predictions(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    cache = root / CACHE
    verified = json.loads(
        (root / "artifacts/annotations/ai-primary-v1/verification.json").read_text()
    )
    checksums = {
        name.replace("\\", "/"): value for name, value in verified["output_sha256"].items()
    }
    for name in ["results.json", "predictions.json"]:
        relative = (CACHE / name).as_posix()
        if file_hash(root / relative) != checksums[relative]:
            raise ValueError("Cached prediction/result checksum changed")
    result = json.loads((cache / "results.json").read_text())
    protocol_path = (
        root / protocol["configuration"]["annotation_directory"] / "internal/protocol.json"
    )
    if result["frozen_protocol_sha256"] != file_hash(protocol_path):
        raise ValueError("Cached predictions do not match frozen model/metric protocol")
    prediction: dict[str, Any] = json.loads((cache / "predictions.json").read_text())
    ids = prediction["review_ids"]
    if len(ids) != 150 or len(set(ids)) != 150 or set(ids) != set(protocol["primary_identity"]):
        raise ValueError("Cached prediction IDs differ from the frozen 150 text bodies")
    outputs = prediction["outputs"]
    for name in protocol["configuration"]["sentiment_models"]:
        if len(outputs[name]) != 150 or not set(outputs[name]) <= set(SENTIMENT):
            raise ValueError("Incomplete or invalid cached sentiment labels")
    for mode in ("rules", "nli", "linear"):
        cells = outputs[mode]
        if (
            len(cells) != 750
            or {(c["review_id"], c["category"]) for c in cells}
            != {(identifier, category) for identifier in ids for category in LABELS}
            or not {c["state"] for c in cells} <= set(STATES)
        ):
            raise ValueError("Incomplete/duplicate cached complaint states")
    return prediction


def sentiment_metrics(truth: list[str], predicted: list[str]) -> dict[str, Any]:
    """Same frozen three-class/zero-division policy, without importing training/SciPy."""
    if not truth or len(truth) != len(predicted) or not set(truth + predicted) <= set(SENTIMENT):
        raise ValueError("Resolved canonical sentiment labels must align")
    matrix = [
        [
            sum(t == first and p == second for t, p in zip(truth, predicted, strict=True))
            for second in SENTIMENT
        ]
        for first in SENTIMENT
    ]
    per_class = {}
    for j, label in enumerate(SENTIMENT):
        tp = matrix[j][j]
        support = sum(matrix[j])
        selected = sum(matrix[i][j] for i in range(3))
        per_class[label] = {
            "precision": tp / selected if selected else 0.0,
            "recall": tp / support if support else 0.0,
            "f1": 2 * tp / (support + selected) if support + selected else 0.0,
            "support": support,
        }
    return {
        "accuracy": sum(matrix[i][i] for i in range(3)) / len(truth),
        "macro_f1": sum(v["f1"] for v in per_class.values()) / 3,
        "weighted_f1": sum(v["f1"] * v["support"] for v in per_class.values()) / len(truth),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "confusion_label_order": list(SENTIMENT),
        "predicted_counts": dict(Counter(predicted)),
        "zero_division_policy": "undefined precision/recall/F1 reported as zero",
    }


def score_reference(
    rows: list[dict[str, str]], predictions: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    ids = predictions["review_ids"]
    outputs = predictions["outputs"]
    sentiment = {}
    known = [r for r in rows if r["human_sentiment"] not in UNKNOWN]
    for name in config["sentiment_models"]:
        by_id = dict(zip(ids, outputs[name], strict=True))
        sentiment[name] = {
            "resolved_reference_rows": len(known),
            "uncertain_reference_excluded": sum(r["human_sentiment"] == "uncertain" for r in rows),
            "missing_reference_excluded": sum(not r["human_sentiment"] for r in rows),
            "agreement_metrics": reference_names(
                sentiment_metrics(
                    [r["human_sentiment"] for r in known], [by_id[r["review_id"]] for r in known]
                )
            )
            if known
            else None,
        }
    complaints = {}
    for mode in ("rules", "nli", "linear"):
        by_key = {(r["review_id"], r["category"]): r["state"] for r in outputs[mode]}
        categories = {}
        for category in LABELS:
            states = [by_key[(r["review_id"], category)] for r in rows]
            # Empty reference cells enter only the excluded mask; never become no/negative.
            truth = [r[category] or "uncertain" for r in rows]
            scored = reference_names(complaint_metrics(truth, states, config))
            unresolved_count = scored.pop("llm_reference_uncertain_excluded")
            categories[category] = {
                **scored,
                "unresolved_reference_excluded": unresolved_count,
                "missing_reference_excluded": sum(not r[category] for r in rows),
                "uncertain_reference_excluded": sum(r[category] == "uncertain" for r in rows),
            }
        complaints[mode] = categories
    return {
        "reference_kind": "LLM_generated_not_human_verified",
        "sentiment": sentiment,
        "complaints": complaints,
        "interpretation": "Reference agreement, not population accuracy",
    }


def consolidate(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    output = root / OUTPUT
    if output.exists():
        raise ValueError("Final consolidation already exists; refusing overwrite")
    protocol = load_protocol(root, check_models=True)
    before = {name: file_hash(root / path) for name, path in SOURCES.items()}
    sources = {
        name: read_reference(
            root / path, protocol["primary_identity"], 20 if name == "human20" else 150
        )
        for name, path in SOURCES.items()
    }
    predictions = checked_predictions(root, protocol)
    # Persist the scope/hash record before statistics; models and thresholds were frozen earlier.
    output.mkdir(parents=True)
    write_json(
        output / "manifest.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "inputs": {
                name: {
                    "path": path.as_posix(),
                    "sha256": before[name],
                    "provenance": SOURCE_KINDS[name],
                }
                for name, path in SOURCES.items()
            },
            "code_sha256": file_hash(Path(__file__)),
            "original_protocol_sha256": file_hash(
                root / protocol["configuration"]["annotation_directory"] / "internal/protocol.json"
            ),
            "cache_predictions_sha256": file_hash(root / CACHE / "predictions.json"),
            "cache_results_sha256": file_hash(root / CACHE / "results.json"),
            "frozen_model_files_verified": len(protocol["model_files"]),
            "new_inference_training_or_threshold_changes": False,
            "missing_and_uncertain_policy": "unresolved and excluded per field; never negative",
            "human_evidence_policy": (
                "flag only; all valid original judgments retained in comparisons"
            ),
            "human_sample": "20 deliberately selected disagreement cases, not population accuracy",
        },
    )
    comparisons, ledger = {}, []
    for first, second in [("llm_a", "llm_b"), ("llm_a", "human20"), ("llm_b", "human20")]:
        summary, cells = compare_annotations(sources[first], sources[second], first, second)
        comparisons[f"{first}_vs_{second}"] = summary
        ledger.extend(cells)
    scored = {
        name: score_reference(sources[name], predictions, protocol["configuration"])
        for name in ("llm_a", "llm_b")
    }
    cached = json.loads((root / CACHE / "results.json").read_text())
    parity = before["llm_a"] == cached["reference_sha256"]
    if parity:
        for name, info in scored["llm_a"]["sentiment"].items():
            actual, expected = (
                info["agreement_metrics"],
                cached["sentiment"][name]["agreement_metrics"],
            )
            if actual["confusion_matrix"] != expected["confusion_matrix"] or any(
                not math.isclose(actual[key], expected[key], abs_tol=1e-12)
                for key in ("label_agreement", "macro_f1", "weighted_f1")
            ):
                raise ValueError("Count-based sentiment metrics differ from frozen prior results")
    result: dict[str, Any] = {
        "source_counts": {
            name: {
                "rows": len(rows),
                "labels": {
                    field: dict(Counter(r[field] for r in rows)) for field in FIELDS_TO_COMPARE
                },
            }
            for name, rows in sources.items()
        },
        "annotation_comparisons": comparisons,
        "model_agreement_by_llm_reference": scored,
        "human_evidence_audit": audit_evidence(sources["human20"]),
        "human_judgments_modified": False,
        "prior_llm_a_sentiment_metric_parity_verified": parity,
        "no_population_prevalence_or_accuracy_claim": True,
    }
    if before != {name: file_hash(root / path) for name, path in SOURCES.items()}:
        raise ValueError("Input changed during consolidation; results not saved")
    write_json(output / "results.json", result)
    with (output / "annotation_comparison_cells.csv").open(
        "x", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ledger[0]), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(ledger)
    print(
        json.dumps(
            {
                "status": "consolidated",
                "input_rows": {n: len(r) for n, r in sources.items()},
                "human_evidence_flagged_rows": result["human_evidence_audit"]["flagged_rows"],
            }
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    sys.addaudithook(deny_test_files)
    consolidate()
