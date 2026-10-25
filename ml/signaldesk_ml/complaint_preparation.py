"""Bounded complaint discovery and blind annotation copies; never train a model."""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from signaldesk.config import PROJECT_ROOT
from signaldesk_data.source import file_hash, write_json

SEED = 3415
LABELS = [
    "product_defect",
    "arrival_damage",
    "wrong_item",
    "delivery_issue",
    "service_return_issue",
]
FIELDS = [
    "review_id",
    "text",
    "human_sentiment",
    *LABELS,
    "ambiguity_flag",
    "evidence",
    "annotator_notes",
]
DATA = Path("data/processed/appliances-v1")
ANNOTATIONS = Path("data/annotations/sentiment-v1")
OUTPUT = Path("artifacts/complaints/foundation-v1")
DISCOVERY = {
    "product_defect": (
        "\\b(?:broke|broken|defective|malfunction|stopped working|do"
        "esn['’]t work|does not work|not working|failed|leaking)\\b"
    ),
    "arrival_damage": (
        "\\b(?:arriv\\w*|receiv\\w*|box|packag\\w*)\\b[^.!?\\n]{0,100}\\b("
        "?:damage\\w*|dent\\w*|crack\\w*|crush\\w*|broken|bent)\\b|\\b(?:"
        "damage\\w*|dent\\w*|crack\\w*|crush\\w*|broken|bent)\\b[^.!?\\n]"
        "{0,65}\\b(?:arriv\\w*|receiv\\w*|box|packag\\w*)\\b"
    ),
    "wrong_item": (
        "\\b(?:wrong|incorrect)\\s+(?:item|product|part|model|size|co"
        "mponent|one)\\b|\\b(?:sent|shipped|received|got)\\b[^.!?\\n]{0"
        ",55}\\b(?:instead|different|wrong)\\b"
    ),
    "delivery_issue": (
        "\\b(?:delivery|shipment|shipping|delivered|package)\\b[^.!?\\"
        "n]{0,65}\\b(?:late|delayed|missing|lost|never)\\b|\\b(?:never"
        "|not|hasn['’]t)\\b[^.!?\\n]{0,35}\\b(?:arrived|delivered|rece"
        "ived)\\b|\\blate (?:delivery|arrival|shipment)\\b"
    ),
    "service_return_issue": (
        "\\b(?:customer (?:service|support)|refund\\w*|return\\w*|replacement|warranty)\\b"
    ),
}


def deny_sealed_files(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        name = os.fsdecode(args[0]).replace("\\", "/").rsplit("/", 1)[-1].lower()
        if name in {"test.parquet", "sentiment.parquet", "split_manifest.parquet"}:
            raise PermissionError("Sealed sentiment files cannot be opened in Phase 3A")


def ranked(identifier: str, purpose: str) -> bytes:
    return hashlib.sha256(f"{SEED}:{purpose}:{identifier}".encode()).digest()


def read_allowed(name: str, columns: list[str], ids: list[str] | None = None) -> pl.DataFrame:
    if name not in {"train", "validation", "investigation"}:
        raise ValueError("Only train/validation/investigation are permitted; test access forbidden")
    if {"text", "model_text"} & set(columns) and (ids is None or len(ids) > 6673):
        raise ValueError("Review text requires a bounded, explicit ID selection")
    manifest = json.loads((PROJECT_ROOT / DATA / "manifest.json").read_text())
    path = PROJECT_ROOT / DATA / f"{name}.parquet"
    if file_hash(path) != manifest["files"][path.name]["sha256"]:
        raise ValueError(f"Frozen {name} checksum mismatch")
    query = pl.scan_parquet(path)
    if ids is not None:
        query = query.filter(pl.col("review_id").is_in(ids))
    frame = query.select(columns).collect()
    if ids is not None and set(frame["review_id"]) != set(ids):
        raise ValueError("Selected IDs are missing from their authorised source")
    return frame


def membership() -> dict[str, Any]:
    baseline = PROJECT_ROOT / "artifacts/sentiment/baselines-v1"
    run = json.loads((baseline / "run.json").read_text())
    if file_hash(baseline / "membership.json") != run["membership_sha256"]:
        raise ValueError("Original sentiment membership changed")
    result: dict[str, Any] = json.loads((baseline / "membership.json").read_text())
    annotation = json.loads((PROJECT_ROOT / ANNOTATIONS / "internal/provenance.json").read_text())
    if (
        file_hash(PROJECT_ROOT / ANNOTATIONS / "annotation_template.csv")
        != annotation["blank_template_sha256"]
    ):
        raise ValueError("Protected annotation master changed")
    if set(result["annotation_ids"]) != {row["review_id"] for row in annotation["reserved"]}:
        raise ValueError("Original annotation reservation changed")
    return result


def investigation_candidates(
    metadata: pl.DataFrame, protected_ids: set[str], protected_groups: set[str]
) -> pl.DataFrame:
    return metadata.filter(
        (pl.col("related_sentiment_split") == "not_selected")
        & ~pl.col("review_id").is_in(sorted(protected_ids))
        & ~pl.col("duplicate_group_id").is_in(sorted(protected_groups))
    )


def sample_ids(frame: pl.DataFrame, size: int, purpose: str) -> list[str]:
    records = sorted(frame.to_dicts(), key=lambda row: ranked(row["review_id"], purpose))
    seen: set[str] = set()
    selected = []
    for row in records:
        if row["duplicate_group_id"] not in seen:
            selected.append(row["review_id"])
            seen.add(row["duplicate_group_id"])
        if len(selected) == size:
            break
    return selected


def discover(text: str) -> list[str]:
    return [label for label, pattern in DISCOVERY.items() if re.search(pattern, text, re.I)]


def audit() -> dict[str, Any]:
    directory = PROJECT_ROOT / OUTPUT
    if (directory / "audit.json").exists():
        saved: dict[str, Any] = json.loads((directory / "audit.json").read_text())
        if saved["discovery_patterns"] != DISCOVERY:
            raise ValueError("Frozen discovery patterns changed; create a new audit version")
        if file_hash(PROJECT_ROOT / DATA / "manifest.json") != saved["dataset_manifest_sha256"]:
            raise ValueError("Frozen manifest changed after the audit")
        for name, checksum in saved["files"].items():
            if file_hash(directory / name) != checksum:
                raise ValueError("Frozen audit evidence checksum mismatch")
        return saved
    train = read_allowed("train", ["review_id", "duplicate_group_id"])
    validation = read_allowed("validation", ["review_id", "duplicate_group_id"])
    frozen = membership()
    if train["review_id"].to_list() != frozen["training_ids"]:
        raise ValueError("Training membership changed")
    inv = read_allowed(
        "investigation", ["review_id", "duplicate_group_id", "related_sentiment_split"]
    )
    eligible = investigation_candidates(
        inv,
        set(train["review_id"]) | set(validation["review_id"]),
        set(train["duplicate_group_id"]) | set(validation["duplicate_group_id"]),
    )
    counts = {}
    frames = []
    examples: list[dict[str, Any]] = []
    for source, frame, maximum in [("train", train, 4096), ("investigation", eligible, 2048)]:
        ids = sample_ids(frame, maximum, f"complaint-audit:{source}")
        texts = read_allowed(source, ["review_id", "text", "model_text", "duplicate_group_id"], ids)
        if not texts["text"].equals(texts["model_text"]):
            raise ValueError("Review-body text changed")
        rows = sorted(texts.to_dicts(), key=lambda row: ranked(row["review_id"], "inspection"))
        for row in rows:
            row["keyword_candidates"] = discover(row["model_text"])
            row["audit_source"] = source
        counts[source] = {
            "sample_rows": len(rows),
            "unique_ids": texts["review_id"].n_unique(),
            "unique_families": texts["duplicate_group_id"].n_unique(),
            "blank_text": sum(not row["model_text"].strip() for row in rows),
            "keyword_hits": {
                label: sum(label in row["keyword_candidates"] for row in rows) for label in LABELS
            },
            "multiple_keyword_categories": sum(len(row["keyword_candidates"]) > 1 for row in rows),
        }
        for label in [*LABELS, "no_keyword_hit"]:
            selected = [
                row
                for row in rows
                if (
                    not row["keyword_candidates"]
                    if label == "no_keyword_hit"
                    else label in row["keyword_candidates"]
                )
            ][:4]
            examples.extend({**row, "inspection_stratum": label} for row in selected)
        frames.append(pl.DataFrame(rows))
    directory.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).write_parquet(directory / "audit_sample.parquet")
    write_json(directory / "audit_examples.json", examples)
    manifest_path = PROJECT_ROOT / DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "purpose": (
            "Taxonomy feasibility only; keyword counts are not complaint truth or prevalence"
        ),
        "sampling": (
            "Seeded SHA256 ordering, one representative per duplicate family; independent sources"
        ),
        "source_counts": counts,
        "investigation_metadata_rows": inv.height,
        "investigation_related_split_counts": dict(
            inv.group_by("related_sentiment_split").len().iter_rows()
        ),
        "investigation_eligible_rows": eligible.height,
        "investigation_eligible_families": eligible["duplicate_group_id"].n_unique(),
        "investigation_text_policy": (
            "Only not_selected families; no train/validation/test-related text"
        ),
        "inspection_records": len(examples),
        "inspection_unique_ids": len({row["review_id"] for row in examples}),
        "human_ground_truth_created": False,
        "test_accessed": False,
        "discovery_patterns": DISCOVERY,
        "code_sha256": file_hash(Path(__file__)),
        "dataset_manifest_sha256": file_hash(manifest_path),
        "source_sha256": {
            name: manifest["files"][f"{name}.parquet"]["sha256"]
            for name in ["train", "validation", "investigation"]
        },
        "files": {
            name: file_hash(directory / name)
            for name in ["audit_sample.parquet", "audit_examples.json"]
        },
    }
    write_json(directory / "audit.json", result)
    return result


def export_working(path: Path, records: list[dict[str, Any]]) -> None:
    """Never overwrite a human-editable copy, including after human annotation starts."""
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            existing = list(reader)
            if reader.fieldnames != FIELDS or [(r["review_id"], r["text"]) for r in existing] != [
                (r["review_id"], r["text"]) for r in records
            ]:
                raise ValueError("Existing annotation identity/text changed; refusing overwrite")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {field: record[field] if field in {"review_id", "text"} else "" for field in FIELDS}
            )


def prepare(categories: list[str], per_category: int) -> dict[str, Any]:
    if (
        set(categories) - set(LABELS)
        or not 1 <= per_category <= 20
        or len(categories) * per_category > 80
    ):
        raise ValueError(
            "Optional targeted supplement must use known categories and at most 80 reviews"
        )
    evidence = audit()
    if any(
        evidence["source_counts"]["train"]["keyword_hits"][label]
        / evidence["source_counts"]["train"]["sample_rows"]
        >= 0.01
        for label in categories
    ):
        raise ValueError(
            "Supplement permitted only for audit-sparse categories (<1% training keyword hits)"
        )
    frozen = membership()
    master = PROJECT_ROOT / ANNOTATIONS / "annotation_template.csv"
    with master.open(encoding="utf-8-sig", newline="") as stream:
        primary = list(csv.DictReader(stream))
    if len(primary) != 240 or {row["review_id"] for row in primary} != set(
        frozen["annotation_ids"]
    ):
        raise ValueError("Protected 240-review master differs from reservation")
    directory = PROJECT_ROOT / ANNOTATIONS / "complaints-v1"
    provenance_path = directory / "internal/provenance.json"
    if provenance_path.exists():
        existing: dict[str, Any] = json.loads(provenance_path.read_text())
        if (
            file_hash(PROJECT_ROOT / "ml/signaldesk_ml/complaint_rules.py")
            != existing["rules_sha256_at_reservation"]
            or file_hash(PROJECT_ROOT / "docs/COMPLAINT_TAXONOMY.md")
            != existing["taxonomy_sha256_at_reservation"]
        ):
            raise ValueError("Rules/taxonomy changed after annotation reservation")
        if (
            existing["target_categories"] != categories
            or existing["per_category_cap"] != per_category
        ):
            raise ValueError("Annotation reservation already frozen; refusing changes")
        if not (directory / "annotation_working.csv").exists():
            raise ValueError(
                "Existing human working copy is missing; restore it rather than overwrite"
            )
        export_working(directory / "annotation_working.csv", primary)
        if existing["optional_rows"]:
            optional_path = directory / "annotation_targeted_optional.csv"
            if not optional_path.exists():
                raise ValueError("Existing optional working copy is missing; restore it")
            optional = read_allowed(
                "validation", ["review_id", "text"], existing["optional_display_ids"]
            )
            lookup = {row["review_id"]: row for row in optional.to_dicts()}
            export_working(
                optional_path,
                [lookup[identifier] for identifier in existing["optional_display_ids"]],
            )
        return existing
    # Freeze reference rules/taxonomy before looking at permitted validation text for selection.
    rules_hash = file_hash(PROJECT_ROOT / "ml/signaldesk_ml/complaint_rules.py")
    taxonomy_hash = file_hash(PROJECT_ROOT / "docs/COMPLAINT_TAXONOMY.md")
    targeted: list[dict[str, Any]] = []
    sampling_keys: list[dict[str, Any]] = []
    if categories:
        validation = read_allowed(
            "validation", ["review_id", "text", "duplicate_group_id"], frozen["scoring_ids"]
        )
        candidates = validation.to_dicts()
        selected_ids: set[str] = set()
        for label in categories:
            choices = sorted(
                [row for row in candidates if re.search(DISCOVERY[label], row["text"], re.I)],
                key=lambda row: ranked(row["review_id"], f"target:{label}"),
            )
            for row in [row for row in choices if row["review_id"] not in selected_ids][
                :per_category
            ]:
                selected_ids.add(row["review_id"])
                targeted.append(row)
                sampling_keys.append(
                    {
                        "review_id": row["review_id"],
                        "duplicate_group_id": row["duplicate_group_id"],
                        "sampling_category": label,
                    }
                )
        targeted.sort(key=lambda row: ranked(row["review_id"], "target-display"))
    export_working(directory / "annotation_working.csv", primary)
    if targeted:
        export_working(directory / "annotation_targeted_optional.csv", targeted)
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "main_rows": 240,
        "optional_rows": len(targeted),
        "primary_ids": [row["review_id"] for row in primary],
        "optional_reservation": sampling_keys,
        "optional_display_ids": [row["review_id"] for row in targeted],
        "target_categories": categories,
        "per_category_cap": per_category,
        "seed": SEED,
        "sampling": (
            "Optional keyword-enriched validation subset, seeded orderi"
            "ng, no model scores or labels"
        ),
        "discovery_patterns": {label: DISCOVERY[label] for label in categories},
        "master_sha256": file_hash(master),
        "rules_sha256_at_reservation": rules_hash,
        "taxonomy_sha256_at_reservation": taxonomy_hash,
        "human_labels_created": False,
        "future_use": (
            "Main and optional IDs/families excluded from complaint fit"
            "ting/tuning; no change to old sentiment memberships"
        ),
        "initial_working_sha256": file_hash(directory / "annotation_working.csv"),
    }
    if targeted:
        result["initial_optional_sha256"] = file_hash(
            directory / "annotation_targeted_optional.csv"
        )
    write_json(provenance_path, result)
    return result


def main() -> None:
    sys.addaudithook(deny_sealed_files)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "prepare"])
    parser.add_argument("--supplement-categories", nargs="*", default=[])
    parser.add_argument("--per-category", type=int, default=12)
    args = parser.parse_args()
    result = (
        audit()
        if args.command == "audit"
        else prepare(args.supplement_categories, args.per_category)
    )
    print(
        json.dumps(
            result
            if args.command == "audit"
            else {
                "main_rows": result["main_rows"],
                "optional_rows": result["optional_rows"],
                "human_labels_created": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
