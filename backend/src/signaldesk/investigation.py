"""Read-only adapter for the frozen investigation cohort; no prediction API.

Run `python -m signaldesk.investigation --as-of 2023-09-12T00:00:00+00:00`
for a compact historical smoke check. Nothing is saved or fitted.
"""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import polars as pl

from signaldesk.config import PROJECT_ROOT
from signaldesk.product_intelligence import HistoricalReview, TextState, build_intelligence
from signaldesk_data.source import file_hash


@dataclass(frozen=True)
class InvestigationSource:
    reviews: tuple[HistoricalReview, ...]
    dataset_revision: str
    manifest_sha256: str
    investigation_sha256: str
    reservation_sha256: dict[str, str]
    source_revisions: tuple[tuple[str, str], ...]
    rules_sha256: str
    body_access_counts: dict[str, int]


def load_investigation(root: Path = PROJECT_ROOT) -> InvestigationSource:
    """Load metadata first, then only unreserved `not_selected` family bodies.

    No train/validation/test, full sentiment or split-manifest Parquet is opened.
    Frozen annotation provenance is used only as an exclusion list, never labels.
    """
    directory = root / "data/processed/appliances-v1"
    manifest_path = directory / "manifest.json"
    path = directory / "investigation.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksum = file_hash(path)
    if checksum != manifest["files"][path.name]["sha256"]:
        raise ValueError("Frozen investigation checksum mismatch")
    primary_path = root / "data/annotations/sentiment-v1/internal/provenance.json"
    optional_path = root / "data/annotations/sentiment-v1/complaints-v1/internal/provenance.json"
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    optional = json.loads(optional_path.read_text(encoding="utf-8"))
    manifest_hash = file_hash(manifest_path)
    if primary["dataset_manifest_sha256"] != manifest_hash:
        raise ValueError("Annotation reservation belongs to another dataset manifest")
    reserved_ids = {row["review_id"] for row in primary["reserved"]}
    if reserved_ids != set(optional["primary_ids"]):
        raise ValueError("Primary reservation metadata disagrees")
    reserved_ids.update(row["review_id"] for row in optional["optional_reservation"])
    reserved_groups = {row["duplicate_group_id"] for row in optional["optional_reservation"]}
    metadata = pl.read_parquet(
        path,
        columns=[
            "review_id",
            "product_id",
            "rating",
            "reviewed_at",
            "duplicate_group_id",
            "related_sentiment_split",
            "text_sha256",
            "source",
            "source_revision",
            "cohort_role",
        ],
    )
    if metadata.height != manifest["files"][path.name]["rows"]:
        raise ValueError("Frozen investigation row count mismatch")
    if metadata["review_id"].n_unique() != metadata.height:
        raise ValueError("Frozen source contains repeated review IDs")
    if metadata.filter(pl.col("cohort_role") != "investigation_only").height:
        raise ValueError("Only the historical investigation cohort is permitted")
    if metadata["related_sentiment_split"].null_count() or not set(
        metadata["related_sentiment_split"]
    ) <= {"not_selected", "train", "validation", "test"}:
        raise ValueError("Unknown protected family assignment")
    if metadata["duplicate_group_id"].null_count():
        raise ValueError("Missing duplicate-family protection")
    reserved_groups.update(
        metadata.filter(pl.col("review_id").is_in(sorted(reserved_ids)))["duplicate_group_id"]
    )
    # Propagate any protection across a family before materialising any review body.
    protected_groups = set(
        metadata.filter(pl.col("related_sentiment_split") != "not_selected")["duplicate_group_id"]
    )
    allowed = metadata.filter(
        (pl.col("related_sentiment_split") == "not_selected")
        & ~pl.col("review_id").is_in(sorted(reserved_ids))
        & ~pl.col("duplicate_group_id").is_in(sorted(reserved_groups | protected_groups))
    )
    bodies = (
        pl.scan_parquet(path)
        .filter(pl.col("review_id").is_in(allowed["review_id"].to_list()))
        .select("review_id", "text")
        .collect()
    )
    if set(bodies["review_id"]) != set(allowed["review_id"]):
        raise ValueError("Eligible source body IDs are missing")
    expected_hashes = dict(allowed.select("review_id", "text_sha256").iter_rows())
    texts: dict[str, str | None] = dict(bodies.iter_rows())
    for identifier, text in texts.items():
        if (
            text is not None
            and hashlib.sha256(text.encode("utf-8")).hexdigest() != expected_hashes[identifier]
        ):
            raise ValueError("Original investigation body checksum mismatch")
    reviews = []
    counts: Counter[str] = Counter()
    for row in metadata.iter_rows(named=True):
        identifier, group = row["review_id"], row["duplicate_group_id"]
        state: TextState
        text = texts.get(identifier)
        if identifier in reserved_ids or group in reserved_groups:
            state = "reserved_annotation"
        elif group in protected_groups:
            state = "protected_family"
        elif not text or not text.strip() or len(text) > 50000:
            state = "unavailable"
            text = None
        else:
            state = "available"
        counts[state] += 1
        reviews.append(
            HistoricalReview(
                identifier,
                row["product_id"],
                row["rating"],
                row["reviewed_at"],
                text,
                state,
            )
        )
    return InvestigationSource(
        tuple(reviews),
        "appliances-v1/investigation",
        manifest_hash,
        checksum,
        {"primary": file_hash(primary_path), "optional": file_hash(optional_path)},
        tuple(sorted(set(metadata.select("source", "source_revision").iter_rows()))),
        file_hash(root / "ml/signaldesk_ml/complaint_rules.py"),
        dict(counts),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True, type=datetime.fromisoformat)
    parser.add_argument(
        "--queue-output", type=Path, help="Optional small JSON queue; refuses overwrite"
    )
    args = parser.parse_args()
    if args.queue_output is not None and args.queue_output.exists():
        parser.error("Queue output already exists; choose a new snapshot path")
    source = load_investigation()
    result = build_intelligence(source.reviews, as_of=args.as_of)
    summary = {
        "engine_version": result.engine_version,
        "as_of_exclusive": result.as_of.isoformat(),
        "source": {
            field.name: getattr(source, field.name)
            for field in fields(source)
            if field.name != "reviews"
        },
        "policy": asdict(result.policy),
        "totals": asdict(result.totals),
        "products": len(result.products),
        "granularity_counts": dict(Counter(product.granularity for product in result.products)),
        "candidate_signals": len(result.queue),
        "candidate_products": len({item.product_id for item in result.queue}),
        "candidate_reasons": dict(Counter(item.reason for item in result.queue)),
        "future_rows_excluded": result.future_rows_excluded,
        "limitations": result.limitations,
    }
    if args.queue_output is not None:
        args.queue_output.parent.mkdir(parents=True, exist_ok=True)
        with args.queue_output.open("x", encoding="utf-8") as stream:
            json.dump(
                {"summary": summary, "candidates": [asdict(item) for item in result.queue]},
                stream,
                indent=2,
                default=_json_default,
            )
            stream.write("\n")
    print(json.dumps(summary, indent=2))


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError("Unsupported historical snapshot value")


if __name__ == "__main__":
    main()
