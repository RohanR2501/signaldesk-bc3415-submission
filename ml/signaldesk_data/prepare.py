"""One CLI for acquisition, auditing, cohort preparation and frozen partitions."""

import argparse
import gzip
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from signaldesk.config import PROJECT_ROOT
from signaldesk.contracts import CustomerReview
from signaldesk_data.duplicates import duplicate_groups, normalized_text
from signaldesk_data.source import acquire, file_hash, normalize, valid_id, write_json

CANONICAL_FIELDS = set(CustomerReview.model_fields)
LABELS = {"negative", "neutral", "positive"}


def source_rows(path: Path, config: dict[str, Any]) -> Iterator[tuple[int, dict[str, Any] | None]]:
    total = 0
    with gzip.open(path, "rb") as stream:
        number = 0
        while line := stream.readline(config["max_line_bytes"] + 1):
            number += 1
            total += len(line)
            if len(line) > config["max_line_bytes"] or total > config["max_uncompressed_bytes"]:
                raise ValueError("Uncompressed source/line bound exceeded")
            if number > config["max_records"]:
                raise ValueError("Source row bound exceeded")
            try:
                value = json.loads(line)
                yield number, value if isinstance(value, dict) else None
            except (ValueError, UnicodeError):
                yield number, None


def weighted_quantiles(counts: Counter[int]) -> dict[str, int]:
    total = counts.total()
    result = {}
    for name, fraction in [("min", 0), ("p25", 0.25), ("median", 0.5), ("p95", 0.95), ("max", 1)]:
        accumulated = 0
        for value, count in sorted(counts.items()):
            accumulated += count
            if accumulated >= max(1, total * fraction):
                result[name] = value
                break
    return result


def quality_flags(text: str) -> list[str]:
    """Low-cost suspicions only; never assert English or infer a verified language."""
    flags = []
    words = re.findall(r"\w+", text.casefold())
    letters = [char for char in text if char.isalpha()]
    if len(words) < 3:
        flags.append("fewer_than_three_tokens")
    if letters and sum(ord(char) > 591 for char in letters) / len(letters) > 0.2:
        flags.append("substantial_non_latin_script")
    foreign_markers = {
        "muy",
        "pero",
        "para",
        "producto",
        "funciona",
        "gracias",
        "avec",
        "pour",
        "tres",
        "produit",
        "mais",
        "bien",
    }
    if len(set(words) & foreign_markers) >= 3:
        flags.append("possible_non_english_latin_text")
    if re.search(r"<[^>]+>|&(?:amp|lt|gt|nbsp);", text):
        flags.append("html_or_entities")
    return flags


def scan_source(
    path: Path, config: dict[str, Any], provenance: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, list[Any]], dict[str, Any]]:
    rng = random.Random(config["seed"])
    sample: list[dict[str, Any]] = []
    products: dict[str, list[Any]] = {}
    reasons: Counter[str] = Counter()
    types: dict[str, Counter[str]] = defaultdict(Counter)
    empty: Counter[str] = Counter()
    ratings: Counter[str] = Counter()
    units: Counter[str] = Counter()
    years: Counter[str] = Counter()
    lengths: Counter[int] = Counter()
    exact: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    valid_count = rejected_count = record_count = 0
    earliest: datetime | None = None
    latest: datetime | None = None
    ingested = datetime.fromisoformat(provenance["downloaded_at"])
    for number, row in source_rows(path, config):
        record_count += 1
        if row is None:
            reasons["invalid_json_record"] += 1
            rejected_count += 1
            continue
        for key, value in row.items():
            types[key][type(value).__name__] += 1
            empty[key] += value is None or value == ""
        rating = row.get("rating")
        if type(rating) in (float, int) and rating in (1, 2, 3, 4, 5):
            ratings[str(int(rating))] += 1
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            lengths[len(text)] += 1
            exact[hashlib.sha256(text.encode()).hexdigest()] += 1
        clean, failures, unit = normalize(row, number, provenance["sha256"], ingested)
        units[unit] += 1
        if clean is None:
            rejected_count += 1
            reasons.update(failures)
            continue
        valid_count += 1
        timestamp = clean["reviewed_at"]
        years[str(timestamp.year)] += 1
        flags.update(quality_flags(clean["text"]))
        earliest = timestamp if earliest is None else min(earliest, timestamp)
        latest = timestamp if latest is None else max(latest, timestamp)
        product_id = clean["product_id"]
        if product_id not in products:
            products[product_id] = [0, timestamp, timestamp]
        product = products[product_id]
        product[0] += 1
        product[1], product[2] = min(product[1], timestamp), max(product[2], timestamp)
        if len(sample) < config["sentiment_sample_size"]:
            sample.append(clean)
        else:
            slot = rng.randrange(valid_count)
            if slot < len(sample):
                sample[slot] = clean
        if number % 250000 == 0:
            print(f"Inspected {number:,} source rows; eligible {valid_count:,}", flush=True)
    if not earliest or not latest:
        raise ValueError("No usable records; no datasets or split manifest released")
    product_counts = sorted((int(v[0]) for v in products.values()), reverse=True)
    return (
        sample,
        products,
        {
            "source_records": record_count,
            "eligible_records": valid_count,
            "rejected_records": rejected_count,
            "rejection_reasons_nonexclusive": dict(reasons),
            "schema": {k: {**v, "absent": record_count - v.total()} for k, v in types.items()},
            "null_or_empty_fields": dict(empty),
            "rating_counts_source": dict(ratings),
            "timestamp_units": dict(units),
            "eligible_year_counts": dict(years),
            "text_length_characters": weighted_quantiles(lengths),
            "exact_text_duplicate_excess_rows": sum(c - 1 for c in exact.values()),
            "exact_text_duplicate_affected_rows": sum(c for c in exact.values() if c > 1),
            "quality_flags_eligible": dict(flags),
            "products": len(products),
            "product_review_count_quantiles": weighted_quantiles(Counter(product_counts)),
            "top_10_product_share": sum(product_counts[:10]) / valid_count,
            "top_100_product_share": sum(product_counts[:100]) / valid_count,
            "time_min_utc": earliest.isoformat(),
            "time_max_utc": latest.isoformat(),
            "usable_parent_id_records": record_count
            - reasons["missing_or_invalid_parent_asin"]
            - reasons["invalid_json_record"],
        },
    )


def select_products(products: dict[str, list[Any]], config: dict[str, Any]) -> list[str]:
    eligible = [
        key
        for key, (count, start, end) in products.items()
        if config["investigation_min_reviews"] <= count <= config["investigation_max_reviews"]
        and (end - start).days >= config["investigation_min_span_days"]
    ]
    eligible.sort(key=lambda key: hashlib.sha256(f"{config['seed']}:{key}".encode()).digest())
    selected = []
    total = 0
    for key in eligible:
        count = products[key][0]
        if total + count <= config["investigation_max_total_reviews"]:
            selected.append(key)
            total += count
        if len(selected) == config["investigation_product_count"]:
            break
    if not selected:
        raise ValueError("No products satisfy investigation history constraints")
    return selected


def chronological_partitions(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, str], list[str]]:
    # Later reviews approximate future use; UTC-day cutoffs avoid splitting one day.
    # Inputs already contain one representative per duplicate family to prevent leakage.
    ordered = sorted(rows, key=lambda row: (row["reviewed_at"], row["review_id"]))
    cut_days = [
        ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]["reviewed_at"].replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        for fraction in [
            config["train_fraction"],
            config["train_fraction"] + config["validation_fraction"],
        ]
    ]
    if cut_days[0] >= cut_days[1]:
        raise ValueError("Insufficient temporal coverage for three chronological partitions")
    result = {
        row["review_id"]: "train"
        if row["reviewed_at"] < cut_days[0]
        else "validation"
        if row["reviewed_at"] < cut_days[1]
        else "test"
        for row in rows
    }
    return result, [day.isoformat() for day in cut_days]


def frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema_overrides={
            "language": pl.String,
            "title": pl.String,
            "asin": pl.String,
            "helpful_vote": pl.Int64,
            "verified_purchase": pl.Boolean,
            "rating": pl.Float64,
            "reviewed_at": pl.Datetime("us", "UTC"),
            "ingested_at": pl.Datetime("us", "UTC"),
        },
    )


def profile(df: pl.DataFrame) -> dict[str, Any]:
    counts = df.group_by("product_id").len()["len"].to_list()
    earliest, latest = df["reviewed_at"].min(), df["reviewed_at"].max()
    if not isinstance(earliest, datetime) or not isinstance(latest, datetime):
        raise ValueError("Cannot profile an empty or non-temporal cohort")
    return {
        "rows": df.height,
        "products": df["product_id"].n_unique(),
        "ratings": {str(int(k)): count for k, count in df.group_by("rating").len().iter_rows()},
        "labels": dict(df.group_by("sentiment_proxy").len().iter_rows()),
        "years": {
            str(year): count
            for year, count in df.with_columns(pl.col("reviewed_at").dt.year().alias("year"))
            .group_by("year")
            .len()
            .iter_rows()
        },
        "product_counts": weighted_quantiles(Counter(counts)),
        "time_min_utc": earliest.isoformat(),
        "time_max_utc": latest.isoformat(),
    }


def membership_hash(df: pl.DataFrame) -> str:
    lines = [
        f"{review_id}\t{split}"
        for review_id, split in df.select("review_id", "split").sort("review_id").iter_rows()
    ]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def verify(output: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        if file_hash(output / name) != record["sha256"]:
            raise ValueError(f"Frozen output checksum mismatch: {name}")
    sentiment = pl.read_parquet(output / "sentiment.parquet")
    partitions, cutoffs = chronological_partitions(sentiment.to_dicts(), manifest["configuration"])
    if cutoffs != manifest["split_cutoffs_utc"]:
        raise ValueError("Chronological split cutoffs do not reproduce")
    if membership_hash(sentiment) != manifest["split_membership_sha256"]:
        raise ValueError("Split membership hash mismatch")
    if sentiment["review_id"].n_unique() != sentiment.height:
        raise ValueError("Duplicate review IDs in modelling cohort")
    if sentiment["duplicate_group_id"].n_unique() != sentiment.height:
        raise ValueError(
            "Detected exact/near-duplicate family crosses or repeats within modelling partitions"
        )
    seen: set[str] = set()
    for split in ["train", "validation", "test"]:
        partition = pl.read_parquet(output / f"{split}.parquet")
        ids = set(partition["review_id"].to_list())
        if (
            not ids
            or seen & ids
            or ids != {key for key, value in partitions.items() if value == split}
        ):
            raise ValueError("Partition membership failed deterministic/disjointness checks")
        if set(partition["sentiment_proxy"].unique()) != LABELS:
            raise ValueError("A split lacks one of the three rating-proxy classes")
        seen.update(ids)
    if len(seen) != sentiment.height:
        raise ValueError("Partition union differs from cleaned cohort")
    if len({normalized_text(text) for text in sentiment["model_text"]}) != sentiment.height:
        raise ValueError("Normalized exact text leakage")
    for name in ["sentiment.parquet", "investigation.parquet"]:
        df = pl.read_parquet(output / name)
        if "user_id" in df.columns or "images" in df.columns:
            raise ValueError("Unnecessary source fields leaked into processed data")
        if df.schema["reviewed_at"] != pl.Datetime("us", "UTC"):
            raise ValueError("Timestamp type/timezone mismatch")
        if not {"review_id", "product_id", "parent_asin", "model_text", "sentiment_proxy"} <= set(
            df.columns
        ):
            raise ValueError("Missing required prepared columns")
        for row in df.iter_rows(named=True):
            CustomerReview.model_validate({key: row[key] for key in CANONICAL_FIELDS})
            if (
                not row["model_text"].strip()
                or row["model_text"] != row["text"]
                or row["sentiment_proxy"] not in LABELS
            ):
                raise ValueError("Invalid model input or proxy label")
    investigation = pl.read_parquet(output / "investigation.parquet")
    sample_ids = set(sentiment["review_id"].to_list())
    protected = dict(sentiment.select("duplicate_group_id", "split").iter_rows())
    for row in investigation.iter_rows(named=True):
        if row["cohort_role"] != "investigation_only" or row["is_sentiment_member"] != (
            row["review_id"] in sample_ids
        ):
            raise ValueError("Investigation cohort role/overlap mismatch")
        if row["related_sentiment_split"] != protected.get(
            row["duplicate_group_id"], "not_selected"
        ):
            raise ValueError("Investigation text family protection mismatch")
    return {
        "status": "passed",
        "sentiment_rows": sentiment.height,
        "investigation_rows": investigation.height,
        "split_membership_sha256": manifest["split_membership_sha256"],
    }


def prepare(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = PROJECT_ROOT / config["output_directory"]
    if (output / "manifest.json").exists():
        existing = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if existing["configuration"] != config:
            raise ValueError(
                "Frozen run configuration changed; existing test partition will not be overwritten"
            )
        return verify(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Output directory contains an incomplete/unrelated run; refusing overwrite"
        )
    provenance = acquire(config, PROJECT_ROOT / "data/raw")
    path = PROJECT_ROOT / "data/raw/Appliances.jsonl.gz"
    print("Auditing full category and drawing a seeded uniform reservoir...", flush=True)
    sample, product_stats, source_audit = scan_source(path, config, provenance)
    selected = select_products(product_stats, config)
    selected_set = set(selected)
    investigation = []
    ingested = datetime.fromisoformat(provenance["downloaded_at"])
    for number, row in source_rows(path, config):
        if row and f"amazon2023:{valid_id(row.get('parent_asin'))}" in selected_set:
            clean, _, _ = normalize(row, number, provenance["sha256"], ingested)
            if clean:
                investigation.append(clean)
    if len(investigation) != sum(product_stats[key][0] for key in selected):
        raise ValueError("Investigation histories are incomplete")
    # Group both cohorts together so shared/near-duplicate histories inherit protection.
    union = {row["review_id"]: row for row in [*sample, *investigation]}
    print(f"Checking duplicate families across {len(union):,} cohort-union records...", flush=True)
    groups, duplicate_audit = duplicate_groups(list(union.values()), config)
    representatives: dict[str, dict[str, Any]] = {}
    for row in sample:
        row["duplicate_group_id"] = groups[row["review_id"]]
        group = row["duplicate_group_id"]
        # Choose a representative without consulting its rating-derived target.
        rank = hashlib.sha256(f"{config['seed']}:{row['review_id']}".encode()).digest()
        if (
            group not in representatives
            or rank
            < hashlib.sha256(
                f"{config['seed']}:{representatives[group]['review_id']}".encode()
            ).digest()
        ):
            representatives[group] = row
    sentiment_rows = list(representatives.values())
    if not 30000 <= len(sentiment_rows) <= 60000:
        raise ValueError(
            "Usable modelling cohort is outside the approved 30k–60k target; inspect audit"
        )
    assignments, cutoffs = chronological_partitions(sentiment_rows, config)
    for row in sentiment_rows:
        row["split"] = assignments[row["review_id"]]
        row["cohort_role"] = "sentiment_modelling"
    protected = {row["duplicate_group_id"]: row["split"] for row in sentiment_rows}
    sentiment_ids = set(assignments)
    for row in investigation:
        row["duplicate_group_id"] = groups[row["review_id"]]
        row["cohort_role"] = "investigation_only"
        row["is_sentiment_member"] = row["review_id"] in sentiment_ids
        row["related_sentiment_split"] = protected.get(row["duplicate_group_id"], "not_selected")
        row.pop("split", None)
    # Copies prevent shared Python dictionaries from mixing the two dataset roles.
    sentiment = (
        frame(
            [
                {
                    **row,
                    "cohort_role": "sentiment_modelling",
                    "split": assignments[row["review_id"]],
                }
                for row in sentiment_rows
            ]
        )
        .drop("is_sentiment_member", "related_sentiment_split", strict=False)
        .sort("reviewed_at", "review_id")
    )
    incidents = frame(investigation).sort("product_id", "reviewed_at", "review_id")
    reservoir = frame(
        [
            {
                key: value
                for key, value in row.items()
                if key
                not in {"is_sentiment_member", "related_sentiment_split", "split", "cohort_role"}
            }
            for row in sample
        ]
    )
    audit = {
        "source": source_audit,
        "duplicates_cohort_union": duplicate_audit,
        "reservoir": profile(reservoir),
        "sentiment": profile(sentiment),
        "investigation": profile(incidents),
        "splits": {
            split: profile(sentiment.filter(pl.col("split") == split))
            for split in ["train", "validation", "test"]
        },
        "removed_from_reservoir_duplicate_rows": len(sample) - sentiment.height,
        "shared_review_ids": incidents["is_sentiment_member"].sum(),
        "selected_product_source_counts": {key: product_stats[key][0] for key in selected},
        "selected_product_span_days": weighted_quantiles(
            Counter((product_stats[key][2] - product_stats[key][1]).days for key in selected)
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        "sentiment.parquet": sentiment,
        "investigation.parquet": incidents,
        "split_manifest.parquet": sentiment.select(
            "review_id", "product_id", "duplicate_group_id", "reviewed_at", "split"
        ),
    }
    frames.update(
        {
            f"{split}.parquet": sentiment.filter(pl.col("split") == split)
            for split in ["train", "validation", "test"]
        }
    )
    for name, df in frames.items():
        df.write_parquet(output / name, compression="zstd")
    write_json(output / "audit.json", audit)
    files = {
        name: {
            "sha256": file_hash(output / name),
            "rows": df.height,
            "bytes": (output / name).stat().st_size,
        }
        for name, df in frames.items()
    }
    files["audit.json"] = {
        "sha256": file_hash(output / "audit.json"),
        "bytes": (output / "audit.json").stat().st_size,
    }
    code_files = sorted((PROJECT_ROOT / "ml/signaldesk_data").glob("*.py"))
    manifest = {
        "configuration": config,
        "source_provenance": provenance,
        "created_at": datetime.now(UTC).isoformat(),
        "split_cutoffs_utc": cutoffs,
        "split_membership_sha256": membership_hash(sentiment),
        "files": files,
        "configuration_sha256": file_hash(config_path),
        "uv_lock_sha256": file_hash(PROJECT_ROOT / "uv.lock"),
        "code_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): file_hash(p) for p in code_files},
        "schema_version": "1.0",
        "split_strategy": "chronological_utc_day_after_label_blind_deduplication",
    }
    write_json(output / "manifest.json", manifest)
    result = verify(output)
    from signaldesk_data.report import write_report

    write_report(audit, manifest, sentiment, incidents, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify"])
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config/data-preparation.json"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        result = (
            prepare(args.config)
            if args.command == "prepare"
            else verify(PROJECT_ROOT / config["output_directory"])
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Data preparation stopped: {exc}\n")
    print(json.dumps(result, indent=2))
