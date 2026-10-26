"""Bounded acquisition and source-to-canonical review normalization."""

import hashlib
import json
import math
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from signaldesk.contracts import CustomerReview

SOURCE = "amazon_reviews_2023"
MIN_DATE = datetime(1996, 1, 1, tzinfo=UTC)
MAX_DATE = datetime(2023, 10, 1, tzinfo=UTC)
ID_PATTERN = re.compile(r"^[A-Z0-9]{10}$")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def acquire(config: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    """Reuse a checksum-verified cache, or download one category without remote code."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "Appliances.jsonl.gz"
    provenance_path = raw_dir / "provenance.json"
    if path.exists():
        if not provenance_path.exists():
            raise ValueError("Raw file exists without provenance; refusing to silently replace it")
        provenance: dict[str, Any] = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance["requested_url"] != config["source_url"]:
            raise ValueError("Cached source URL differs from configuration")
        if path.stat().st_size > config["max_download_bytes"]:
            raise ValueError("Cached source exceeds configured download bound")
        if file_hash(path) != provenance["sha256"]:
            raise ValueError("Raw source checksum mismatch")
        return provenance

    request = urllib.request.Request(
        config["source_url"], headers={"User-Agent": "SignalDesk-academic-data-audit/0.1"}
    )
    temporary = path.with_suffix(".gz.part")
    digest = hashlib.sha256()
    total = 0
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as output,
        ):
            expected = int(response.headers.get("Content-Length", "0"))
            if expected > config["max_download_bytes"]:
                raise ValueError("Source Content-Length exceeds configured download bound")
            if expected != config["expected_source_bytes"]:
                raise ValueError("Source size changed since preflight; re-audit before acquisition")
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > config["max_download_bytes"]:
                    raise ValueError("Download exceeds configured bound")
                output.write(chunk)
                digest.update(chunk)
            if total != expected:
                raise ValueError("Incomplete source download")
            provenance = {
                "requested_url": config["source_url"],
                "resolved_url": response.url,
                "downloaded_at": datetime.now(UTC).isoformat(),
                "bytes": total,
                "sha256": digest.hexdigest(),
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "content_type": response.headers.get("Content-Type"),
                "usage_guidance": "Research-oriented; maintainers assign no conventional license. "
                "No unrestricted redistribution claim.",
                "usage_guidance_url": "https://huggingface.co/datasets/McAuley-Lab/"
                "Amazon-Reviews-2023/discussions/1",
                "dataset_page": "https://amazon-reviews-2023.github.io/",
                "remote_code_executed": False,
                "metadata_acquired": False,
            }
        temporary.replace(path)
        write_json(provenance_path, provenance)
        return provenance
    finally:
        temporary.unlink(missing_ok=True)


def valid_id(value: Any) -> str | None:
    return value if isinstance(value, str) and ID_PATTERN.fullmatch(value) else None


def timestamp_utc(value: Any) -> tuple[datetime | None, str]:
    """Accept verified second/millisecond forms, reject out-of-snapshot values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None, "invalid"
    unit = "milliseconds" if value >= 100_000_000_000 else "seconds"
    try:
        result = datetime.fromtimestamp(value / 1000 if unit == "milliseconds" else value, tz=UTC)
    except (ValueError, OverflowError, OSError):
        return None, "invalid"
    return (result, unit) if MIN_DATE <= result < MAX_DATE else (None, "invalid")


def weak_label(rating: float) -> str:
    # Ratings provide weak supervision for sentiment, not independent human judgments.
    # They create targets only; the classifiers receive review-body text as input.
    return "negative" if rating <= 2 else "neutral" if rating == 3 else "positive"


def normalize(
    row: dict[str, Any], number: int, revision: str, ingested_at: datetime
) -> tuple[dict[str, Any] | None, list[str], str]:
    """Validate without title concatenation, stemming, punctuation/negation removal or IDs."""
    reasons = []
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        reasons.append("missing_or_blank_text")
    elif len(text) > 50000:
        reasons.append("text_exceeds_contract_limit")
    rating = row.get("rating")
    if (
        isinstance(rating, bool)
        or not isinstance(rating, (int, float))
        or rating not in (1, 2, 3, 4, 5)
    ):
        reasons.append("invalid_rating")
    parent = valid_id(row.get("parent_asin"))
    asin = valid_id(row.get("asin"))
    if parent is None:
        reasons.append("missing_or_invalid_parent_asin")
    reviewed_at, unit = timestamp_utc(row.get("timestamp", row.get("sort_timestamp")))
    if reviewed_at is None:
        reasons.append("invalid_timestamp")
    if reasons:
        return None, reasons, unit
    assert isinstance(rating, (int, float))

    helpful = row.get("helpful_vote", row.get("helpful_votes"))
    helpful = helpful if type(helpful) is int and helpful >= 0 else None
    verified = row.get("verified_purchase")
    title = row.get("title")
    review = CustomerReview(
        review_id=f"amazon2023:appliances:{number}",
        product_id=f"amazon2023:{parent}",
        source=SOURCE,
        source_revision=revision,
        text=text,
        title=title if isinstance(title, str) else None,
        rating=rating,
        reviewed_at=reviewed_at,
        ingested_at=ingested_at,
        language=None,  # Heuristics below are flags, not verified language labels.
    )
    return (
        {
            **review.model_dump(),
            "asin": asin,
            "parent_asin": parent,
            "verified_purchase": verified if type(verified) is bool else None,
            "helpful_vote": helpful,
            "model_text": review.text,
            "sentiment_proxy": weak_label(float(rating)),
            "label_source": "rating_proxy_v1",
            "text_sha256": hashlib.sha256(review.text.encode("utf-8")).hexdigest(),
        },
        [],
        unit,
    )
