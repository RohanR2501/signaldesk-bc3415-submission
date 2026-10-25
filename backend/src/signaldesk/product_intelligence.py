"""Historical, probability-free review aggregation and human reading priorities.

These internal service results are not API contracts or verified incident records.
Complaint rules are reused directly; training and evaluation modules are never imported.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from typing import Literal

from signaldesk_ml.complaint_rules import (
    RULE_VERSION,
    TAXONOMY_VERSION,
    ComplaintCategory,
    ComplaintEvidence,
    detect_complaints,
)

ENGINE_VERSION = "product-intelligence-sprint-a-v1"
TextState = Literal["available", "protected_family", "reserved_annotation", "unavailable"]
Period = Literal["month", "quarter"]
LIMITATIONS = (
    "Historical selected-product snapshot, not current/live or population prevalence.",
    "Ratings 1–2 are negative-rating proxies, not text sentiment or defect findings.",
    "Rule matches are unvalidated customer reports; no match is unknown, not issue-free.",
    "Rule coverage excludes protected families/reservations and invalid/missing bodies.",
    "Defect rules have low coverage and can confuse scope/design with malfunction.",
    "Arrival, wrong-item, delivery and service have insufficient independent positive support.",
    "Wrong-item rules miss known examples; NLI/student overprediction precludes findings.",
    "Delivery student unavailable; delivery rules/NLI have unreliable detection.",
    "Priorities are configurable reading heuristics, not validated alerts or probabilities.",
    "Queue retains the most recent qualifying historical window per product/metric; "
    "these are not current incidents.",
    "Distinct source review IDs count once; copied text families may still contribute separately.",
)


@dataclass(frozen=True)
class HistoricalReview:
    review_id: str
    product_id: str
    rating: float | None
    reviewed_at: datetime | None
    text: str | None = None
    text_state: TextState = "unavailable"


@dataclass(frozen=True)
class PriorityPolicy:
    """Declared reading-budget heuristics; never fitted on frozen evaluation outcomes."""

    min_denominator: int = 20
    min_reports: int = 3
    negative_increase: float = 0.10
    complaint_increase: float = 0.05
    sustained_negative_share: float = 0.30
    sustained_complaint_share: float = 0.10
    min_sustained_reports: int = 5
    evidence_limit: int = 5

    def __post_init__(self) -> None:
        if min(self.min_denominator, self.min_reports, self.evidence_limit) < 1:
            raise ValueError("Support and evidence limits must be positive")
        if self.min_sustained_reports < self.min_reports:
            raise ValueError("Sustained support cannot be below minimum reports")
        for share in (
            self.negative_increase,
            self.complaint_increase,
            self.sustained_negative_share,
            self.sustained_complaint_share,
        ):
            if not isfinite(share) or not 0 < share <= 1:
                raise ValueError("Policy shares must be finite and in (0, 1]")


@dataclass(frozen=True)
class ReviewMetrics:
    unique_reviews: int
    rated_reviews: int
    negative_rating_reviews: int
    negative_rating_share: float | None
    rule_scanned_reviews: int
    any_complaint_reviews: int
    complaint_reviews: dict[ComplaintCategory, int]
    complaint_shares: dict[ComplaintCategory, float | None]
    unknown_no_evidence_reviews: int
    text_unavailable_reasons: dict[str, int]
    missing_timestamp_reviews: int
    sentiment_model_state: Literal["unavailable"] = "unavailable"
    complaint_model_states: tuple[str, ...] = (
        "nli:not_used",
        "linear:not_used",
        "delivery_linear:unavailable",
        "model_abstentions:not_available_not_zero",
    )
    negative_provenance: str = "source_rating:1–2 / valid_ratings:1–5"
    complaint_provenance: str = f"{TAXONOMY_VERSION}/{RULE_VERSION}:matched_reviews/scanned_reviews"
    review_provenance: str = "source_review_id:unique_count"


@dataclass(frozen=True)
class PeriodMetrics:
    start: datetime
    end: datetime
    metrics: ReviewMetrics


@dataclass(frozen=True)
class Comparison:
    metric: str
    state: Literal["comparable", "insufficient_support"]
    current_count: int
    current_denominator: int
    previous_count: int
    previous_denominator: int
    share_change: float | None


@dataclass(frozen=True)
class SourceExcerpt:
    review_id: str
    start: int
    end: int
    text: str
    method: Literal["rule_match", "source_excerpt"]
    rule_id: str | None = None
    resolution: str = "not_established"
    resolution_excerpt: str | None = None
    resolution_start: int | None = None
    resolution_end: int | None = None


@dataclass(frozen=True)
class InvestigationCandidate:
    product_id: str
    metric: str
    reason: Literal["rising_share", "sustained_burden"]
    comparison: Comparison
    window_start: datetime | None
    window_end: datetime
    window_report_count: int
    window_denominator: int
    supporting_review_ids: tuple[str, ...]
    evidence: tuple[SourceExcerpt, ...]
    evidence_unavailable_reviews: int
    status: Literal["candidate_requires_human_investigation"] = (
        "candidate_requires_human_investigation"
    )
    signal_type: Literal["historical_customer_reported_or_rating_proxy"] = (
        "historical_customer_reported_or_rating_proxy"
    )


@dataclass(frozen=True)
class ProductIntelligence:
    product_id: str
    totals: ReviewMetrics
    granularity: Period
    periods: tuple[PeriodMetrics, ...]
    comparisons: tuple[Comparison, ...]
    incomplete_period_reviews: int


@dataclass(frozen=True)
class IntelligenceResult:
    as_of: datetime
    policy: PriorityPolicy
    totals: ReviewMetrics
    products: tuple[ProductIntelligence, ...]
    queue: tuple[InvestigationCandidate, ...]
    duplicate_rows_ignored: int
    future_rows_excluded: int
    limitations: tuple[str, ...] = LIMITATIONS
    engine_version: str = ENGINE_VERSION


def _start(at: datetime, period: Period) -> datetime:
    at = at.astimezone(UTC)
    month = at.month if period == "month" else ((at.month - 1) // 3) * 3 + 1
    return datetime(at.year, month, 1, tzinfo=UTC)


def _shift(at: datetime, months: int) -> datetime:
    index = at.year * 12 + at.month - 1 + months
    return datetime(index // 12, index % 12 + 1, 1, tzinfo=UTC)


def _valid_rating(rating: float | None) -> bool:
    return rating is not None and isfinite(rating) and 1 <= rating <= 5


def _negative(row: HistoricalReview) -> bool:
    return _valid_rating(row.rating) and row.rating is not None and row.rating <= 2


def _ratio(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def _metrics(
    rows: list[HistoricalReview], detections: dict[str, tuple[ComplaintEvidence, ...]]
) -> ReviewMetrics:
    rated = sum(_valid_rating(row.rating) for row in rows)
    negative = sum(_negative(row) for row in rows)
    scanned = sum(row.review_id in detections for row in rows)
    counts = {category: 0 for category in ComplaintCategory}
    any_complaint = 0
    unavailable: Counter[str] = Counter()
    for row in rows:
        evidence = detections.get(row.review_id)
        if evidence is None:
            unavailable[row.text_state if row.text_state != "available" else "invalid_text"] += 1
            continue
        categories = {item.category for item in evidence}
        any_complaint += bool(categories)
        for category in categories:
            counts[category] += 1
    return ReviewMetrics(
        len(rows),
        rated,
        negative,
        _ratio(negative, rated),
        scanned,
        any_complaint,
        counts,
        {label: _ratio(count, scanned) for label, count in counts.items()},
        scanned - any_complaint,
        dict(unavailable),
        sum(row.reviewed_at is None for row in rows),
    )


def _comparison(
    metric: str, current: ReviewMetrics, previous: ReviewMetrics, minimum: int
) -> Comparison:
    if metric == "negative_rating":
        count, denominator = current.negative_rating_reviews, current.rated_reviews
        old_count, old_denominator = previous.negative_rating_reviews, previous.rated_reviews
    else:
        category = ComplaintCategory(metric)
        count, denominator = current.complaint_reviews[category], current.rule_scanned_reviews
        old_count = previous.complaint_reviews[category]
        old_denominator = previous.rule_scanned_reviews
    comparable = min(denominator, old_denominator) >= minimum
    return Comparison(
        metric,
        "comparable" if comparable else "insufficient_support",
        count,
        denominator,
        old_count,
        old_denominator,
        count / denominator - old_count / old_denominator if comparable else None,
    )


def _excerpt(row: HistoricalReview, evidence: ComplaintEvidence | None) -> SourceExcerpt:
    assert row.text is not None
    if evidence is None:
        # Context from a rated review, never a claim that these words explain a model.
        return SourceExcerpt(
            row.review_id, 0, min(len(row.text), 240), row.text[:240], "source_excerpt"
        )
    span = evidence.span
    if row.text[span.start : span.end] != span.text:
        raise ValueError("Rule evidence must match the original source exactly")
    resolution = evidence.resolution_span
    return SourceExcerpt(
        row.review_id,
        span.start,
        span.end,
        span.text,
        "rule_match",
        evidence.rule_id,
        evidence.resolution,
        resolution.text if resolution else None,
        resolution.start if resolution else None,
        resolution.end if resolution else None,
    )


def _candidate(
    product_id: str,
    comparison: Comparison,
    previous: PeriodMetrics,
    current: PeriodMetrics,
    rows: list[HistoricalReview],
    detections: dict[str, tuple[ComplaintEvidence, ...]],
    policy: PriorityPolicy,
) -> InvestigationCandidate | None:
    """Prioritise one complete historical comparison with inspectable source evidence."""
    metric = comparison.metric
    negative = metric == "negative_rating"
    increase = policy.negative_increase if negative else policy.complaint_increase
    rising = (
        comparison.share_change is not None
        and comparison.share_change + 1e-12 >= increase
        and comparison.current_count >= policy.min_reports
    )
    count = comparison.current_count + comparison.previous_count
    denominator = comparison.current_denominator + comparison.previous_denominator
    floor = policy.sustained_negative_share if negative else policy.sustained_complaint_share
    sustained = (
        comparison.state == "comparable"
        and count >= policy.min_sustained_reports
        and denominator > 0
        and count / denominator >= floor
    )
    if not rising and not sustained:
        return None
    start, end = current.start if rising else previous.start, current.end
    supporting = [
        row
        for row in rows
        if row.reviewed_at is not None
        and start <= row.reviewed_at < end
        and (
            _negative(row)
            if negative
            else any(item.category.value == metric for item in detections.get(row.review_id, ()))
        )
    ]
    supporting.sort(key=lambda row: (row.reviewed_at or end, row.review_id), reverse=True)
    available = [row for row in supporting if row.review_id in detections]
    if not available:
        return None
    excerpts = tuple(
        _excerpt(
            row,
            None
            if negative
            else next(item for item in detections[row.review_id] if item.category.value == metric),
        )
        for row in available[: policy.evidence_limit]
    )
    return InvestigationCandidate(
        product_id,
        metric,
        "rising_share" if rising else "sustained_burden",
        comparison,
        start,
        end,
        len(supporting),
        comparison.current_denominator if rising else denominator,
        tuple(row.review_id for row in supporting),
        excerpts,
        len(supporting) - len(available),
    )


def build_intelligence(
    reviews: Iterable[HistoricalReview],
    *,
    as_of: datetime,
    policy: PriorityPolicy | None = None,
) -> IntelligenceResult:
    """Aggregate unique IDs before as_of (exclusive), without fitting or model inference.

    Null timestamps stay in historical totals, but never enter temporal comparisons.
    Monthly comparisons fall back to adjacent complete calendar quarters when sparse.
    """
    if as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    as_of = as_of.astimezone(UTC)
    policy = policy or PriorityPolicy()
    seen: dict[str, HistoricalReview] = {}
    duplicates = future = 0
    for row in reviews:
        if not row.review_id or not row.product_id:
            raise ValueError("Source review and product identifiers are required")
        if row.reviewed_at is not None and row.reviewed_at.utcoffset() is None:
            raise ValueError("Review timestamps must be timezone-aware")
        if row.review_id in seen:
            if row != seen[row.review_id]:
                raise ValueError("Conflicting rows for one source review ID")
            duplicates += 1
            continue
        seen[row.review_id] = row
    rows = []
    detections: dict[str, tuple[ComplaintEvidence, ...]] = {}
    by_product: dict[str, list[HistoricalReview]] = defaultdict(list)
    for row in seen.values():
        if row.reviewed_at is not None and row.reviewed_at >= as_of:
            future += 1
            continue
        rows.append(row)
        by_product[row.product_id].append(row)
        if (
            row.text_state == "available"
            and row.text
            and row.text.strip()
            and len(row.text) <= 50000
        ):
            detections[row.review_id] = detect_complaints(row.text).evidence
    products = []
    queue = []
    for product_id, product_rows in sorted(by_product.items()):
        dated = [row for row in product_rows if row.reviewed_at is not None]
        monthly_end = _start(as_of, "month")
        monthly_counts = Counter(
            _start(row.reviewed_at, "month") for row in dated if row.reviewed_at
        )
        period: Period = (
            "month"
            if min(monthly_counts[_shift(monthly_end, -1)], monthly_counts[_shift(monthly_end, -2)])
            >= policy.min_denominator
            else "quarter"
        )
        step = 1 if period == "month" else 3
        end = _start(as_of, period)
        previous_start = _shift(end, -2 * step)
        buckets: dict[datetime, list[HistoricalReview]] = defaultdict(list)
        for row in dated:
            if row.reviewed_at and row.reviewed_at < end:
                buckets[_start(row.reviewed_at, period)].append(row)
        cursor = min([previous_start, *buckets])
        periods = []
        while cursor < end:
            periods.append(
                PeriodMetrics(cursor, _shift(cursor, step), _metrics(buckets[cursor], detections))
            )
            cursor = _shift(cursor, step)
        current, previous = periods[-1].metrics, periods[-2].metrics
        totals = _metrics(product_rows, detections)
        comparisons = tuple(
            _comparison(metric, current, previous, policy.min_denominator)
            for metric in ("negative_rating", *(category.value for category in ComplaintCategory))
        )
        products.append(
            ProductIntelligence(
                product_id,
                totals,
                period,
                tuple(periods),
                comparisons,
                sum(row.reviewed_at is not None and row.reviewed_at >= end for row in product_rows),
            )
        )
        pending = {comparison.metric for comparison in comparisons}
        # Replay fixed adjacent calendar periods, retaining the newest qualifying
        # window. Old evidence remains dated history even if latest support is zero.
        for index in range(len(periods) - 1, 0, -1):
            current_period, previous_period = periods[index], periods[index - 1]
            for metric in sorted(pending):
                comparison = _comparison(
                    metric, current_period.metrics, previous_period.metrics, policy.min_denominator
                )
                candidate = _candidate(
                    product_id,
                    comparison,
                    previous_period,
                    current_period,
                    dated,
                    detections,
                    policy,
                )
                if candidate is not None:
                    queue.append(candidate)
                    pending.remove(metric)
    queue.sort(
        key=lambda item: (
            item.reason != "rising_share",
            -item.window_end.timestamp(),
            -(item.comparison.share_change or 0),
            -len(item.supporting_review_ids),
            item.product_id,
            item.metric,
        )
    )
    return IntelligenceResult(
        as_of, policy, _metrics(rows, detections), tuple(products), tuple(queue), duplicates, future
    )
