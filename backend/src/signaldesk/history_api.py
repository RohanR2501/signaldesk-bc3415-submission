"""Read-only historical routes backed by one lazy, provenance-labelled snapshot per process."""

import base64
import binascii
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, timedelta
from threading import Lock
from typing import Annotated, TypeVar

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from pydantic import AwareDatetime

from signaldesk.analyzer import complaint_results
from signaldesk.contracts import (
    ComplaintCategory,
    ExecutiveSummary,
    HistoricalContext,
    HistoricalMetrics,
    InvestigationDetail,
    InvestigationSignal,
    MetricName,
    Page,
    PriorityRules,
    ProductDetail,
    ProductSummary,
    ProductTimeline,
    ReviewEvidenceResponse,
    ReviewExcerpt,
    TimelinePeriod,
)
from signaldesk.investigation import InvestigationSource, load_investigation
from signaldesk.product_intelligence import HistoricalReview, IntelligenceResult, build_intelligence
from signaldesk_ml.complaint_rules import RULE_VERSION, TAXONOMY_VERSION


@dataclass(frozen=True)
class HistoricalSnapshot:
    source: InvestigationSource
    result: IntelligenceResult
    context: HistoricalContext
    products: dict[str, ProductSummary]
    signals: dict[str, InvestigationDetail]
    reviews: dict[str, HistoricalReview]


class HistoricalStore:
    """Cache aggregation, not a duplicate database; loading never occurs at import/startup."""

    def __init__(self) -> None:
        self._snapshot: HistoricalSnapshot | None = None
        self._lock = Lock()

    def get(self) -> HistoricalSnapshot:
        with self._lock:
            if self._snapshot is None:
                self._snapshot = self._load()
            return self._snapshot

    def _load(self) -> HistoricalSnapshot:
        source = load_investigation()
        dates = [row.reviewed_at for row in source.reviews if row.reviewed_at is not None]
        # Observation ends in 2023; never anchor old data to today's calendar.
        if not dates:
            raise ValueError("No observation dates available for a historical snapshot")
        last = max(dates)
        as_of = last.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
            days=1
        )
        result = build_intelligence(source.reviews, as_of=as_of)
        identity = (
            f"{source.investigation_sha256}:{source.rules_sha256}:"
            f"{result.engine_version}:{as_of.isoformat()}"
        )
        context = HistoricalContext(
            snapshot_id=hashlib.sha256(identity.encode()).hexdigest(),
            as_of=as_of,
            first_review_at=min(dates) if dates else None,
            last_review_at=last,
            dataset_revision=source.dataset_revision,
            manifest_sha256=source.manifest_sha256,
            investigation_sha256=source.investigation_sha256,
            rules_sha256=source.rules_sha256,
            reservation_sha256=source.reservation_sha256,
            source_revisions=list(source.source_revisions),
            engine_version=result.engine_version,
            taxonomy_version=TAXONOMY_VERSION,
            rule_version=RULE_VERSION,
            warnings=list(result.limitations),
        )
        products = {
            item.product_id: ProductSummary.model_validate(item) for item in result.products
        }
        reviews = {row.review_id: row for row in source.reviews}
        signals = {}
        for candidate in result.queue:
            key = (
                f"{identity}:{candidate.product_id}:{candidate.metric}:"
                f"{candidate.window_end.isoformat()}"
            )
            signal_id = hashlib.sha256(key.encode()).hexdigest()[:24]
            data = asdict(candidate)
            evidence, ids = data.pop("evidence"), data.pop("supporting_review_ids")
            if candidate.reason == "rising_share":
                change = candidate.comparison.share_change
                explanation = (
                    f"Historical share increased by {100 * (change or 0):.1f} percentage points; "
                    f"current support {candidate.comparison.current_count}/"
                    f"{candidate.comparison.current_denominator}, previous "
                    f"{candidate.comparison.previous_count}/{candidate.comparison.previous_denominator}."
                )
            else:
                explanation = (
                    f"Sustained historical burden: {candidate.window_report_count}/"
                    f"{candidate.window_denominator} reviews across two supported periods."
                )
            signals[signal_id] = InvestigationDetail(
                context=context,
                signal=InvestigationSignal(signal_id=signal_id, explanation=explanation, **data),
                supporting_review_ids=list(ids),
                evidence=[
                    ReviewExcerpt.model_validate(
                        {
                            **item,
                            "reviewed_at": reviews[item["review_id"]].reviewed_at,
                        }
                    )
                    for item in evidence
                ],
                policy=PriorityRules.model_validate(result.policy),
            )
        # Review response construction (and rule matching) is deferred to explicit retrieval.
        return HistoricalSnapshot(source, result, context, products, signals, reviews)


def _snapshot(request: Request) -> HistoricalSnapshot:
    store: HistoricalStore = request.app.state.history
    try:
        return store.get()
    except (OSError, ValueError, KeyError, ImportError):
        raise HTTPException(status_code=503) from None


def _invalid_query(name: str) -> None:
    raise RequestValidationError([{"loc": ("query", name), "type": "value_error"}])


T = TypeVar("T")


def _page(
    items: list[T], context: HistoricalContext, limit: int, cursor: str | None, key: str
) -> Page[T]:
    """Bind offset cursors to endpoint filters and the immutable snapshot."""
    fingerprint = hashlib.sha256(f"{context.snapshot_id}:{key}".encode()).hexdigest()[:24]
    offset = 0
    if cursor is not None:
        try:
            decoded = json.loads(base64.b64decode(cursor, altchars=b"-_", validate=True))
            if not isinstance(decoded, dict) or decoded.get("key") != fingerprint:
                raise ValueError
            offset = decoded["offset"]
            if type(offset) is not int or not 0 <= offset <= len(items):
                raise ValueError
        except (ValueError, KeyError, TypeError, binascii.Error, UnicodeError):
            _invalid_query("cursor")
    end = offset + limit
    next_cursor = None
    if end < len(items):
        next_cursor = base64.urlsafe_b64encode(
            json.dumps({"key": fingerprint, "offset": end}).encode()
        ).decode()
    return Page[T](
        context=context,
        items=items[offset:end],
        total=len(items),
        limit=limit,
        next_cursor=next_cursor,
    )


router = APIRouter(prefix="/api/v1", tags=["historical intelligence"])
Limit = Annotated[int, Query(ge=1, le=100)]
Cursor = Annotated[str | None, Query(max_length=1024)]


@router.get("/summary", response_model=ExecutiveSummary)
def executive_summary(request: Request) -> ExecutiveSummary:
    snapshot = _snapshot(request)
    result = snapshot.result
    return ExecutiveSummary(
        context=snapshot.context,
        totals=HistoricalMetrics.model_validate(result.totals),
        product_count=len(result.products),
        candidate_count=len(result.queue),
        candidate_product_count=len({item.product_id for item in result.queue}),
        policy=PriorityRules.model_validate(result.policy),
    )


@router.get("/products", response_model=Page[ProductSummary])
def list_products(
    request: Request,
    limit: Limit = 25,
    cursor: Cursor = None,
    q: Annotated[str | None, Query(max_length=200)] = None,
    category: ComplaintCategory | None = None,
    min_reviews: Annotated[int, Query(ge=0)] = 0,
    has_signals: bool | None = None,
) -> Page[ProductSummary]:
    snapshot = _snapshot(request)
    signal_products = {item.signal.product_id for item in snapshot.signals.values()}
    items = [
        item
        for item in snapshot.products.values()
        if (q is None or q.casefold() in item.product_id.casefold())
        and item.totals.unique_reviews >= min_reviews
        and (category is None or item.totals.complaint_reviews[category] > 0)
        and (has_signals is None or (item.product_id in signal_products) == has_signals)
    ]
    items.sort(key=lambda item: item.product_id)
    key = json.dumps(["products", q, category, min_reviews, has_signals])
    return _page(items, snapshot.context, limit, cursor, key)


@router.get("/products/{product_id}", response_model=ProductDetail)
def product_detail(request: Request, product_id: str) -> ProductDetail:
    snapshot = _snapshot(request)
    if product_id not in snapshot.products:
        raise HTTPException(status_code=404)
    return ProductDetail(
        context=snapshot.context,
        product=snapshot.products[product_id],
        candidate_count=sum(
            item.signal.product_id == product_id for item in snapshot.signals.values()
        ),
    )


@router.get("/products/{product_id}/timeline", response_model=ProductTimeline)
def product_timeline(
    request: Request,
    product_id: str,
    from_date: Annotated[AwareDatetime | None, Query(alias="from")] = None,
    to_date: Annotated[AwareDatetime | None, Query(alias="to")] = None,
) -> ProductTimeline:
    if from_date is not None and to_date is not None and from_date >= to_date:
        _invalid_query("to")
    snapshot = _snapshot(request)
    product = next(
        (item for item in snapshot.result.products if item.product_id == product_id), None
    )
    if product is None:
        raise HTTPException(status_code=404)
    minimum = snapshot.result.policy.min_denominator
    periods = [
        TimelinePeriod(
            start=item.start,
            end=item.end,
            metrics=HistoricalMetrics.model_validate(item.metrics),
            rating_support_sufficient=item.metrics.rated_reviews >= minimum,
            rule_support_sufficient=item.metrics.rule_scanned_reviews >= minimum,
        )
        for item in product.periods
        if (from_date is None or item.end > from_date) and (to_date is None or item.start < to_date)
    ]
    return ProductTimeline(
        context=snapshot.context,
        product_id=product_id,
        granularity=product.granularity,
        min_denominator=minimum,
        incomplete_period_reviews=product.incomplete_period_reviews,
        periods=periods,
    )


@router.get("/investigations", response_model=Page[InvestigationSignal])
def list_investigations(
    request: Request,
    limit: Limit = 25,
    cursor: Cursor = None,
    product_id: str | None = None,
    metric: MetricName | None = None,
    reason: Annotated[str | None, Query(pattern="^(rising_share|sustained_burden)$")] = None,
) -> Page[InvestigationSignal]:
    snapshot = _snapshot(request)
    items = [
        detail.signal
        for detail in snapshot.signals.values()
        if (product_id is None or detail.signal.product_id == product_id)
        and (metric is None or detail.signal.metric == metric)
        and (reason is None or detail.signal.reason == reason)
    ]
    return _page(
        items,
        snapshot.context,
        limit,
        cursor,
        json.dumps(["investigations", product_id, metric, reason]),
    )


@router.get("/investigations/{signal_id}", response_model=InvestigationDetail)
def investigation_detail(request: Request, signal_id: str) -> InvestigationDetail:
    snapshot = _snapshot(request)
    if signal_id not in snapshot.signals:
        raise HTTPException(status_code=404)
    return snapshot.signals[signal_id]


@router.get("/reviews/{review_id}", response_model=ReviewEvidenceResponse)
def review_evidence(request: Request, review_id: str) -> ReviewEvidenceResponse:
    snapshot = _snapshot(request)
    row = snapshot.reviews.get(review_id)
    if row is None:
        raise HTTPException(status_code=404)
    return ReviewEvidenceResponse(
        context=snapshot.context,
        review_id=row.review_id,
        product_id=row.product_id,
        reviewed_at=row.reviewed_at,
        rating=row.rating,
        text_state=row.text_state,
        text=row.text,
        complaints=complaint_results(row.text, row.review_id),
    )
