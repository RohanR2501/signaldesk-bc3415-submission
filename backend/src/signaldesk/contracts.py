"""Canonical 1.1 business schemas and preserved 1.0 offline experiment contracts.

Offsets refer to Python Unicode code points in the original review text.
Predicted complaints and alerts are customer-reported signals, never findings.
"""

from enum import StrEnum
from math import isclose
from typing import Annotated, Generic, Literal, Self, TypeVar

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=200)]
UnitInterval = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.1"] = "1.1"
    status: Literal["ok"] = "ok"
    service: Literal["signaldesk"] = "signaldesk"
    version: str
    api_version: Literal["v1"] = "v1"
    phase: Literal[4] = 4
    model_status: Literal["not_checked"] = "not_checked"


class ErrorDetail(BaseModel):
    location: list[str | int]
    type: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)


class ErrorResponse(Contract):
    error: ErrorBody
    request_id: str


class CustomerReview(Contract):
    review_id: Identifier
    product_id: Identifier
    source: Identifier
    source_revision: Identifier
    text: Annotated[str, Field(min_length=1, max_length=50000)]
    title: str | None = None
    rating: Annotated[float, Field(ge=1, le=5, allow_inf_nan=False)] | None = None
    reviewed_at: AwareDatetime | None = None
    ingested_at: AwareDatetime
    language: str | None = None


class SentimentLabel(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ComplaintLabel(StrEnum):
    """Legacy preliminary vocabulary; retained for frozen experiment compatibility only."""

    SUSPECTED_PRODUCT_DEFECT = "suspected_product_defect"
    WRONG_ITEM = "wrong_item"
    DELIVERY_PROBLEM = "delivery_problem"
    PACKAGING_DAMAGE = "packaging_damage"
    CUSTOMER_SERVICE = "customer_service"


class EvidenceSpan(BaseModel):
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]
    text: Annotated[str, Field(min_length=1)]
    method: Literal["feature_attribution", "extractive_model", "human_annotation"]

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        if self.end <= self.start or self.end - self.start != len(self.text):
            raise ValueError("Evidence length must match the half-open span [start, end)")
        return self


class PredictionMetadata(Contract):
    review_id: Identifier
    model_version: Identifier
    preprocessing_version: Identifier
    predicted_at: AwareDatetime
    requires_human_review: bool
    escalation_reasons: list[str] = Field(default_factory=list)


class SentimentPrediction(PredictionMetadata):
    label: SentimentLabel
    probabilities: dict[SentimentLabel, UnitInterval]
    calibration: Literal["uncalibrated", "validated_calibration"]
    evidence: list[EvidenceSpan] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_distribution(self) -> Self:
        if set(self.probabilities) != set(SentimentLabel):
            raise ValueError("All three sentiment probabilities are required")
        if not isclose(sum(self.probabilities.values()), 1, abs_tol=1e-6):
            raise ValueError("Sentiment probabilities must sum to one")
        return self


class ComplaintScore(BaseModel):
    label: ComplaintLabel
    probability: UnitInterval
    threshold: UnitInterval
    selected: bool
    evidence: list[EvidenceSpan] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if self.selected != (self.probability >= self.threshold):
            raise ValueError("Selection must agree with the recorded threshold")
        return self


class ComplaintPrediction(PredictionMetadata):
    taxonomy_version: Identifier
    calibration: Literal["uncalibrated", "validated_calibration"]
    scores: list[ComplaintScore]

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        labels = [score.label for score in self.scores]
        if len(labels) != len(set(labels)) or set(labels) != set(ComplaintLabel):
            raise ValueError("Each complaint category must have exactly one score")
        return self


class ProductIncidentAlert(Contract):
    alert_id: Identifier
    product_id: Identifier
    category: ComplaintLabel
    window_start: AwareDatetime
    window_end: AwareDatetime
    eligible_review_count: Annotated[int, Field(gt=0)]
    complaint_review_count: Annotated[int, Field(ge=0)]
    supporting_review_ids: list[Identifier] = Field(min_length=1)
    rule_version: Identifier
    model_version: Identifier
    generated_at: AwareDatetime
    signal_type: Literal["customer_reported_suspected_issue"] = "customer_reported_suspected_issue"
    status: Literal["open", "under_review", "closed"] = "open"

    @model_validator(mode="after")
    def validate_window_and_counts(self) -> Self:
        if self.window_end <= self.window_start:
            raise ValueError("Alert window must have positive duration")
        if self.complaint_review_count > self.eligible_review_count:
            raise ValueError("Complaint count cannot exceed eligible review count")
        if len(set(self.supporting_review_ids)) != len(self.supporting_review_ids):
            raise ValueError("Supporting review identifiers must be unique")
        if len(self.supporting_review_ids) > self.complaint_review_count:
            raise ValueError("Evidence subset cannot exceed complaint count")
        return self


class EvaluationMetric(BaseModel):
    name: Identifier
    value: Annotated[float, Field(allow_inf_nan=False)]
    averaging: Literal["macro", "micro", "weighted", "per_class", "none"]
    label: str | None = None
    unit: Literal["fraction", "milliseconds", "count"]


class ModelEvaluationResult(Contract):
    evaluation_id: Identifier
    task: Literal["sentiment", "complaint_multilabel"]
    model_version: Identifier
    dataset_revision: Identifier
    split_manifest_hash: Identifier
    annotation_version: Identifier
    code_revision: Identifier
    environment_lock_hash: Identifier
    evaluated_at: AwareDatetime
    sample_count: Annotated[int, Field(gt=0)]
    split: Literal["validation", "test"]
    metrics: list[EvaluationMetric] = Field(min_length=1)
    confusion_matrix_artifact: str | None = None
    limitations: list[str] = Field(min_length=1)


class HumanReviewDecision(Contract):
    decision_id: Identifier
    alert_id: Identifier
    reviewer_id: Identifier
    decided_at: AwareDatetime
    decision: Literal[
        "investigate", "dismiss_signal", "request_more_evidence", "close_investigation"
    ]
    rationale: Annotated[str, Field(min_length=1, max_length=5000)]
    supporting_review_ids: list[Identifier] = Field(default_factory=list)
    supersedes_decision_id: Identifier | None = None


# New business responses use 1.1. Existing 1.0 experiment and health payloads stay
# readable; legacy probability-required complaint/alert types are not served.
class ApiPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class ApiResponse(ApiPayload):
    schema_version: Literal["1.1"] = "1.1"


class ComplaintCategory(StrEnum):
    PRODUCT_DEFECT = "product_defect"
    ARRIVAL_DAMAGE = "arrival_damage"
    WRONG_ITEM = "wrong_item"
    DELIVERY_ISSUE = "delivery_issue"
    SERVICE_RETURN_ISSUE = "service_return_issue"


class DetectionState(StrEnum):
    DETECTED = "detected"
    UNKNOWN = "unknown"
    ABSTAINED = "abstained"
    UNAVAILABLE = "unavailable"


Count = Annotated[int, Field(ge=0)]
MetricName = Literal[
    "negative_rating",
    "product_defect",
    "arrival_damage",
    "wrong_item",
    "delivery_issue",
    "service_return_issue",
]


class ReviewExcerpt(ApiPayload):
    review_id: Identifier
    reviewed_at: AwareDatetime | None = None
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]
    text: Annotated[str, Field(min_length=1)]
    method: Literal["rule_match", "source_excerpt"]
    rule_id: str | None = None
    resolution: Literal["not_established", "reported_resolved"] = "not_established"
    resolution_excerpt: str | None = None
    resolution_start: int | None = None
    resolution_end: int | None = None
    is_model_attribution: Literal[False] = False

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.end - self.start != len(self.text):
            raise ValueError("Excerpt offsets must match exact Unicode text length")
        if self.method == "rule_match" and not self.rule_id:
            raise ValueError("A rule match must identify its rule")
        parts = (self.resolution_excerpt, self.resolution_start, self.resolution_end)
        if any(value is not None for value in parts):
            if any(value is None for value in parts):
                raise ValueError("Resolution text and both offsets must be supplied together")
            assert self.resolution_start is not None and self.resolution_end is not None
            assert self.resolution_excerpt is not None
            if self.resolution_start < 0 or self.resolution_end - self.resolution_start != len(
                self.resolution_excerpt
            ):
                raise ValueError("Invalid resolution span")
        return self


class ComplaintDetection(ApiPayload):
    category: ComplaintCategory
    state: DetectionState
    method: Literal["explicit_rules"] = "explicit_rules"
    rule_version: str
    taxonomy_version: str
    evidence: list[ReviewExcerpt] = Field(default_factory=list)
    reason: str
    reliability_warning: str
    learned_model_state: Literal["not_used", "unavailable"] = "not_used"

    @model_validator(mode="after")
    def validate_detection(self) -> Self:
        if (self.state == DetectionState.DETECTED) != bool(self.evidence):
            raise ValueError("Rule detection requires evidence; other states carry no matches")
        return self


class ModelAttempt(ApiPayload):
    model: str
    state: Literal["used", "unavailable"]
    reason: str | None = None


class LiveSentiment(ApiPayload):
    state: DetectionState
    label: SentimentLabel | None = None
    model_version: str | None = None
    device: Literal["cuda:0", "cpu"] | None = None
    manifest_sha256: str | None = None
    artifact_sha256: dict[str, str] = Field(default_factory=dict)
    preprocessing: str | None = None
    max_tokens: int | None = None
    fallback_reason: str | None = None
    attempts: list[ModelAttempt] = Field(default_factory=list)
    input_provenance: Literal["submitted_review_text_only"] = "submitted_review_text_only"
    score_state: Literal["not_returned_by_saved_label_interface"] = (
        "not_returned_by_saved_label_interface"
    )
    calibration: Literal["uncalibrated"] = "uncalibrated"

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.state == DetectionState.DETECTED:
            if (
                self.label is None
                or not self.model_version
                or not self.device
                or not self.manifest_sha256
            ):
                raise ValueError(
                    "A sentiment prediction requires its actual label and model identity"
                )
        elif self.label is not None or self.model_version is not None:
            raise ValueError("An unresolved sentiment must not supply a prediction")
        return self


class AnalyzeRequest(ApiResponse):
    text: Annotated[str, Field(min_length=1, max_length=50000)]

    @field_validator("text")
    @classmethod
    def validate_text(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("Review text cannot be blank")
        return text  # Preserve original code-point offsets, including whitespace.


class AnalyzeResponse(ApiResponse):
    review_id: Identifier
    analyzed_at: AwareDatetime
    status: Literal["complete", "partial"]
    sentiment: LiveSentiment
    complaints: list[ComplaintDetection]
    warnings: list[str]

    @model_validator(mode="after")
    def validate_categories(self) -> Self:
        categories = [item.category for item in self.complaints]
        if len(categories) != 5 or set(categories) != set(ComplaintCategory):
            raise ValueError("Exactly one result per complaint category is required")
        if (self.status == "complete") != (self.sentiment.state == DetectionState.DETECTED):
            raise ValueError("Analysis status must reflect sentiment availability")
        return self


class HistoricalMetrics(ApiPayload):
    unique_reviews: Count
    rated_reviews: Count
    negative_rating_reviews: Count
    negative_rating_share: UnitInterval | None
    rule_scanned_reviews: Count
    any_complaint_reviews: Count
    complaint_reviews: dict[ComplaintCategory, Count]
    complaint_shares: dict[ComplaintCategory, UnitInterval | None]
    unknown_no_evidence_reviews: Count
    text_unavailable_reasons: dict[str, Count]
    missing_timestamp_reviews: Count
    sentiment_model_state: Literal["unavailable"]
    complaint_model_states: list[str]
    negative_provenance: str
    complaint_provenance: str
    review_provenance: str

    @model_validator(mode="after")
    def validate_coverage(self) -> Self:
        if set(self.complaint_reviews) != set(ComplaintCategory) or set(
            self.complaint_shares
        ) != set(ComplaintCategory):
            raise ValueError("All five complaint category metrics are required")
        if not 0 <= self.negative_rating_reviews <= self.rated_reviews <= self.unique_reviews:
            raise ValueError("Invalid rating denominator")
        if not self.any_complaint_reviews <= self.rule_scanned_reviews <= self.unique_reviews:
            raise ValueError("Invalid scan denominator")
        if (
            self.unknown_no_evidence_reviews + self.any_complaint_reviews
            != self.rule_scanned_reviews
        ):
            raise ValueError("Rule coverage must retain unknown reviews")
        if (
            sum(self.text_unavailable_reasons.values()) + self.rule_scanned_reviews
            != self.unique_reviews
        ):
            raise ValueError("Scan and unavailable coverage must sum to total reviews")
        if (self.negative_rating_share is None) != (self.rated_reviews == 0):
            raise ValueError("Missing denominators require null shares")
        for category, count in self.complaint_reviews.items():
            if count > self.any_complaint_reviews:
                raise ValueError("Category support exceeds unique complaint reviews")
            if (self.complaint_shares[category] is None) != (self.rule_scanned_reviews == 0):
                raise ValueError("Missing scan denominators require null shares")
        return self


class MetricComparison(ApiPayload):
    metric: MetricName
    state: Literal["comparable", "insufficient_support"]
    current_count: Count
    current_denominator: Count
    previous_count: Count
    previous_denominator: Count
    share_change: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)] | None


class PriorityRules(ApiPayload):
    min_denominator: int
    min_reports: int
    negative_increase: float
    complaint_increase: float
    sustained_negative_share: float
    sustained_complaint_share: float
    min_sustained_reports: int
    evidence_limit: int


class HistoricalContext(ApiPayload):
    snapshot_id: str
    as_of: AwareDatetime
    first_review_at: AwareDatetime | None
    last_review_at: AwareDatetime | None
    dataset_revision: str
    manifest_sha256: str
    investigation_sha256: str
    rules_sha256: str
    reservation_sha256: dict[str, str]
    source_revisions: list[tuple[str, str]]
    engine_version: str
    taxonomy_version: str
    rule_version: str
    mode: Literal["historical_snapshot"] = "historical_snapshot"
    warnings: list[str]


class ExecutiveSummary(ApiResponse):
    context: HistoricalContext
    totals: HistoricalMetrics
    product_count: Count
    candidate_count: Count
    candidate_product_count: Count
    policy: PriorityRules


class ProductSummary(ApiPayload):
    product_id: Identifier
    totals: HistoricalMetrics
    granularity: Literal["month", "quarter"]
    comparisons: list[MetricComparison]
    incomplete_period_reviews: Count


class ProductDetail(ApiResponse):
    context: HistoricalContext
    product: ProductSummary
    candidate_count: Count


class TimelinePeriod(ApiPayload):
    start: AwareDatetime
    end: AwareDatetime
    metrics: HistoricalMetrics
    rating_support_sufficient: bool
    rule_support_sufficient: bool


class ProductTimeline(ApiResponse):
    context: HistoricalContext
    product_id: Identifier
    granularity: Literal["month", "quarter"]
    min_denominator: int
    incomplete_period_reviews: Count
    periods: list[TimelinePeriod]


class InvestigationSignal(ApiPayload):
    signal_id: Identifier
    product_id: Identifier
    metric: MetricName
    reason: Literal["rising_share", "sustained_burden"]
    comparison: MetricComparison
    window_start: AwareDatetime
    window_end: AwareDatetime
    window_report_count: Count
    window_denominator: Count
    evidence_unavailable_reviews: Count
    status: Literal["candidate_requires_human_investigation"]
    signal_type: Literal["historical_customer_reported_or_rating_proxy"]
    explanation: str


class InvestigationDetail(ApiResponse):
    context: HistoricalContext
    signal: InvestigationSignal
    supporting_review_ids: list[Identifier]
    evidence: list[ReviewExcerpt]
    policy: PriorityRules


class ReviewEvidenceResponse(ApiResponse):
    context: HistoricalContext
    review_id: Identifier
    product_id: Identifier
    reviewed_at: AwareDatetime | None
    rating: float | None
    text_state: Literal["available", "protected_family", "reserved_annotation", "unavailable"]
    text: str | None
    complaints: list[ComplaintDetection]


Item = TypeVar("Item")


class Page(ApiResponse, Generic[Item]):
    context: HistoricalContext
    items: list[Item]
    total: Count
    limit: Annotated[int, Field(ge=1, le=100)]
    next_cursor: str | None
