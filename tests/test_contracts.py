"""Synthetic unit-test fixtures validate interfaces, not measured ML behavior."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from signaldesk.contracts import (
    ComplaintLabel,
    ComplaintPrediction,
    ComplaintScore,
    CustomerReview,
    EvidenceSpan,
    ProductIncidentAlert,
    SentimentLabel,
    SentimentPrediction,
)

NOW = datetime(2026, 10, 24, tzinfo=UTC)


def test_review_requires_timezone_and_known_schema_version() -> None:
    review = CustomerReview(
        review_id="test:r1",
        product_id="test:p1",
        source="unit-test",
        source_revision="fixture-v1",
        text="Synthetic fixture text",
        ingested_at=NOW,
    )
    with pytest.raises(ValidationError):
        CustomerReview.model_validate({**review.model_dump(), "schema_version": "2.0"})
    with pytest.raises(ValidationError):
        CustomerReview.model_validate(
            {**review.model_dump(), "ingested_at": NOW.replace(tzinfo=None)}
        )
    with pytest.raises(ValidationError):
        CustomerReview.model_validate({**review.model_dump(), "unexpected": True})


def test_sentiment_requires_complete_normalized_probabilities() -> None:
    prediction = SentimentPrediction(
        review_id="test:r1",
        model_version="fixture-only",
        preprocessing_version="fixture-only",
        predicted_at=NOW,
        requires_human_review=True,
        label=SentimentLabel.POSITIVE,
        probabilities={
            SentimentLabel.POSITIVE: 0.6,
            SentimentLabel.NEUTRAL: 0.3,
            SentimentLabel.NEGATIVE: 0.1,
        },
        calibration="uncalibrated",
    )
    with pytest.raises(ValidationError):
        SentimentPrediction.model_validate(
            {**prediction.model_dump(), "probabilities": {"positive": 1}}
        )
    with pytest.raises(ValidationError):
        SentimentPrediction.model_validate(
            {**prediction.model_dump(), "probabilities": {label: 0.9 for label in SentimentLabel}}
        )


def test_complaints_allow_multiple_selected_labels_without_normalizing_scores() -> None:
    scores = [
        ComplaintScore(label=label, probability=0.8, threshold=0.6, selected=True)
        for label in ComplaintLabel
    ]
    prediction = ComplaintPrediction(
        review_id="test:r1",
        model_version="fixture-only",
        preprocessing_version="fixture-only",
        predicted_at=NOW,
        requires_human_review=True,
        taxonomy_version="fixture-only",
        calibration="uncalibrated",
        scores=scores,
    )
    assert sum(score.selected for score in prediction.scores) == len(ComplaintLabel)
    with pytest.raises(ValidationError):
        ComplaintPrediction.model_validate({**prediction.model_dump(), "scores": scores[:-1]})
    with pytest.raises(ValidationError):
        ComplaintScore(
            label=ComplaintLabel.WRONG_ITEM, probability=0.2, threshold=0.6, selected=True
        )


def test_evidence_offsets_use_code_points() -> None:
    span = EvidenceSpan(start=2, end=4, text="📦!", method="human_annotation")
    assert span.end - span.start == 2
    with pytest.raises(ValidationError):
        EvidenceSpan(start=2, end=5, text="📦!", method="human_annotation")


def test_alert_counts_window_and_suspected_semantics() -> None:
    alert = ProductIncidentAlert(
        alert_id="test:a1",
        product_id="test:p1",
        category=ComplaintLabel.WRONG_ITEM,
        window_start=NOW - timedelta(days=1),
        window_end=NOW,
        eligible_review_count=3,
        complaint_review_count=1,
        supporting_review_ids=["test:r1"],
        rule_version="fixture-only",
        model_version="fixture-only",
        generated_at=NOW,
    )
    assert alert.signal_type == "customer_reported_suspected_issue"
    with pytest.raises(ValidationError):
        ProductIncidentAlert.model_validate({**alert.model_dump(), "complaint_review_count": 4})
    with pytest.raises(ValidationError):
        ProductIncidentAlert.model_validate({**alert.model_dump(), "window_start": NOW})
    with pytest.raises(ValidationError):
        ProductIncidentAlert.model_validate(
            {**alert.model_dump(), "supporting_review_ids": ["test:r1", "test:r1"]}
        )
