"""Synthetic regression cases for the versioned coverage correction, not recall estimates."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from signaldesk.config import Settings
from signaldesk.main import create_app
from signaldesk_ml.complaint_rules import RULE_VERSION, ComplaintCategory, detect_complaints


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("The replacement motor was faulty.", ComplaintCategory.PRODUCT_DEFECT),
        ("This pump is non-functional.", ComplaintCategory.PRODUCT_DEFECT),
        ("These pods don’t filter coffee.", ComplaintCategory.PRODUCT_DEFECT),
        ("Water would not flow through the filter.", ComplaintCategory.PRODUCT_DEFECT),
        ("The interior paint chips off when wet.", ComplaintCategory.PRODUCT_DEFECT),
        ("The coating is peeling during use.", ComplaintCategory.PRODUCT_DEFECT),
        ("The housing was scuffed upon unboxing.", ComplaintCategory.ARRIVAL_DAMAGE),
        ("We received a scratched unit.", ComplaintCategory.ARRIVAL_DAMAGE),
        ("It came with bent steel bars.", ComplaintCategory.ARRIVAL_DAMAGE),
    ],
)
def test_explicit_failure_and_receipt_damage(text: str, category: ComplaintCategory) -> None:
    result = detect_complaints(text)
    assert result.categories == (category,)
    for item in result.evidence:
        assert text[item.span.start : item.span.end] == item.span.text


@pytest.mark.parametrize(
    "text",
    [
        "My old dryer was defective; this replacement works.",
        "My previous pump was faulty; this replacement works.",
        "The item doesn't fit my model.",
        "The paint did not chip.",
        "The paint did not chip off when wet.",
        "The surface became scuffed after six months.",
        "I returned it.",
        "The exterior paint chipped off.",
        "I was worried the motor might be faulty.",
        "I was worried water would not flow through it.",
        "I was afraid these filters would not filter water.",
        "The interior paint may chip off when wet.",
        "The interior paint will chip off when wet.",
        "A faulty design, but the unit operates as described.",
        "The door is misaligned and may be missing a screw.",
        "The package contained one filter; two more were missing.",
    ],
)
def test_unsupported_negated_and_ambiguous_reports_stay_unknown(text: str) -> None:
    result = detect_complaints(text)
    assert not result.categories and not result.evidence
    assert result.status == "unknown_no_detected_evidence"


def test_unicode_resolution_and_version_provenance() -> None:
    text = "😀 The motor was faulty. I repaired it and now it works."
    result = detect_complaints(text)
    assert result.rule_version == RULE_VERSION == "complaint-rules-v2"
    assert result.taxonomy_version == "complaints-v1"
    assert result.independent_validation == "not_performed"
    evidence = result.evidence[0]
    assert evidence.resolution == "reported_resolved"
    resolution = evidence.resolution_span
    assert resolution is not None
    assert text[resolution.start : resolution.end] == resolution.text
    assert text[evidence.span.start : evidence.span.end] == evidence.span.text


def test_api_serves_corrected_rules_without_loading_sentiment(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        environment="test",
        transformer_directory=tmp_path / "absent-transformer",
        baseline_directory=tmp_path / "absent-baseline",
    )
    text = "😀 The coating inside chips off when wet."
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/reviews/analyze", json={"text": text})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "partial" and body["sentiment"]["state"] == "unavailable"
        assert len(body["complaints"]) == 5
        for complaint in body["complaints"]:
            assert complaint["rule_version"] == RULE_VERSION
            assert "probability" not in complaint
            for evidence in complaint["evidence"]:
                assert text[evidence["start"] : evidence["end"]] == evidence["text"]
        assert body["complaints"][0]["state"] == "detected"
