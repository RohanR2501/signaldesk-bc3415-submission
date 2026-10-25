"""Small offline complaint reference: explicit patterns, exact spans, no probabilities.

Customer-reported occurrences can be resolved. No match means unknown/no detected
evidence, never verified absence. This module imports no training or model runtime.
"""

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

TAXONOMY_VERSION = "complaints-v1"
RULE_VERSION = "complaint-rules-v2"
SuppressionReason = Literal[
    "negated",
    "hypothetical",
    "unrelated_old_product",
    "non_delivery_object",
    "packaging_only",
    "cosmetic_only",
]


class ComplaintCategory(StrEnum):
    PRODUCT_DEFECT = "product_defect"
    ARRIVAL_DAMAGE = "arrival_damage"
    WRONG_ITEM = "wrong_item"
    DELIVERY_ISSUE = "delivery_issue"
    SERVICE_RETURN_ISSUE = "service_return_issue"


@dataclass(frozen=True, slots=True)
class TextSpan:
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class ComplaintEvidence:
    category: ComplaintCategory
    span: TextSpan
    rule_id: str
    resolution: Literal["not_established", "reported_resolved"]
    resolution_span: TextSpan | None = None


@dataclass(frozen=True, slots=True)
class SuppressedEvidence:
    category: ComplaintCategory
    span: TextSpan
    rule_id: str
    reason: SuppressionReason


@dataclass(frozen=True, slots=True)
class ComplaintDetection:
    categories: tuple[ComplaintCategory, ...]
    evidence: tuple[ComplaintEvidence, ...]
    suppressed: tuple[SuppressedEvidence, ...]
    status: Literal["customer_reported_evidence", "unknown_no_detected_evidence"]
    taxonomy_version: str = TAXONOMY_VERSION
    rule_version: str = RULE_VERSION
    independent_validation: Literal["not_performed"] = "not_performed"


@dataclass(frozen=True, slots=True)
class Rule:
    identifier: str
    category: ComplaintCategory
    pattern: re.Pattern[str]
    intrinsic_negation: bool = False


def _rule(
    identifier: str, category: ComplaintCategory, pattern: str, intrinsic_negation: bool = False
) -> Rule:
    return Rule(identifier, category, re.compile(pattern, re.I), intrinsic_negation)


RULES = (
    _rule(
        "defect.function",
        ComplaintCategory.PRODUCT_DEFECT,
        r"\b(?P<issue>(?:stopped|quit) (?:working|functioning|cooling|heating)|"
        "(?:does not|doesn['’]t|did not|didn['’]t|will not|won['’]t"
        "|would not|wouldn['’]t|cannot|can['’]t) "
        r"(?:work|function|turn on|start|cool|heat)|no longer (?:works|functions)|"
        r"(?:motor|compressor|bearing|unit|machine) (?:died|failed))\b",
        True,
    ),
    _rule(
        "defect.breakage",
        ComplaintCategory.PRODUCT_DEFECT,
        r"\b(?P<issue>broke|broken|defective|malfunction(?:ed|ing)?)\b",
    ),
    _rule(
        "defect.leak",
        ComplaintCategory.PRODUCT_DEFECT,
        "\\b(?P<issue>(?:it|unit|product|machine|washer|hose|pump) ("
        "?:is )?leak(?:s|ing)|water leak(?:ed|s|ing))\\b",
    ),
    _rule(
        "defect.component",
        ComplaintCategory.PRODUCT_DEFECT,
        r"\b(?P<issue>faulty|non[- ]functional|non[- ]working|inoperative)\b"
        r"(?!\s+(?:design|fit|compatibility|appearance|description|listing|advertising|"
        r"box|package|packaging)\b)",
    ),
    _rule(
        "defect.function_specific",
        ComplaintCategory.PRODUCT_DEFECT,
        r"\b(?P<issue>(?:(?:does not|doesn['’]t|do not|don['’]t|did not|didn['’]t|"
        r"will not|won['’]t|would not|wouldn['’]t|cannot|can['’]t)\s+filter|"
        r"(?:water|liquid)\s+(?:does not|doesn['’]t|did not|didn['’]t|will not|"
        r"won['’]t|would not|wouldn['’]t|cannot|can['’]t)\s+(?:flow|pass through)))\b",
        True,
    ),
    _rule(
        "defect.coating",
        ComplaintCategory.PRODUCT_DEFECT,
        r"\b(?:paint|coating|enamel|lining|liner|seal)\b[^.!?;\n]{0,65}\b"
        r"(?P<issue>(?:chips?|chipped|chipping|peels?|peeled|peeling|flakes?|flaked|flaking)"
        r"\s+off|(?:is|was|has been|started|began)\s+(?:chipping|peeling|flaking))\b",
    ),
    _rule(
        "arrival.condition",
        ComplaintCategory.ARRIVAL_DAMAGE,
        r"\b(?:arrived|came|received|delivered|got)\b[^.!?;\n]{0,55}\b"
        r"(?P<issue>damage(?:d)?|dent(?:ed)?|crack(?:ed)?|crushed|broken|bent|scuffed|scratched)\b",
    ),
    _rule(
        "arrival.unboxing",
        ComplaintCategory.ARRIVAL_DAMAGE,
        r"\b(?P<issue>scuffed|scratched|damaged|dented|cracked|broken|bent)\s+"
        r"(?:(?:on|upon|at)\s+(?:(?:first|initial)\s+)?"
        r"(?:opening|unboxing|arrival|delivery|receipt)|out of (?:the )?box)\b",
    ),
    _rule(
        "arrival.packaging",
        ComplaintCategory.ARRIVAL_DAMAGE,
        r"\b(?:box|package|packaging)\b[^.!?;\n]{0,30}\b(?:was|is|arrived|came) "
        r"(?:(?:badly|severely) )?(?P<issue>damaged|crushed|torn|dented)\b",
    ),
    _rule(
        "wrong.fulfilment",
        ComplaintCategory.WRONG_ITEM,
        r"\b(?:received|sent|shipped|delivered|got)\b[^.!?;\n]{0,35}\b"
        r"(?P<issue>wrong|incorrect) (?:item|product|part|model|component|size|colou?r|one)\b",
    ),
    _rule(
        "wrong.ordered",
        ComplaintCategory.WRONG_ITEM,
        "\\b(?:received|got)\\b[^.!?;\\n]{0,35}\\b(?P<issue>not what I "
        "ordered|not the (?:item|product|parts?) I ordered)\\b",
        True,
    ),
    _rule(
        "delivery.nonreceipt",
        ComplaintCategory.DELIVERY_ISSUE,
        r"\b(?P<issue>(?:never|not|haven['’]t|hasn['’]t|didn['’]t|have not|has not|did not) "
        r"(?:(?:been|yet|even|ever) ){0,2}(?:arrived|delivered|received))\b",
        True,
    ),
    _rule(
        "delivery.delay",
        ComplaintCategory.DELIVERY_ISSUE,
        r"\b(?P<issue>(?:package|order|shipment|delivery) (?:(?:was|is|has been) )?"
        r"(?:late|delayed|lost|missing)|(?:arrived|delivered) (?:very |too )?late|late delivery)\b",
    ),
    _rule(
        "service.refusal",
        ComplaintCategory.SERVICE_RETURN_ISSUE,
        r"\b(?:customer (?:support|service)|support|seller|vendor|company) "
        r"(?:\w+ ){0,3}(?P<issue>refused|denied|ignored|unhelpful|unresponsive|rude|"
        r"wouldn['’]t|would not|won['’]t|will not)\b[^.!?;\n]{0,45}",
    ),
    _rule(
        "service.process",
        ComplaintCategory.SERVICE_RETURN_ISSUE,
        r"\b(?:refund|return|replacement|warranty)\b[^.!?;\n]{0,40}\b"
        r"(?P<issue>refused|denied|rejected|impossible|difficult|hassle|problem)\b",
    ),
    _rule(
        "service.obtain",
        ComplaintCategory.SERVICE_RETURN_ISSUE,
        r"\b(?P<issue>(?:could not|couldn['’]t|cannot|can['’]t|unable to) "
        "(?:get (?:a )?)?(?:refund|return|replacement)|(?:had|have)"
        " to pay\\b[^.!?;\\n]{0,25}\\breturn shipping)\\b",
        True,
    ),
    _rule(
        "service.refund_refusal",
        ComplaintCategory.SERVICE_RETURN_ISSUE,
        "\\b(?P<issue>refused|denied) (?:to )?(?:refund|replace|rese"
        "nd|accept (?:the |a )?return)\\b",
    ),
)


def _prefix(text: str, position: int) -> str:
    return re.split(
        r"[.!?;\n]|\b(?:but|however|yet)\b", text[max(0, position - 100) : position], flags=re.I
    )[-1]


def _suppression(text: str, match: re.Match[str], rule: Rule) -> SuppressionReason | None:
    before = _prefix(text, match.start("issue"))
    if rule.category == ComplaintCategory.PRODUCT_DEFECT and re.search(
        r"\b(?:broken|defective) (?:box|package|packaging)\b",
        text[match.start() : match.end() + 24],
        re.I,
    ):
        return "packaging_only"
    # Negation of the complaint cue, not negation intrinsic to 'does not work'.
    negation_context = re.sub(r"\bnot only\b", "", before, flags=re.I)
    if not rule.intrinsic_negation and re.search(
        r"\b(?:no|not|never|without|nothing)\b[^,]{0,18}$", negation_context, re.I
    ):
        return "negated"
    if re.search(r"\b(?:if|might|may|afraid|worried|hope|concerned)\b[^,]{0,45}$", before, re.I):
        return "hypothetical"
    if rule.identifier == "defect.coating":
        if re.search(r"\b(?:will|would|could|should)\b[^,]{0,25}$", before, re.I):
            return "hypothetical"
        # Coating loss needs an interior/use context, rather than cosmetic dislike.
        after = re.split(r"[.!?;\n]", text[match.end() : match.end() + 80])[0]
        if not re.search(
            r"\b(?:inside|interior|wet|water|ice|food|coffee|use|uses|used|using|wash|washing)\b",
            before + match.group("issue") + after,
            re.I,
        ):
            return "cosmetic_only"
    if rule.category == ComplaintCategory.PRODUCT_DEFECT and re.search(
        "\\b(?:old|previous|original)\\b.{0,40}$|\\b(?:to replace|repl"
        "acing|replacement for)\\b.{0,50}$",
        before,
        re.I,
    ):
        return "unrelated_old_product"
    if rule.identifier == "delivery.nonreceipt" and re.match(
        r"\s+(?:(?:a|the|my|any)\s+)?(?:refund|response|reply|instructions)\b",
        text[match.end() :],
        re.I,
    ):
        return "non_delivery_object"
    return None


def _resolution(text: str, end: int, category: ComplaintCategory) -> TextSpan | None:
    patterns = {
        ComplaintCategory.PRODUCT_DEFECT: r"\b(?:now (?:it )?works|replacement (?:now )?works|"
        r"(?:problem|issue) (?:is|was|has been) (?:fixed|resolved))\b",
        ComplaintCategory.ARRIVAL_DAMAGE: (
            r"\b(?:damage|dent) (?:was|is|has been) (?:fixed|repaired|resolved)\b"
        ),
        ComplaintCategory.WRONG_ITEM: (
            r"\b(?:correct|right) (?:item|part|product) "
            r"(?:was |has been )?(?:received|delivered)\b"
        ),
        ComplaintCategory.DELIVERY_ISSUE: (
            r"\b(?:finally arrived|eventually arrived|package was found)\b"
        ),
        ComplaintCategory.SERVICE_RETURN_ISSUE: (
            r"\b(?:refund (?:was |has been )?(?:issued|received|processed)|"
            r"(?:problem|issue) (?:is|was|has been) resolved)\b"
        ),
    }
    match = re.search(patterns[category], text[end : end + 220], re.I)
    if match is None:
        return None
    return TextSpan(end + match.start(), end + match.end(), match.group())


def detect_complaints(text: str) -> ComplaintDetection:
    """Offsets are half-open Python Unicode code points in the unmodified input."""
    if not isinstance(text, str):
        raise TypeError("Complaint input must be raw review text")
    if not text.strip() or len(text) > 50000:
        raise ValueError("Provide nonblank review text of at most 50,000 characters")
    evidence = []
    suppressed = []
    for rule in RULES:
        for match in rule.pattern.finditer(text):
            span = TextSpan(match.start(), match.end(), match.group())
            reason = _suppression(text, match, rule)
            if reason is not None:
                suppressed.append(SuppressedEvidence(rule.category, span, rule.identifier, reason))
                continue
            resolution = _resolution(text, match.end(), rule.category)
            evidence.append(
                ComplaintEvidence(
                    rule.category,
                    span,
                    rule.identifier,
                    "reported_resolved" if resolution else "not_established",
                    resolution,
                )
            )
    evidence.sort(key=lambda item: (item.span.start, item.span.end, item.rule_id))
    suppressed.sort(key=lambda item: (item.span.start, item.span.end, item.rule_id))
    categories = tuple(
        category
        for category in ComplaintCategory
        if any(item.category == category for item in evidence)
    )
    return ComplaintDetection(
        categories,
        tuple(evidence),
        tuple(suppressed),
        "customer_reported_evidence" if categories else "unknown_no_detected_evidence",
    )
