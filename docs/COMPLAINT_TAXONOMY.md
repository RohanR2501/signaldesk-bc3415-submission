# Complaint taxonomy and human annotation guide

Version **complaints-v1**, Phase 3A, 2026-10-25 (Asia/Singapore). These five labels
describe **customer-reported experiences**, never independently verified defects,
seller fault or incident severity. Sentiment and complaints are separate decisions.
The examples below are illustrative, not labelled dataset records.

| Label | Include when the review reports | Illustrative yes | Counterexample / exclusion |
|---|---|---|---|
| `product_defect` | The reviewed item/component malfunctions, breaks, fails prematurely or cannot perform its intended function. | “This replacement pump stopped working after a week.” | “My old pump broke; this replacement works.” A dent alone, a disliked design or “overpriced” does not establish malfunction. |
| `arrival_damage` | Physical item or packaging damage at receipt/unboxing, including cosmetic dents even when operation is satisfactory. | “It arrived dented, although it works well.” | “It arrived without damage.” Wear after use without an arrival link. Do not infer carrier responsibility. |
| `wrong_item` | An incorrect item/component/model/variant was supplied relative to what was ordered or explicitly described as supplied. | “I received the wrong model.” | “I ordered the wrong size.” “It doesn't fit” alone is insufficient; use uncertain when a fulfilment or listing mismatch cannot be established. |
| `delivery_issue` | The order/shipment was missing, delayed, lost or unsuccessfully delivered. | “Tracking says delivered, but it never arrived.” | Timely delivery; “not what I received” about the item; a missing refund or response. A missing supplied component may be wrong-item/uncertain, not automatically a delivery failure. |
| `service_return_issue` | A problem with support, refund, warranty handling, replacement negotiation or the return process. | “Support refused a replacement” or “I could not get a refund.” | “Customer service was helpful,” an exact replacement, or a neutral warranty/return mention. A return-window fact alone is not necessarily a complaint. |

Keep all five categories. Arrival and service/returns consolidate the earlier packaging-
damage and customer-service concepts; they avoid splitting closely related operational
problems. Wrong-item and delivery are distinct despite sparse exploratory evidence.
Do not create further labels for every appliance part, shipping carrier or design flaw.

## Consistent decisions

- Label each category independently: any number of **yes** labels, including zero, is
  allowed. Positive sentiment never cancels a complaint. Arrival damage plus a refusal
  to replace can receive both labels. Add product defect only for an explicit functional
  failure or breakage, rather than assuming every dent makes a product nonfunctional.
- **Occurrence, not current incident status:** a problem with the reviewed item/order
  still receives yes when it was subsequently repaired, replaced or refunded. Mark
  “resolved” in notes and include both problem and resolution excerpts. Resolution
  does not prove that all issues are fixed or that a product is safe.
- Exclude failure of an unrelated old appliance when a review praises the replacement
  item. If which product failed is genuinely unclear, use uncertain and explain.
- Negated, hypothetical, feared or quoted-other-review problems are not affirmative
  reports by this customer. “No leaks,” “I feared it might break,” and “other reviewers
  say it arrived damaged” alone are not yes. Treat sarcasm/unclear scope as uncertain.
- A general negative opinion can have no specific category. Use **no** only after
  reading the text for that category; no means no qualifying report in this review,
  not verified absence of real-world problems. Do not force out-of-taxonomy issues
  (such as a disliked design) into a defect label; record the issue in notes.
- Distinguish yes/no/uncertain from an untouched blank. Do not guess the language or
  translate with a classifier; flag insufficient comprehension/insufficient context.

## Completing the working copies

Use only this guide, the existing sentiment guide, and your assigned CSV copy. Do not
open internal provenance, audit examples, keyword outputs, model results or rating
metadata. Original review text may itself mention stars; no external star value is shown.
Keep files local and import ID/text columns as plain text in a spreadsheet application
to avoid interpreting review text as formulas. Preserve review ID, text and row order.

1. Make an independently named copy of `annotation_working.csv`. The optional targeted
   file is a separate task; keep it separate when saving and reporting results.
2. Read the full original text. Fill `human_sentiment` with `positive`, `neutral` or
   `negative` following the existing sentiment guide; leave unresolved sentiment blank
   and explain why with the ambiguity flag set to true.
3. Fill **every** complaint column with `yes`, `no` or `uncertain`. Multiple yes labels
   are allowed. Do not copy a rule's output or let AI fill these fields.
4. Fill `ambiguity_flag` with `true` or `false`. For yes/uncertain, paste exact supporting
   excerpts into `evidence`, prefixed by category, e.g. `arrival_damage: arrived dented`.
   Separate multiple excerpts with ` | `. Include resolution excerpts where relevant;
   offsets are not required from annotators. Explain unclear references and reasons
   for uncertainty in `annotator_notes` without adding personal identifiers.
5. A second annotator independently reviews a pilot and a documented evaluation subset
   (preferably all records if capacity permits). Measure agreement per category and
   sentiment, adjudicate disagreements while still blind, and record annotator/date/
   guide version separately. Preserve original annotator copies and unresolved cases.

The primary 240-review reservation and any optional supplement stay out of complaint
training and model selection. Human labels must be frozen before final independent
evaluation. Neither file is currently annotated. The primary sample was balanced by
rating-proxy strata, not randomly representative of complaint prevalence; a keyword-
targeted supplement is even less representative and must be reported separately.

## Reference detector and future integration

`signaldesk_ml.complaint_rules.detect_complaints(text)` is an unvalidated, conservative
English pattern reference. It returns exact half-open Unicode code-point spans, rule
IDs, suppression reasons and limited resolution evidence; it supplies no probabilities.
It can miss implicit wording, colour substitutions, misspellings, HTML boundaries,
non-English text and complex scope/reference relationships. A detected occurrence is
not necessarily ongoing or urgent; no detected evidence is unknown, not a negative gold label.

The offline taxonomy intentionally precedes transport integration. Existing canonical
API contracts still have preliminary category names and require per-label probabilities;
their evidence methods do not include rule matches. Do not fabricate probabilities or
mislabel rule spans as attribution to force compatibility. A later authorised schema
review must handle names, evidence provenance and unknown/resolved states before API use.
