"""Compact Markdown audit and two source-backed plots; no customer text is published."""

from pathlib import Path
from typing import Any

import matplotlib
import polars as pl

from signaldesk.config import PROJECT_ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def percent(count: int, denominator: int) -> str:
    return f"{100 * count / denominator:.2f}%" if denominator else "n/a"


def label_row(name: str, profile: dict[str, Any]) -> str:
    values = profile["labels"]
    cells = [
        f"{values.get(label, 0):,} ({percent(values.get(label, 0), profile['rows'])})"
        for label in ["negative", "neutral", "positive"]
    ]
    return f"| {name} | {profile['rows']:,} | " + " | ".join(cells) + " |"


def write_report(
    audit: dict[str, Any],
    manifest: dict[str, Any],
    sentiment: pl.DataFrame,
    incidents: pl.DataFrame,
    output: Path,
) -> None:
    source = audit["source"]
    plots = PROJECT_ROOT / "docs/figures"
    plots.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout="constrained")
    names = ["Source", "Reservoir", "Deduplicated"]
    distributions = [
        source["rating_counts_source"],
        audit["reservoir"]["ratings"],
        audit["sentiment"]["ratings"],
    ]
    for offset, (name, distribution, color) in enumerate(
        zip(names, distributions, ["#9cb7b6", "#4d8c80", "#163b3d"], strict=True)
    ):
        total = sum(distribution.values())
        axes[0].bar(
            [rating - 0.24 + offset * 0.24 for rating in range(1, 6)],
            [100 * distribution.get(str(rating), 0) / total for rating in range(1, 6)],
            width=0.23,
            label=name,
            color=color,
        )
    axes[0].set(
        title="Deduplication changes the rating mix",
        xlabel="Original numeric rating",
        ylabel="Share of reviews (%)",
        xticks=range(1, 6),
    )
    axes[0].legend(frameon=False, fontsize=8)
    for name, years, color in [
        ("Eligible source", source["eligible_year_counts"], "#9cb7b6"),
        ("Sentiment cohort", audit["sentiment"]["years"], "#163b3d"),
    ]:
        total = sum(years.values())
        axes[1].plot(
            [int(year) for year in sorted(years)],
            [100 * years[year] / total for year in sorted(years)],
            label=name,
            color=color,
            linewidth=2,
        )
    axes[1].set(
        title="Historical coverage; snapshot ends in 2023",
        xlabel="Review year (UTC)",
        ylabel="Share of reviews (%)",
    )
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(plots / "data-rating-time.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4, 3.6), layout="constrained")
    for df, name, color in [
        (sentiment, "Sentiment sample", "#9cb7b6"),
        (incidents, "Investigation histories", "#163b3d"),
    ]:
        counts = df.group_by("product_id").len()["len"].sort().to_list()
        ax.step(
            counts,
            [100 * (index + 1) / len(counts) for index in range(len(counts))],
            where="post",
            label=name,
            color=color,
            linewidth=2,
        )
    ax.set(
        xscale="log",
        xlabel="Retained reviews per product (log scale)",
        ylabel="Cumulative share of products (%)",
        title="Random review samples do not preserve product histories",
    )
    ax.legend(frameon=False)
    fig.savefig(plots / "data-product-coverage.png", dpi=150)
    plt.close(fig)

    label_table = "\n".join(
        label_row(name, audit[key])
        for name, key in [
            ("Uniform reservoir", "reservoir"),
            ("Deduplicated sentiment", "sentiment"),
            ("Investigation only", "investigation"),
        ]
    )
    split_table = "\n".join(
        label_row(name, audit["splits"][name]) for name in ["train", "validation", "test"]
    )
    source_rating_rows = []
    for rating in range(1, 6):
        count = source["rating_counts_source"].get(str(rating), 0)
        share = percent(count, sum(source["rating_counts_source"].values()))
        source_rating_rows.append(f"| {rating} | {count:,} | {share} |")
    source_rating_table = "\n".join(source_rating_rows)
    schema_table = "\n".join(
        f"| {key} | {', '.join(f'{kind}: {count:,}' for kind, count in values.items() if count)} |"
        for key, values in sorted(source["schema"].items())
    )
    rejects = (
        "\n".join(
            f"| {key} | {value:,} | {percent(value, source['source_records'])} |"
            for key, value in sorted(source["rejection_reasons_nonexclusive"].items())
        )
        or "| None | 0 | 0.00% |"
    )
    flags = "\n".join(
        f"| {key} | {value:,} | {percent(value, source['eligible_records'])} |"
        for key, value in sorted(source["quality_flags_eligible"].items())
    )
    source_products = source["product_review_count_quantiles"]
    sent_products = audit["sentiment"]["product_counts"]
    incident_products = audit["investigation"]["product_counts"]
    lengths = source["text_length_characters"]
    duplicate = audit["duplicates_cohort_union"]
    provenance = manifest["source_provenance"]
    text = f"""# SignalDesk Appliances data quality report

Prepared on 2026-10-25 (Asia/Singapore). These are actual corpus statistics, not model
performance results. No model was trained. The frozen holdout is for rating-proxy
evaluation; independently verified text sentiment and complaint labels remain necessary.

## Suitability decision

**Conditional Go for Phase 2 sentiment baselines against disclosed rating proxies.**
All three classes are present in every chronological split. The separate investigation
cohort provides substantial product histories for retrospective signal exploration.
It cannot validate complaint classification or verified defect detection: complaint
annotations and incident ground truth are absent. The corpus is historical, not a live
retail feed, and cannot establish current operational performance.

## Source and acquisition

Category: **Appliances**, McAuley Lab Amazon Reviews 2023. The
[official category listing](https://amazon-reviews-2023.github.io/) and
[dataset card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
identify the source. The [maintainer's usage clarification](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/discussions/1)
states that they cannot assign a license or dictate usage terms and primarily make the
data available for research. This is not an unrestricted redistribution license. Keep raw
text, processed review files and identifiers local; review terms before any public release.
Cite Hou et al. (2024), *Bridging Language and Items for Retrieval and Recommendation*.

One [official gzip category file]({provenance["requested_url"]}) was downloaded:
{provenance["bytes"]:,} bytes ({provenance["bytes"] / 2**20:.2f} MiB), last modified
{provenance["last_modified"]}. SHA256:
`{provenance["sha256"]}`.
A 128 KiB preflight inspected 100 rows without storing excerpts. The download cap is
320 MiB; uncompressed/row caps are 1.5 GiB / 2.5 million. The Hugging Face mirror lists
929,451,412 uncompressed bytes. No remote dataset code, images or metadata were acquired.
The alternative UCSD host had an untrusted certificate chain in this environment;
the accessible official McAuley host passed normal TLS verification. TLS checks were not disabled.

## Counts, schema and validity

| Stage | Records | Products |
|---|---:|---:|
| Category rows inspected | {source["source_records"]:,} | Not inferred from invalid rows |
| Eligible source rows | {source["eligible_records"]:,} | {source["products"]:,} |
| Excluded source rows | {source["rejected_records"]:,} | — |
| Seeded review reservoir | {audit["reservoir"]["rows"]:,} | {audit["reservoir"]["products"]:,} |
| Sentiment cohort | {audit["sentiment"]["rows"]:,} | {audit["sentiment"]["products"]:,} |
| Investigation | {audit["investigation"]["rows"]:,} | {audit["investigation"]["products"]:,} |

One source row represents a customer review record. There is no provided review ID;
stable project IDs combine category and line ordinal under the pinned source checksum.
`parent_asin` is the aggregation key; `asin` is retained. Invalid/missing parent IDs are
excluded rather than silently mixing variant and parent namespaces. Usable parent IDs:
{source["usable_parent_id_records"]:,}/{source["source_records"]:,}
({percent(source["usable_parent_id_records"], source["source_records"])}).

| Source field | Actual JSON type counts (absent reported separately) |
|---|---|
{schema_table}

Source null/empty-string counts: `{json_summary(source["null_or_empty_fields"])}`.
Missing and whitespace-only text are also checked explicitly. Exclusion reasons are
nonexclusive and use all source rows as denominator:

| Quality failure | Rows | Source share |
|---|---:|---:|
{rejects}

Observed timestamp units: `{json_summary(source["timestamp_units"])}`. Actual millisecond
values were verified, converted to timezone-aware UTC, and bounded to the source's
1996–September 2023 snapshot. Known `sort_timestamp`/`helpful_votes` aliases are supported;
the actual source uses `timestamp`/`helpful_vote`. Product IDs remain strings. Processed
Parquet drops `user_id` and `images`; review text may still contain personal information,
so no raw examples are published in this report. Canonical CustomerReview validation is
run for retained rows. Original `text` is preserved exactly and is the only `model_text`;
title/rating/product identifiers must not be classifier features.

## Ratings, proxy labels and text quality

| Rating | Source rows with valid ratings | Share |
|---|---:|---:|
{source_rating_table}

Proxy mapping: 1–2 → negative, 3 → neutral, 4–5 → positive; `label_source=rating_proxy_v1`.
These labels do not establish the text's true sentiment. Mixed opinions, sarcasm,
mis-clicked ratings and three-star ambiguity can disagree with the mapping. No class
balancing or sentiment-based sampling was performed. Any class weighting later must be
learned/chosen using training/validation data only. Preserve negation and punctuation.

| Cohort | Reviews | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|
{label_table}

Nonempty source text character counts: min {lengths["min"]:,}, p25 {lengths["p25"]:,},
median {lengths["median"]:,}, p95 {lengths["p95"]:,}, max {lengths["max"]:,}.
Flags below are nonexclusive heuristics over eligible rows, **not verified language
identification**. Latin-script non-English text can escape them; `language` stays null.
No language or short-text filter was applied, so later annotation must audit these slices.

| Suspected text issue | Rows | Eligible share |
|---|---:|---:|
{flags}

## Duplicates and sampling effects

Across the full source, identical original review text has
{source["exact_text_duplicate_excess_rows"]:,} excess occurrences
({percent(source["exact_text_duplicate_excess_rows"], source["source_records"])} of source rows);
{source["exact_text_duplicate_affected_rows"]:,} rows participate in repeated-text groups.
This counts text equality, not verified repeated customers or duplicated review events.
Generic phrases can be legitimate independent reviews.

Near-duplicate analysis covers the **{duplicate["union_records"]:,}-row cohort union**, not
the entire category. NFKC/case/whitespace equivalence yields
{duplicate["normalized_exact_redundant_rows"]:,} redundant rows. For distinct texts with
at least eight tokens, deterministic 64-bit SimHash (Hamming distance ≤3) plus word-trigram
Jaccard ≥0.85 identifies {duplicate["near_duplicate_edges"]:,} near-duplicate edges from
{duplicate["candidate_comparisons"]:,} candidate comparisons. Four 16-bit bands provide
complete candidate coverage for this SimHash radius; semantic paraphrases and pairs
outside that signature radius may remain. These are the detector's limits, not a claim
that every possible near duplicate was found.

Connected families span both cohorts. {duplicate["proxy_conflicting_families"]:,} families
contain conflicting proxy labels, illustrating text/rating inconsistency or generic text.
The sentiment cohort keeps one seed-hash-ranked reservoir review per family, independent
of rating, timestamp or class. {audit["removed_from_reservoir_duplicate_rows"]:,} reservoir
rows were removed. Review-level reservoir sampling was uniform over eligible source
rows (seed 3415); deduplication intentionally reduces common phrases and changes that
distribution. See the before/after rating plot. No post-dedup representative claim is
made for the full natural review population.

## Product histories and temporal coverage

Eligible source UTC range: **{source["time_min_utc"]} to {source["time_max_utc"]}**.
The source snapshot ends in 2023, so historical quality signals cannot be described as
current emerging incidents. Source reviews/product min/median/p95/max:
{source_products["min"]:,}/{source_products["median"]:,}/{source_products["p95"]:,}/{source_products["max"]:,}.
Top 10 products account for {100 * source["top_10_product_share"]:.2f}% of eligible reviews;
top 100 account for {100 * source["top_100_product_share"]:.2f}%.

Sentiment-sample reviews/product min/median/p95/max:
{sent_products["min"]:,}/{sent_products["median"]:,}/{sent_products["p95"]:,}/{sent_products["max"]:,}.
Investigation reviews/product: {incident_products["min"]:,}/{incident_products["median"]:,}/
{incident_products["p95"]:,}/{incident_products["max"]:,}.
The investigation cohort selects up to 80 products by seeded product hash, from products
with 100–2,000 eligible reviews and at least 365 days' history, capped at 60,000 total
reviews. All eligible source records for each selected product are retained across time.
This excludes sparse, short-lived and very high-volume products and is not representative
of all products. It is complete only within this category snapshot and stated quality rules.
Metadata was skipped because titles/categories are optional and not needed to validate
the primary tasks; no metadata join coverage is implied.

![Observed rating mix and temporal coverage](figures/data-rating-time.png)

![Retained review coverage per product](figures/data-product-coverage.png)

## Frozen evaluation partitions

Primary strategy: **chronological by UTC day**, approximately 70/15/15 after deduplication.
Cutoffs are {manifest["split_cutoffs_utc"][0]} and {manifest["split_cutoffs_utc"][1]};
training precedes the first, validation lies between them, and test starts at the second.
The same day never crosses a boundary. All detected text families have exactly one
sentiment representative; there is no detected family/ID/normalized-exact leakage.

| Partition | Reviews | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|
{split_table}

Random stratification would stabilize label shares but mix future/past and shared products.
Product grouping measures unseen-product generalization but changes the operational
question and can struggle with large linked duplicate components. Chronology matches
historical-to-future review classification, permits known products, and exposes time/class
shift; it does not measure unseen-product generalization. New-product evaluation remains
a later separate study. No strict chronology class-coverage failure occurred.

{audit["shared_review_ids"]:,} review IDs occur in both cohorts, each with one frozen
sentiment assignment. Investigation rows carry `cohort_role=investigation_only`, an
`is_sentiment_member` flag and `related_sentiment_split` for any matching text family.
They are never additional training examples. Full histories include later reviews and
must not be used to construct sentiment features or tune models. Historical complaint
replays must filter by observation time to avoid future information.

All Parquet files, source/config/code/lock hashes, cutoffs and the split membership hash
are recorded in `{output.relative_to(PROJECT_ROOT).as_posix()}/manifest.json`.
Repeated preparation verifies the frozen files without rewriting them; changed
configuration or corrupted outputs fail. The test set is sealed for final evaluation;
training/validation alone drive model/threshold/calibration choices.

## Recommendations and reproducibility

- Proceed to Phase 2 only on instruction, starting with text-only TF-IDF baselines.
  Report proxy agreement clearly; do not describe it as verified sentiment accuracy.
- Independently annotate sentiment and multi-label complaints, including neutral,
  mixed-sentiment, short and suspected non-English reviews. Keep gold evaluation blind
  to ratings/model predictions; do not use model-generated labels as independent truth.
- Audit long reviews/truncation and duplicate-policy sensitivity using training and
  validation only. Study class/time shift without optimizing against the frozen test set.
- Treat historical complaint monitoring as exploratory until complaint labels, alert
  denominators/support rules and temporal replay evaluation are established.
- Keep all review-bearing files local. Public deployment/redistribution requires a
  separate terms/privacy decision; no conventional license has been asserted.

Run `uv run --frozen --extra data signaldesk-data prepare` and
`uv run --frozen --extra data signaldesk-data verify`. Structured evidence is in the
ignored `audit.json` and `manifest.json`; preparation code and configuration are versioned.
Focused tests, lint/types and real-output verification are documented in the repository
handoff. No model, dashboard feature or architecture redesign was introduced.
"""
    (PROJECT_ROOT / "docs/DATA_QUALITY_REPORT.md").write_text(text, encoding="utf-8")


def json_summary(value: dict[str, Any]) -> str:
    return "; ".join(f"{key}={count:,}" for key, count in sorted(value.items()) if count) or "none"
