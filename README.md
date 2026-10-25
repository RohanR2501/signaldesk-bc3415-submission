# SignalDesk — AI-Powered Customer Review Intelligence

SignalDesk helps retail customer experience teams, e-commerce operations managers and
analysts turn customer reviews into evidence for investigation. It combines sentiment
classification, independent complaint detection and historical product analytics in a
three-view application, helping users prioritise products, inspect customer-reported
patterns and trace signals back to their supporting reviews. Developed for NTU BC3415 Option A
(Text Classification).

[Quick Start](#quick-start--installation-and-running) ·
[Application Features](#application-features) ·
[Technical Architecture](#technical-architecture) ·
[Machine Learning Methodology](#machine-learning-methodology) ·
[Reproducibility and Data](#reproducibility-and-data)

## Quick Start — Installation and Running

### Prerequisites

Use Windows x64, **Python 3.13**, **Node.js 24** and **uv 0.13.5**. The tested
environment used Python 3.13.5, Node 24.18.0 and npm 11.16.0. An NVIDIA GPU is optional:
saved balanced logistic regression provides CPU sentiment inference. No API key,
Docker installation or database server is required.

Open PowerShell in the project root. All commands below use that directory.

### 1. Install dependencies

```powershell
py -3.13 -m venv .tools
.\.tools\Scripts\python.exe -m pip install uv==0.13.5

$env:UV_CACHE_DIR = Join-Path (Get-Location) '.runtime/uv-cache'
$env:PYTHONUTF8 = '1'
.\.tools\Scripts\uv.exe sync --frozen --all-groups --extra data --extra baselines
npm.cmd --prefix frontend ci
```

These commands install from `uv.lock` and `frontend/package-lock.json`.

### 2. Configure the environment and runtime artifacts

Create local configuration files if they do not already exist:

```powershell
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
}
if (-not (Test-Path frontend/.env.local)) {
    Copy-Item frontend/.env.example frontend/.env.local
}
```

The defaults connect the frontend to the local backend. Model directories can be changed
through `SIGNALDESK_BASELINE_DIRECTORY` and `SIGNALDESK_TRANSFORMER_DIRECTORY` in
`.env`; relative paths resolve from the project root.

For full functionality, place the separately supplied runtime artifacts in:

- `artifacts/sentiment/baselines-v1/` — saved CPU sentiment pipeline and run metadata.
- `artifacts/sentiment/distilbert-weighted-v1/` — saved transformer checkpoint and run
  metadata, when using GPU inference.
- `data/processed/appliances-v1/` — historical investigation records and manifest,
  accompanied by their matching exclusion metadata under `data/annotations/`.

The source submission does not bundle these artifacts. The application can start
without them and perform complaint-rule analysis; sentiment and historical views depend
on the corresponding artifacts. See [the exact file list](#runtime-artifacts) below.

### 3. Start the backend

In terminal 1:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.runtime/uv-cache'
$env:PYTHONUTF8 = '1'
.\.tools\Scripts\uv.exe run --frozen --extra data --extra baselines uvicorn signaldesk.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

### 4. Start the frontend

In terminal 2, from the same project root:

```powershell
npm.cmd --prefix frontend run dev
```

Open **[SignalDesk at http://127.0.0.1:5173](http://127.0.0.1:5173)**.
Interactive API documentation is available at
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
Vite proxies `/api` requests to the backend. Stop each service with **Ctrl+C**.

### Optional: enable GPU sentiment inference

With a compatible NVIDIA GPU and the saved transformer artifacts, add the GPU extra
to installation and backend startup:

```powershell
.\.tools\Scripts\uv.exe sync --frozen --all-groups --extra data --extra baselines --extra gpu
.\.tools\Scripts\uv.exe run --frozen --extra data --extra baselines --extra gpu uvicorn signaldesk.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

The tested GPU is an RTX 4070 Laptop GPU with 8 GB VRAM. The locked Windows PyTorch
wheel includes CUDA 12.6 runtime support; a separate CUDA Toolkit is not required.
If transformer loading or CUDA inference is unavailable, the backend automatically
tries the saved balanced-LR CPU pipeline. DistilBERT itself does not run on CPU.

### Basic troubleshooting

| Symptom | Action |
|---|---|
| App cannot reach the backend | Start both terminals and check ports 8000 and 5173. The frontend requires port 5173. |
| First analysis takes longer | Allow time for local model initialisation; progress is displayed. |
| Sentiment unavailable or historical data returns 503 | Check the artifact paths and matching metadata, then restart the backend to clear cached load failures. |
| Health reports connected but analysis is unavailable | Health checks service liveness; data and model readiness are reported by the relevant responses. |

For a frontend production build, use `npm.cmd --prefix frontend run build`, then
`npm.cmd --prefix frontend run preview` on port 4173 with the backend running.

## Application Features

| View | Main capabilities |
|---|---|
| **Executive Overview** | Historical review volume, rating-derived negative burden, complaint scan coverage, product search and investigation priorities. |
| **Live Review Analyzer** | On-demand single-review and batch CSV sentiment inference, five complaint categories, exact evidence, model provenance and CSV export. |
| **Product Investigation Workbench** | Product and candidate filters, paginated investigation queues, historical calendars, supported period comparisons and source-review exploration. |

A typical workflow starts with the overview to identify a product worth investigating,
continues in the workbench to compare periods and inspect supporting reviews, and uses
the analyzer to examine additional review text. Candidate signals support human
investigation; they do not verify product defects.

### Single-review and batch analysis

Single-review mode accepts original text and displays positive, neutral or negative
sentiment alongside independent complaint results. Batch mode accepts a UTF-8 CSV with
a required `text` column and optional `review_id`. It supports quoted commas,
escaped quotes and multiline text, with limits of **100 reviews, 2 MiB per file and
50,000 Unicode characters per review**. A preview appears before processing.

Batch requests run sequentially through the same inference endpoint. Progress separates
processed, failed and unprocessed rows; cancellation stops remaining work after the
current request finishes. Results include sentiment counts, complaint counts and a
filterable review table with original text, evidence and actual model identity.
Request failures stop further dispatch without automatic retries.

CSV export includes all input rows and their processing states, regardless of filters.
Formula-leading cells are prefixed with an apostrophe; `spreadsheet_sanitized_fields`
identifies those reversible adjustments. Uploaded text and results stay in temporary
browser memory until cleared or the analyzer is left. An explicit export saves a local
file; batch results are not added to historical product analytics.

## Technical Architecture

| Component | Implementation |
|---|---|
| Frontend | React, strict TypeScript and Tailwind, with shared evidence components and a typed API client. |
| Backend | One FastAPI application with validated settings, Pydantic contracts and sanitised errors. |
| Sentiment inference | Lazily loaded saved weighted DistilBERT on CUDA, with automatic saved balanced-LR CPU fallback. |
| Complaint detection | Versioned explicit rules evaluated independently of sentiment, returning exact source spans and per-category states. |
| Historical intelligence | Polars-backed aggregation of permitted records into a cached snapshot, with provenance, denominators and transparent priority rules. |

The API uses `/api/v1` for health, summary, product, timeline, investigation and review
resources. Single and batch analysis both call `POST /api/v1/reviews/analyze`.
Canonical contracts are defined in
[contracts.py](backend/src/signaldesk/contracts.py), with exported schemas in
[docs/contracts/schema-v1.json](docs/contracts/schema-v1.json).
Training and evaluation stay offline; API requests do not execute training workflows.

```text
backend/src/signaldesk/   API, inference orchestration and historical intelligence
frontend/src/            Three application views, typed client and shared components
frontend/tests/          CSV parsing, processing and export tests
ml/signaldesk_data/      Acquisition, preprocessing and duplicate-aware partitioning
ml/signaldesk_ml/        Training, model selection, inference and evaluation
config/                 Data, model and evaluation configurations
scripts/                Contract export and pretrained-model setup utilities
docs/                   Canonical schema and frozen complaint taxonomy
tests/                  Backend API, configuration and contract tests
```

## Machine Learning Methodology

### Data preparation and sentiment labels

The project uses the **Amazon Reviews 2023 Appliances** dataset. Preprocessing validates
review text, ratings, product identifiers and UTC timestamps, removes reviewer IDs and
images, groups exact and bounded near-duplicate text families, and produces chronological,
family-aware partitions. Original review bodies remain available for traceable evidence.

The sentiment cohort contains 46,064 reviews: 32,230 training, 6,913 validation and
6,921 sealed test records. A 240-review annotation reservation is excluded from fitting
and model-selection scoring, leaving **6,673 validation reviews** for comparison.

Training labels are rating-derived proxies: **1–2 stars → negative, 3 → neutral,
4–5 → positive**. They are not independently human-verified sentiment ground truth.
Only review-body `model_text` is used as a classifier feature; ratings, titles,
identifiers, dates and product aggregates are excluded from model inputs.

### Models and selection

Four classical baselines compare a majority classifier, TF-IDF with Naive Bayes,
logistic regression and balanced logistic regression. TF-IDF uses word unigrams/bigrams
and is fitted on training text only. Validation Macro-F1 selects the balanced-LR
classical pipeline.

Two two-epoch DistilBERT experiments compare unweighted cross-entropy with
training-derived square-root inverse-frequency class weighting. Their original
configurations retain seed 3415, training settings and resource limits. Weighted
DistilBERT is the provisional GPU sentiment component; balanced LR supplies CPU fallback.

### Evaluation

The following frozen results use the **same 6,673-review rating-proxy validation set**:

| Model | Validation accuracy | Macro-F1 |
|---|---:|---:|
| Balanced Logistic Regression | 82.65% | 0.6269 |
| Unweighted DistilBERT | 89.84% | 0.6669 |
| Weighted DistilBERT | 87.97% | 0.6756 |

Weighted DistilBERT has the highest observed Macro-F1 in these runs, while the unweighted
model has higher overall accuracy. These comparisons guide model selection; they do
not establish statistical superiority or independently verified population accuracy.
Evaluation code retains per-class support, precision/recall/F1, confusion matrices
and error analysis.

A separate frozen analysis compared predictions against two 150-review LLM references
and 20 deliberately selected independent human judgments, preserving uncertainty.
LLM-reference agreement and a selected challenge set do not replace representative
human ground truth. The test partition remains sealed, and evaluation is closed.

### Complaint detection and explainability

The runtime detector, `complaint-rules-v2`, covers explicit component/filtering/water-flow
failures, coating loss with interior or use context, and receipt/unboxing damage.
Coverage remains conservative and is not independently validated. Frozen offline
complaint comparisons refer to v1 and do not establish performance for v2.

The five categories are **product defect, arrival damage, wrong item, delivery issue,
and service/returns**. Runtime rules return exact matched passages, half-open Unicode
code-point offsets, rule provenance and reported-resolution context where available.
The [frozen complaint taxonomy](docs/COMPLAINT_TAXONOMY.md) defines category scope and
is retained unchanged because offline code verifies its checksum.

Sentiment and complaints remain independent: positive sentiment or a reported resolution
does not erase a complaint. No rule match means **unknown**, rather than confirmed
absence; abstained and unavailable states remain explicit. Rule excerpts are not
sentiment-model attribution spans, and uncalibrated scores are not probabilities.
Offline NLI/student experiments remain assessable in the ML source, but runtime uses
the rules; unreliable wrong-item and unsupported delivery students are not authoritative
classifiers.

## Reproducibility and Data

### Runtime artifacts

The source submission includes implementation, configurations, contracts, lockfiles
and tests. Runtime data and saved model artifacts are supplied separately through an
approved private handoff from the project authors; no public checkpoint download is
provided.

| Capability | Files and locations relative to the project root | Approximate size |
|---|---|---:|
| Historical intelligence | `data/processed/appliances-v1/investigation.parquet` and `data/processed/appliances-v1/manifest.json` | 4.87 MiB + 4.1 KiB |
| Required historical exclusions | `data/annotations/sentiment-v1/internal/provenance.json` and `data/annotations/sentiment-v1/complaints-v1/internal/provenance.json` | 40 KiB combined |
| CPU sentiment | `artifacts/sentiment/baselines-v1/run.json` and `logistic_regression_balanced.joblib` in that directory | About 2 MiB |
| GPU sentiment | `artifacts/sentiment/distilbert-weighted-v1/run.json` and its `checkpoint/` directory | About 256 MiB |

The transformer checkpoint contains `class_mapping.json`, `config.json`,
`model.safetensors`, `tokenizer.json` and `tokenizer_config.json`.
Historical records, manifest and exclusion metadata must belong to the same frozen
artifact set. Saved loaders verify checksums; load only trusted original joblib files.

Without saved sentiment artifacts, the analyzer returns partial results with unavailable
sentiment and independent complaint-rule outputs. Without historical artifacts, historical
views report unavailable data. Health and the application shell remain accessible.

### Research reproduction

The supplied [data configuration](config/data-preparation.json) and
[ML configurations](config/) record source acquisition, preparation and training parameters.
Original entry points include `signaldesk-data prepare`,
`signaldesk-baselines train` and
`python -m signaldesk_ml.transformer_train train --config config/sentiment-transformer-weighted.json`.
Transformer training additionally requires the pinned cache prepared by
`scripts/cache_distilbert.py`; weighted training references the original unweighted run.

These offline workflows require authorised source data, frozen membership/provenance
and the appropriate dependencies; they are not application startup steps. Downloading
the source alone does not reproduce private annotation judgments or guarantee identical
frozen artifacts. Do not reopen annotation/evaluation or run the full data verifier,
which reads the sealed test partition, as part of setup. Clean-machine installation
has not been independently verified.

### Attribution and interpretation

Data source: McAuley Lab,
[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) —
[dataset card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
Cite Hou et al. (2024), *Bridging Language and Items for Retrieval and Recommendation*.
Review bodies, identifiers and protected annotations are excluded from the submission;
redistribution requires appropriate permission. Dependencies and separately obtained
pretrained artifacts retain their upstream licences and notices.

Historical intelligence covers **22,838 reviews across 80 selected products**, ending
**11 September 2023**, with **38 investigation candidates across 26 products**.
Negative-rating burden is a rating proxy, and complaint shares use scanned bodies as
their denominator. Version 2 rules match 1,491 of 20,282 scanned reviews; category
overlaps do not inflate that unique-review total. Priorities
use documented minimum-support and temporal rules, with quarterly comparisons when
monthly observations are sparse. These are customer-reported historical patterns,
not verified incidents or a continuously refreshed Amazon feed.
