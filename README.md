# SignalDesk — BC3415 Option A

Retail customer intelligence from review text, with evidence for human investigation.

**This is a source-code submission. Model weights, review datasets, annotations and
frozen result artifacts are not bundled.** The backend and frontend can start without
them, but historical views show unavailable data and sentiment analysis is partial.
Complaint-rule analysis remains available. No mock data or predictions are substituted.

## Product and architecture

Three application views:

1. **Executive Overview:** historical review/rating burden, complaint scan coverage,
   product search and investigation priorities.
2. **Live Review Analyzer:** three-class sentiment with actual model identity and
   CPU fallback, plus five independent complaint categories and exact rule excerpts.
3. **Product Investigation Workbench:** filtered/paginated candidates, product calendars,
   temporal comparisons and source-review evidence, including protected-body states.

One FastAPI application serves a React/TypeScript frontend using Tailwind, Lucide and
Radix tooltips. Pydantic defines canonical API contracts; exported JSON Schema generates
frontend types and validates responses. Polars loads permitted historical records into
one lazy in-memory snapshot. SQLite is reserved, not required. Inference is lazy and
separate from offline training/evaluation; there are no external paid prediction APIs,
automatic customer actions or persisted triage decisions. Submitted text is not saved.

Implemented API paths under `/api/v1`: `health`, `summary`, `products`,
`products/{product_id}`, `products/{product_id}/timeline`, `investigations`,
`investigations/{signal_id}`, `reviews/{review_id}` (GET), and
`reviews/analyze` (POST). Business payloads use schema 1.1; legacy offline contracts
and error envelopes retain 1.0. Health is liveness only, not model/data readiness.

### Batch CSV analysis

Inside Live Review Analyzer, choose **Batch CSV Analysis**. Upload a UTF-8 `.csv`
with a case-sensitive `text` header and optional `review_id`; other columns are
ignored. Files are limited to **2 MiB and 100 review rows**, with nonblank text of
at most 50,000 Unicode characters per row. Quoted commas, escaped quotes and multiline
text are supported. Invalid files are rejected before any analysis; a short preview
precedes processing. Each input row is counted separately, even if IDs repeat.

Requests use the existing analysis endpoint sequentially. Controls prevent duplicate
starts; **Cancel remaining** waits for the current response, retains it and stops
further requests. Leaving the analyzer stops further browser dispatch, but cannot
guarantee cancellation of an already-running server request. A request failure stops
the batch without automatic retries. Cold loading can take minutes; actual model,
device, fallback reason and partial/unavailable states remain visible.

Totals distinguish completed API responses (including partial analyses), failed
requests and unprocessed rows. Sentiment counts and five independent complaint counts
use completed responses only. Category counts can overlap. Search/outcome/complaint
filters affect the results table, not the exported rows. Exact evidence and its
Unicode offsets use the original uploaded text, independently of sentiment.

**Export CSV** includes every input row, original ID/text, processing state, actual
predictions, provenance and complaint states/evidence. Spreadsheet-formula-leading
cells receive an apostrophe; `spreadsheet_sanitized_fields` lists adjusted columns,
so removing only the added apostrophe recovers the exact original value. JSON columns
retain model/rule metadata and warnings; no confidence scores are invented.

Uploads/results stay in temporary browser memory, including when switching analyzer
modes. **Clear batch** or leaving the analyzer discards them. Only an explicit export
creates a local results file. Nothing enters SQLite, local storage or the historical
2023 product intelligence. Backend inference, saved models and evaluation are unchanged.

## Data and ML approach

Source: McAuley Lab, [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/),
Appliances category. Cite Hou et al. (2024), *Bridging Language and Items for Retrieval
and Recommendation*. [Dataset card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
The acquisition URL and bounds are in `config/data-preparation.json`. Original gzip
SHA256: `150f209befceaa6f837abc997065b2d251034bbbda19bebc4ad56dac779730c2`.
This submission does not redistribute review bodies, identifiers or annotations;
do not assume public redistribution permission.

Preprocessing validates bodies, ratings, product IDs and UTC timestamps; preserves
original text; removes reviewer IDs/images; groups normalised exact and bounded
SimHash/Jaccard near-duplicate families; and creates chronological, family-aware
partitions. The sentiment cohort has 46,064 reviews: 32,230 train, 6,913 validation
and 6,921 sealed test. A 240-review blind reservation is excluded from fitting and
model-selection scoring. Test data and annotation reservations remain protected.

Only review-body `model_text` is a sentiment feature. Rating labels are disclosed
proxies: 1–2 negative, 3 neutral, 4–5 positive. Ratings, IDs, timestamps and product
aggregates are never classifier inputs. TF-IDF is fitted on training text only,
using word unigrams/bigrams without stop-word removal. Four CPU baselines compare
majority, Naive Bayes, logistic regression and balanced logistic regression;
validation Macro-F1 selects the balanced pipeline.

Two bounded DistilBERT experiments compare unweighted cross-entropy against
training-derived square-root inverse-frequency weighting. Seed, two-epoch schedules,
resource limits and model-selection settings are retained in configuration/code.
The weighted model remains the provisional CUDA sentiment component; saved balanced
logistic regression is the CPU alternative. No training runs during application startup.

Five complaints are independent of sentiment: product defect, arrival damage, wrong
item, delivery issue, and service/returns. Runtime detection uses frozen explicit rules
and Unicode code-point evidence offsets. No match is **unknown**, not confirmed absence.
Positive sentiment and reported resolution do not erase a complaint. Offline NLI and
weakly supervised student experiments remain included for assessment, but the large
BART checkpoint is not bundled or launched by the application. Wrong-item students are
unreliable; the delivery student lacks support and is unavailable. Uncalibrated NLI
scores are not probabilities.

The original history contains 22,838 unique reviews across 80 selected products, ending
11 September 2023. Negative-rating burden is a rating proxy; rule shares use scanned
bodies, with protected/unavailable bodies excluded. The 37 candidates across 26 products
are historical investigation aids, not confirmed incidents or live alerts. Minimum
sample denominators and complete-period comparisons govern priorities; quarterly
aggregation is used when monthly support is sparse. Category overlaps never inflate
the unique-review total.

## Evaluation

Frozen evaluation compares each model separately against two 150-review LLM references.
Uncertain/missing cells remain unresolved, with explicit denominators. Twenty independent
human judgments were deliberately selected for disagreement analysis. Neither LLM
agreement nor that challenge set estimates population accuracy.

| Sentiment model | LLM reference A: agreement / Macro-F1 | LLM reference B: agreement / Macro-F1 |
|---|---:|---:|
| Balanced logistic regression | 116/143 (81.12%) / 0.663 | 117/148 (79.05%) / 0.599 |
| Unweighted DistilBERT | 115/143 (80.42%) / 0.638 | 117/148 (79.05%) / 0.595 |
| Weighted DistilBERT | 112/143 (78.32%) / 0.617 | 117/148 (79.05%) / 0.625 |

The references do not identify a stable winner. Neutral support is only six/four
resolved labels. Complaint evaluation reports per-label support, precision/recall/F1,
decision coverage and abstentions; rare categories lack sufficient positive support.
For example, rule defect detection F1 is 0.392/0.345 against A/B, with low coverage.
No result establishes verified-defect detection or production accuracy.

Metric, confusion-matrix, alignment and evidence-audit implementations are in
`ml/signaldesk_ml/`. Original per-class results and provenance remain in the private
research archive, not this source package. Aggregate values above are transcribed from
the frozen consolidation, not newly evaluated. No thresholds/models were tuned using
these references. Evaluation and annotation are closed; do not rerun them for setup.

## Install and run (Windows PowerShell)

Verified development environment: Windows x64, Python 3.13.5, Node 24.18.0,
npm 11.16.0 and uv 0.13.5. Python metadata permits 3.11–3.13; other versions/platforms
and clean-machine installation have not been verified. Internet is needed to install
dependencies, not for saved inference. No API key, Docker or database server is needed.

From the extracted/cloned repository root:

```powershell
py -3.13 -m venv .tools
.\.tools\Scripts\python.exe -m pip install uv==0.13.5
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.runtime/uv-cache'
$env:PYTHONUTF8 = '1'
.\.tools\Scripts\uv.exe sync --frozen --all-groups --extra data --extra baselines
npm.cmd --prefix frontend ci
```

Terminal 1:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.runtime/uv-cache'
$env:PYTHONUTF8 = '1'
.\.tools\Scripts\uv.exe run --frozen --extra data --extra baselines uvicorn signaldesk.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

Terminal 2:

```powershell
npm.cmd --prefix frontend run dev
```

Open [the app](http://127.0.0.1:5173) or [API docs](http://127.0.0.1:8000/docs).
Vite proxies `/api` to port 8000; port 5173 is strict. Stop each server with Ctrl+C.
Optional configuration: copy `.env.example` to `.env`, or `frontend/.env.example`
to `frontend/.env.local`. Paths resolve from the repository root; `VITE_` values
are public. Keep access logging disabled to avoid raw URL logging.

For the saved CUDA model, install and retain the GPU extra in both commands:

```powershell
.\.tools\Scripts\uv.exe sync --frozen --all-groups --extra data --extra baselines --extra gpu
.\.tools\Scripts\uv.exe run --frozen --extra data --extra baselines --extra gpu uvicorn signaldesk.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

The verified GPU is an RTX 4070 Laptop GPU (8 GB), with driver 577.02 and the locked
PyTorch CUDA 12.6 wheel. A separate CUDA Toolkit is not required for that wheel.
Without CUDA/dependencies/checkpoint, the backend tries saved balanced LR on CPU;
DistilBERT itself does not run on CPU.

### Runtime artifacts — not included

Obtain the **original compatible artifact set** separately from the project authors
through a permitted private handoff. There is no public checkpoint download supplied.
Do not copy protected annotations or raw reviews into GitHub, even into a private
repository, without verified permission.

| Capability | Required paths relative to root | Approximate sizes |
|---|---|---:|
| Historical intelligence | `data/processed/appliances-v1/investigation.parquet`, `manifest.json` | 4.87 MiB + 4.1 KiB |
| Mandatory history exclusions | `data/annotations/sentiment-v1/internal/provenance.json`; `data/annotations/sentiment-v1/complaints-v1/internal/provenance.json` | 25.0 + 15.1 KiB |
| CPU sentiment | `artifacts/sentiment/baselines-v1/run.json`, `logistic_regression_balanced.joblib` | 25.6 KiB + 1.98 MiB |
| CUDA sentiment | `artifacts/sentiment/distilbert-weighted-v1/run.json`; `checkpoint/class_mapping.json`, `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json` under that directory | 11.3 KiB + 256.11 MiB |

Historical manifest, investigation checksum and exclusion metadata must agree. Do not
relax exclusions or edit hashes. Runtime does not read sealed modelling partitions or
annotation labels. Only load trusted original joblib files. Optional
`SIGNALDESK_TRANSFORMER_DIRECTORY` and `SIGNALDESK_BASELINE_DIRECTORY` override the
model directories.

Research code is retained for inspection/reproduction under separately authorised
access to source data, exact frozen membership/protocol and original checkpoints:
`signaldesk-data prepare`, `signaldesk-baselines train`, and
`python -m signaldesk_ml.transformer_train train --config config/sentiment-transformer-weighted.json`
are the original entry points (use the appropriate installed extras). Transformer
initialisation also requires the pinned cache built by `scripts/cache_distilbert.py`;
weighted training references the original unweighted run. These are **not installation
steps**. Downloading the source alone does not recreate protected annotation judgments
or guarantee byte-identical frozen artifacts. Do not retrain, create annotations or run
the full data verifier against the sealed test partition to get the app running.

### Missing artifacts and troubleshooting

- Historical endpoints return 503 for absent/incompatible data. The UI reports this;
  health can still return 200 with `model_status=not_checked`.
- If both saved sentiment models fail, analysis returns partial/unavailable sentiment
  and independent rules. No fake sentiment label is supplied.
- Models and load failures are cached until restart. First load can be slow; an actual
  CPU cold request took about 54 seconds locally. Inspect the returned fallback reason.
- Connection failure: check both services, ports 8000/5173 and the Vite proxy. Separate
  hosting needs `VITE_API_BASE_URL` and the exact CORS origin configured on the backend.
- Production frontend: `npm.cmd --prefix frontend run build`; preview with
  `npm.cmd --prefix frontend run preview` on port 4173, with the backend running.

## Code map and verification

```text
backend/src/signaldesk/   App, contracts, lazy inference, protected history and priorities
frontend/src/            Three views, typed client, evidence and calendar components
ml/signaldesk_data/      Acquisition, validation, duplicate families and frozen partitions
ml/signaldesk_ml/        Training, selection, inference, metrics and frozen evaluation
config/                 Original data/model/evaluation parameters
scripts/                Contract export, GPU diagnostic and pinned pretrained caches
docs/contracts/         Required canonical schema, imported by the frontend
docs/COMPLAINT_TAXONOMY.md  Immutable, checksum-referenced research specification
tests/                  Small API/configuration/contract behavior tests
frontend/tests/         Focused CSV, cancellation and export behavior tests
```

The taxonomy preserves its original historical annotation protocol because offline code
checks its exact hash; it is not an instruction to start further annotation. Research
report files/figures are generated outputs and are not included. The submission copy
redirects one generated-report annotation hyperlink to this README; algorithms and
application source are unchanged.

Development verification already passed frontend typecheck/lint/build, 24 scoped backend
tests, Ruff/mypy and schema-export checks. Real local walkthroughs exercised historical
navigation/evidence, saved CPU fallback and honest unavailable states; earlier checks
exercised the saved CUDA model. These results are from the original development
environment, not a new clean-install or ML evaluation claim. Packaging verification
checks file completeness, imports/references, privacy and ZIP/Git byte equality only.

Batch extension verification: seven focused Node tests passed, as did frontend
typecheck, lint and production build. A real browser batch of three synthetic reviews
used saved balanced-LR CPU fallback: 1 positive, 2 negative, 0 neutral/unavailable;
rule counts were defect 1, arrival 1, service/returns 2, wrong-item/delivery 0.
Export round-trip preserved original text/IDs (after the documented formula guard)
and exact Unicode evidence offsets, including quoted commas and multiline text.
Cancellation retained 9 completed responses and left 91 of a 100-row input unprocessed.
With both model paths absent, all three responses were partial/unavailable for
sentiment while retaining rule evidence. Single-review inference, filtering and
narrow layout were checked. Normal backend configuration was restored. No new GPU
inference, training or evaluation was run.

Optional lightweight checks, after installation:

```powershell
.\.tools\Scripts\uv.exe run --frozen --extra data --extra baselines pytest
.\.tools\Scripts\uv.exe run --frozen python scripts/export_contracts.py --check
npm.cmd --prefix frontend run test:batch
npm.cmd --prefix frontend run typecheck
npm.cmd --prefix frontend run lint
npm.cmd --prefix frontend run build
```

Third-party dependencies are installed separately; package metadata/lockfiles retain
their attribution/license metadata. No third-party implementation bundles or model
weights are copied into this submission. Pinned pretrained model identifiers are in
configuration; their separately obtained artifacts retain their upstream notices.
