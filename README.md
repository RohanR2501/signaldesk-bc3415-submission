# SignalDesk

SignalDesk helps retail analysts turn appliance reviews into evidence for investigation.
It combines saved sentiment models, independent complaint rules and historical product
analytics in three views. Developed for NTU BC3415 Option A (Text Classification).

## Quick Start

### 1. Prerequisites and clone

Use **Windows x64, Python 3.13, Node.js 24, Git and GitHub CLI (`gh`)**. The default
setup preserves **weighted DistilBERT on NVIDIA CUDA**; the reference hardware is an
RTX 4070 Laptop GPU with 8 GB VRAM and a compatible NVIDIA driver. The locked PyTorch
wheel includes CUDA 12.6 runtime support; a separate CUDA Toolkit is not required.
Setup installs the pinned **uv 0.13.5** in a separate `.tools` environment.

Your GitHub account must have access to the private repository and its Releases:

```powershell
gh auth login
gh repo clone RohanR2501/signaldesk-bc3415-submission
cd signaldesk-bc3415-submission
```

### 2. Obtain the runtime artifacts

Download the **saved CPU and GPU models**, with checksums and upstream licence notices,
from the private [runtime-models-v1 Release](https://github.com/RohanR2501/signaldesk-bc3415-submission/releases/tag/runtime-models-v1):

```powershell
gh release download runtime-models-v1 --repo RohanR2501/signaldesk-bc3415-submission --pattern SignalDesk_Runtime_Models.zip --dir .runtime
Expand-Archive -LiteralPath .runtime/SignalDesk_Runtime_Models.zip -DestinationPath .
```

Extract into the repository root. Do not use `-Force`: existing artifacts should not
be silently overwritten. The bundle supplies only the two runtime models, their run
metadata, a checksum manifest and licence notices. No retraining is needed.

**Historical data is not distributed through GitHub.** Permission to redistribute
Amazon review text has not been established. Until an authorised handoff is arranged,
use the existing project machine for the populated Overview and Workbench. A GitHub
clone with the model bundle can run the Analyzer, but cannot reproduce those populated
historical views by itself.

| Capability | Required paths relative to the repository root | Total size |
|---|---|---:|
| CPU fallback | `artifacts/sentiment/baselines-v1/{run.json,logistic_regression_balanced.joblib}` | 2.00 MiB |
| CUDA sentiment | `artifacts/sentiment/distilbert-weighted-v1/run.json` and `checkpoint/{class_mapping.json,config.json,model.safetensors,tokenizer.json,tokenizer_config.json}` | 256.12 MiB |
| Historical views | `data/processed/appliances-v1/{investigation.parquet,manifest.json}` plus the two exclusion metadata files below | 4.91 MiB |

The required exclusion files are `data/annotations/sentiment-v1/internal/provenance.json`
and `data/annotations/sentiment-v1/complaints-v1/internal/provenance.json`. They contain
reservation metadata, not completed annotation CSVs, and must match the frozen records.
Their distribution also requires authorisation. The exact **12 paths, byte sizes and
SHA256 checksums** are in [config/runtime-artifacts.json](config/runtime-artifacts.json).
Do not supply protected annotations, training/test partitions or the unused BART model.

### 3. Install dependencies

For a clone with the model bundle and no authorised historical files:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -AnalyzerOnly
```

When all authorised historical files are already in place, omit `-AnalyzerOnly` for
full runtime verification. Setup checks artifact sizes/checksums before installing
locked dependencies using `uv.lock` and `frontend/package-lock.json`. It preserves
configuration and model files; defaults work without creating `.env` files.
Use `-VerifyOnly` to check artifacts without installing anything.

### 4. Start the application

In terminal 1, from the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\start.ps1 Backend
```

In terminal 2, from the same directory:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\start.ps1 Frontend
```

Open **[http://127.0.0.1:5173](http://127.0.0.1:5173)**.
API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
Stop each foreground service with **Ctrl+C**. The helpers refuse occupied ports and
never terminate existing services.

**Without compatible CUDA hardware:** add `-CpuOnly` to setup and backend startup.
This omits GPU dependencies; the existing backend automatically falls back to saved
balanced logistic regression. It does not run DistilBERT on CPU or change model selection.
First model initialisation may be slow. Missing sentiment artifacts produce explicit
partial/unavailable results; missing historical files produce unavailable historical views.

## Application Features

- **Executive Overview:** historical volume, negative-rating burden, complaint scan
  coverage, product search and investigation priorities.
- **Live Review Analyzer:** single-review or UTF-8 CSV analysis (`text` required,
  `review_id` optional), up to **100 reviews / 2 MiB**, processed sequentially. Includes
  actual model identity, five independent complaint results, exact excerpts, cancellation,
  filters and spreadsheet-safe CSV export. Original text stays in temporary browser memory.
  Other / Uncategorised Feedback flags successful negative sentiment with five completed
  scans and no complaint evidence; it is a manual-review hint, not a sixth category.
- **Product Investigation Workbench:** product and candidate filters, pagination,
  monthly/quarterly comparisons and original source evidence. The frozen history contains
  **22,838 reviews, 80 products and 38 candidates**, ending **11 September 2023**.
  Version 2 rules match 1,491 unique reviews among 20,282 scanned bodies.

## Machine Learning

Preprocessing validates text, ratings and UTC timestamps, removes reviewer identities
and images, and keeps duplicate text families together in chronological partitions.
Only review-body `model_text` enters sentiment classifiers. Training labels are rating
proxies: **1–2 stars negative, 3 neutral, 4–5 positive**.

Classical baselines compare majority, TF-IDF Naive Bayes, logistic regression and balanced
logistic regression. Two two-epoch fine-tuned DistilBERT runs compare unweighted loss
with training-only square-root inverse-frequency weighting. **Weighted DistilBERT** is
the provisional GPU model; **balanced LR** is the saved CPU fallback. Macro-F1 gives
each sentiment class equal weight during selection.

| Model | Validation accuracy | Macro-F1 |
|---|---:|---:|
| Balanced Logistic Regression | 82.65% | 0.6269 |
| Unweighted DistilBERT | 89.84% | 0.6669 |
| Weighted DistilBERT | 87.97% | 0.6756 |

These frozen comparisons use **6,673 rating-proxy validation reviews** after excluding
240 reserved annotation records. The **6,921-review test set remains sealed**. Separate
LLM-reference comparisons and 20 selected human judgments are descriptive, not estimates
of population accuracy. Evaluation is closed; setup runs no training or evaluation.

`complaint-rules-v2` independently detects **product defect, arrival damage, wrong item,
delivery issue and service/returns**. It preserves exact Unicode offsets, negation and
reported-resolution context. Positive sentiment never erases complaint evidence. No match
means unknown, not confirmed absence; v2 coverage is not independently validated and
frozen v1 comparisons do not validate v2. Historical candidates are not verified incidents.

## Technical Details

React/strict TypeScript/Tailwind calls one FastAPI backend through `/api/v1`. Saved models
load lazily and are reused; historical intelligence uses a cached Polars snapshot with
traceable denominators and minimum-support rules. No Docker, hosted inference or paid API
is needed. Runtime artifacts remain outside ordinary Git history.

```text
backend/src/signaldesk/   API, inference orchestration and historical intelligence
frontend/src/            Three views, typed client, batch processing and evidence UI
ml/signaldesk_data/      Acquisition, preprocessing and duplicate-aware partitions
ml/signaldesk_ml/        Training, model selection, inference and evaluation
config/                  Frozen experiment settings and runtime checksum inventory
docs/                    Canonical API schema and frozen complaint taxonomy
scripts/                 Model-cache, GPU and contract-export utilities
tests/; frontend/tests/  Focused backend and CSV/triage tests
setup.ps1; start.ps1     Native Windows setup and foreground service startup
```

[Canonical contracts](backend/src/signaldesk/contracts.py) generate the shared API schemas.
Offline research code is retained for grading; reproducing experiments requires authorised
source data and frozen memberships, not application setup. Source alone cannot recreate
private judgments or guarantee identical models.

Data: McAuley Lab, [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/),
Hou et al. (2024), *Bridging Language and Items for Retrieval and Recommendation*.
The [DistilBERT base model](https://huggingface.co/distilbert/distilbert-base-uncased)
is Apache-2.0 licensed; the model bundle retains its licence and identifies our fine-tuning.
Saved LR parameters/vocabulary and run metadata are project-created; no original review
bodies or annotation files are included in that bundle. Upstream model licensing does
not establish permission to redistribute Amazon reviews. Only load trusted joblib files.

## Troubleshooting

| Problem | Action |
|---|---|
| Private repository or Release returns 404 | Sign into `gh` with an account granted repository access. |
| Missing artifacts or checksum mismatch | Check `config/runtime-artifacts.json`; obtain the matching frozen files. Nothing is regenerated. Use `-AnalyzerOnly` only when historical views are intentionally unavailable. |
| Port 8000 or 5173 occupied | Use the existing service or stop it in its own terminal, then retry. |
| Sentiment unavailable | Check model paths/dependencies; restart backend after correcting them because load failures are cached. `.env.example` lists path overrides. |
| Historical views unavailable | The four historical files are missing, incompatible or failed integrity checks; the model bundle does not contain them. |
| Health is connected but analysis is unavailable | Health checks liveness, not model/data readiness. Inspect the analysis response. |
