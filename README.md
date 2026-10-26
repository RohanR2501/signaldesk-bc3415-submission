# SignalDesk

SignalDesk helps retail analysts turn appliance reviews into evidence for investigation.
It combines saved sentiment models, independent complaint rules and historical product
analytics in three views. Developed for NTU BC3415 Option A (Text Classification).

## Quick Start

### 1. Prerequisites and clone

Use **Windows 10/11 x64, Python 3.13 (with the `py` launcher), Node.js 24 and Git**.
Setup uses the `curl.exe` included with current Windows installations.
The default setup preserves **weighted DistilBERT on NVIDIA CUDA**; the reference
hardware is an RTX 4070 Laptop GPU with 8 GB VRAM and a compatible NVIDIA driver.
The locked PyTorch wheel includes CUDA 12.6 runtime support; a separate CUDA Toolkit
is not required. Setup installs pinned **uv 0.13.5** in a separate `.tools` environment.
GitHub CLI and a GitHub account are not required for the public repository or model download.

```powershell
git clone https://github.com/RohanR2501/signaldesk-bc3415-submission.git
cd signaldesk-bc3415-submission
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

### 2. Runtime availability

Setup downloads the existing public [runtime-models-v1 Release](https://github.com/RohanR2501/signaldesk-bc3415-submission/releases/tag/runtime-models-v1)
when required models are missing, verifies the pinned archive SHA256 and each saved
model, and installs missing files with licence notices. Valid existing files are reused;
conflicting files stop setup without replacement. The 238 MiB archive is cached in
`.runtime` for interrupted installations; subsequent complete installations do not download it.
No training, model conversion or dataset preparation runs during setup.

**PARTIALLY REPRODUCIBLE from a public clone.** Single-review and batch CSV analysis
use the saved real models and independent complaint-rules-v2. The populated Executive
Overview and Product Investigation Workbench require four historical files that are
not publicly distributed. Setup reports this limitation and installs Analyzer dependencies;
it does not claim the complete historical application is ready.

| Capability | Required paths relative to the repository root | Total size |
|---|---|---:|
| CPU fallback | `artifacts/sentiment/baselines-v1/{run.json,logistic_regression_balanced.joblib}` | 2.00 MiB |
| CUDA sentiment | `artifacts/sentiment/distilbert-weighted-v1/run.json` and `checkpoint/{class_mapping.json,config.json,model.safetensors,tokenizer.json,tokenizer_config.json}` | 256.12 MiB |
| Historical views | `data/processed/appliances-v1/{investigation.parquet,manifest.json}` plus the two exclusion metadata files below | 4.91 MiB |

The exclusion files are `data/annotations/sentiment-v1/internal/provenance.json` and
`data/annotations/sentiment-v1/complaints-v1/internal/provenance.json`. They contain
frozen reservation metadata, not completed annotation CSVs. All **12 runtime paths,
byte sizes and SHA256 checksums**, plus the public model archive checksum, are in
[config/runtime-artifacts.json](config/runtime-artifacts.json).

### 3. Historical reproduction boundary

The [official provider](https://amazon-reviews-2023.github.io/) still offers the Appliances
source (270,209,794 compressed bytes, checked by HEAD only). Existing preparation uses
seed 3415 and deterministic product selection, but is a research pipeline: it scans the
full category, constructs the 50,000-row reservoir and duplicate families, writes all
sentiment partitions, and verifies the sealed test. **Do not run it for application setup.**
The frozen history also depends on those family assignments and the main/optional
annotation reservations. Acquisition/manifest timestamps change output hashes, and
reservation generation depends on earlier research files and rules; the current code
is not a standalone restorer of the four original runtime files. No exact reconstruction
from a fresh download was executed or verified. No source reviews were downloaded or
reprocessed during this deployment audit.

Amazon review redistribution permission remains unresolved: the
[maintainers do not assign a dataset licence](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/discussions/1).
For the same **22,838 reviews, 80 products and 38 investigation candidates**, use the
original project computer. If an authorised private handoff is established, copy only
the four matching historical files to the paths above, then run `setup.ps1 -VerifyOnly`.
Do not publish review bodies, protected annotations or research partitions. This handoff
requires permission; it is not supplied by the model Release or implied by public access.

`setup.ps1 -VerifyOnly` checks all required artifacts without installing or downloading;
missing historical files fail that check. Add `-AnalyzerOnly` to verify/install only the
Analyzer artifacts. Normal setup checks any present historical files and stops on a
checksum mismatch. Existing configuration is preserved; defaults need no `.env` files.

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
Stop each foreground service with **Ctrl+C**. Stop the services before rerunning setup;
Windows can lock frontend dependencies while Vite is running. The helpers refuse occupied
ports and never terminate existing services.

**Without compatible CUDA hardware:** add `-CpuOnly` to setup and backend startup.
This omits GPU dependencies; the existing backend automatically falls back to saved
balanced logistic regression. It does not run DistilBERT on CPU or change model selection.
First model initialisation can take several minutes (206 seconds in the fresh-clone
check); subsequent requests reuse the loaded model. Missing sentiment artifacts produce explicit
partial/unavailable results; missing historical files produce unavailable historical views.

Verified on **26 October 2026**, Windows / Python 3.13 / Node.js 24: a temporary public
clone downloaded the Release anonymously, passed archive/model checksums, installed
74 locked Python packages and 199 frontend packages, and started both helpers on
ports 8000/5173. The released weighted model produced genuine `cuda:0` inference;
disabling CUDA in a separate process used saved balanced LR. Repeated setup with
services stopped reused all eight model files and the archive without changing bytes
or timestamps. `-VerifyOnly`, missing-file and tamper/no-overwrite checks passed.
Fresh history returned unavailable (503); the original local files separately reproduced
22,838 reviews / 80 products / 38 candidates and 182 exact evidence excerpts. No dataset
rebuild, training, frozen evaluation or large validation suite was run.

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
not establish permission to redistribute Amazon reviews or settle rights in models trained
on them; downstream dataset terms remain an unresolved licensing concern. The Release
contains no review bodies or annotation judgments. Only load trusted joblib files.

## Troubleshooting

| Problem | Action |
|---|---|
| Public model download fails | Check GitHub/network access and retry; no login is required. Inspect any reported cache mismatch before removing that cached archive. |
| Artifact checksum mismatch | Restore the matching frozen file from a trusted source; setup refuses to overwrite it. Historical files require an authorised handoff. |
| Port 8000 or 5173 occupied | Use the existing service or stop it in its own terminal, then retry. |
| Sentiment unavailable | Check model paths/dependencies; restart backend after correcting them because load failures are cached. `.env.example` lists path overrides. |
| Historical views unavailable | The four historical files are missing, incompatible or failed integrity checks; the model bundle does not contain them. |
| Health is connected but analysis is unavailable | Health checks liveness, not model/data readiness. Inspect the analysis response. |
