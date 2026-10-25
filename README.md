# SENTINEL — Adaptive Industrial Intelligence

Sentinel is a Streamlit application for visual anomaly inspection, temporal triage, SOP-grounded quality reports, and an operator copilot.

## Repository status

- The Streamlit app is the supported frontend. Its image and video routes use the existing Python inference engine; the audit and copilot use the local RAG package.
- The five bundled images and conveyor clips are illustrative examples. They are not a held-out MVTec evaluation set.
- The active category calibration arrays and grid STFPM calibration are included under `assets/`. Other generated calibration/evaluation arrays, trained weights, and serialized model objects remain excluded from Git.
- Pipeline A live inference is unavailable until its ResNet-18/Mahalanobis export and trained Q-table are supplied. No PPO checkpoint is included; video triage uses the explicitly labeled adaptive fallback.
- The dashboard's batch telemetry is read from `outputs/`, which is git-ignored. Missing exports are reported as unavailable; scheduler risk tiers are never converted into fabricated detector scores. Supply an authorized scheduler export to enable held-out telemetry views.

## Quick start

Use Python 3.10–3.12. Install the app dependencies and start Streamlit:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

For live visual inference, obtain the trained bundle and place its `models/` directory beside `app.py`. The tracked `assets/` directory contains active calibration files and sample inputs. If the bundle lives elsewhere, set `MVTEC_PROJECT_DIR` to the directory containing both `models/` and `assets/`. See [MODEL_ASSETS.md](MODEL_ASSETS.md) for required files, validation guidance, and current distribution limitations:

```bash
export MVTEC_PROJECT_DIR="/path/to/sentinel-bundle"
```

The required Pipeline B files are `yolo26n.pt`, `wide_resnet50_2_patch_extractor.pth`, `stfpm_teacher_resnet18_grid.pth`, `stfpm_student_resnet18_grid.pth`, and per-category `yolo26_embedding_knn_<category>.joblib` and `patchcore_patch_bank_<category>.pt` files. When RL-assisted mode is selected, the temporal transformer and scaler are also needed. Files are listed in `model_manifest.json`.

Optional settings:

- `GROQ_API_KEY` enables the Groq-backed copilot. Without it, the local response fallback is used. Copy `.env.example` to `.env` for local configuration; never commit `.env`.
- `sentence-transformers` enables the dense embedding backend when the model is cached or can be downloaded. A built-in feature-hashing encoder is used otherwise.
- RAG startup is offline by default. Set `RAG_ALLOW_MODEL_DOWNLOAD=1` to allow downloading the optional embedding model when it is not cached.
- FFmpeg playback support is provided by `imageio-ffmpeg` from `requirements.txt`.

## Architecture

1. **Perception:** STFPM for grid; YOLO26 embeddings and PatchCore-style memory for the other supported categories; optional DINOv2/LinearSVM subtype prediction.
2. **Triage:** temporal model and PPO when their checkpoints are available. The current bundle lacks PPO weights, so the video route clearly labels and uses the adaptive fallback.
3. **Quality intelligence:** local SOP retrieval, shift audit reports, and copilot answers. Groq is optional.
4. **Workspaces:** inspection, quality audit, operator copilot, and model benchmark.

Scores are empirical percentiles against normal-reference calibration data, not probabilities of defects. Current bundled sample scores can saturate at 1.0 and should not be treated as operational performance evidence.

## Install optional dense RAG embeddings

```bash
python -m pip install 'sentence-transformers>=3.3,<7'
```

The first uncached load may download `sentence-transformers/all-MiniLM-L6-v2`; for an offline environment, pre-cache that model or use the built-in encoder fallback.

## Tests and checks

Run the lightweight regression checks and Python syntax checks with:

```bash
python -m unittest discover -s tests -v
python -m py_compile app.py inference.py inference2.py rag/*.py
python -m py_compile scripts/*.py
python scripts/check_git_safety.py
```

GitHub Actions runs these checks for pushes and pull requests. Before starting visual inference,
you can check a downloaded model bundle and its SHA-256 digests with
`python scripts/verify_model_bundle.py` (add `--with-rl` to require the temporal model files).

The MVTec training notebooks are research artifacts and document their own Colab workflow below.

## Before publishing

- No license is included. Choose and add a license only after confirming the project owners agree on its terms.
- Confirm that you may redistribute the bundled sample images and derived clips on a public repository.
- The trained model bundle is not in Git; arrange an authorized download location, publish checksums, and share it with repository users before advertising inference as ready to run.
- This personal repository is initialized from a clean project snapshot; upstream commit history and its contributor metadata are not copied. The original `origin` remote remains configured in this local checkout.

---

## Research notebook reference

## Step 1 – Vision in a nutshell

| Stage | What happens |
|---|---|
| Preprocessing | Resize to 224×224, ImageNet normalisation, histogram equalisation or Gaussian blur (configurable) |
| Features | Pretrained ResNet-18 (no training), last layer removed → 512-d embedding per image |
| Anomaly detection | kNN (cosine, k=5), Mahalanobis distance, One-Class SVM – trained on good images only |
| Evaluation | AUC, confusion matrix, recall, false-alarm rate per category and method |
| Defect type | Logistic regression predicts the defect type of defective images |

Categories used: `grid`, `zipper`, `hazelnut`, `leather`, `toothbrush`.

> **Thresholds:** two per category and method.
> - **Default:** the 95th percentile of the cross-validated good training scores (normalised score ≈ 0.95).
>   It uses no test labels, and about 5% of good parts are expected to be flagged.
> - **Comparison:** the threshold with the best F1 on the test set. It uses test labels, so its numbers are optimistic.
>
> Recall and false-alarm rate are reported for both thresholds.

## How to open the notebook in Colab

The notebook is designed for **Google Colab only** (works the same for Windows and macOS users).

1. Go to <https://colab.research.google.com> → **File → Open notebook → GitHub** tab.
2. Authorise GitHub access if the repository is private.
3. Open `notebooks/step1_vision.ipynb` from this repository in Colab.
4. *(Recommended)* **Runtime → Change runtime type → T4 GPU**. CPU also works, just slower.
5. **Runtime → Run all** and allow Google Drive access when asked.

To save your changes back to GitHub use **File → Save a copy in GitHub** (clear outputs first:
**Edit → Clear all outputs**).

All paths and settings are in the **config cell** at the top of the notebook.

## Google Drive folder layout

The dataset and all outputs live in Google Drive, never in this repository.

```
MyDrive/Integrated/
├── data/
│   └── mvtec/
│       ├── grid/
│       │   ├── train/good/
│       │   ├── test/good/
│       │   ├── test/<defect_type>/     e.g. bent, broken, ...
│       │   └── ground_truth/           (masks – not used)
│       ├── zipper/  hazelnut/  leather/  toothbrush/   (same layout)
└── outputs/
    ├── vision/                         ← created by Step 1 notebook
    │   ├── current_anomaly_score.csv   ← daily use
    │   ├── archive/
    │   │   └── anomaly_scores_old_vs_new_threshold.csv   ← for the report
    │   ├── metrics.json
    │   ├── summary_table.csv
    │   ├── method_choice.json
    │   ├── defect_type_metrics.json
    │   ├── defect_type_summary.csv
    │   ├── features/                   cached ResNet-18 embeddings (.npy)
    │   └── plots/                      PNG figures
    ├── scheduler_outputs/              ← created by Step 2 notebook
    │   ├── scheduler_inference_ready.csv
    │   ├── scheduler_evaluation.csv
    │   ├── scheduler_metadata.json
    │   ├── realistic_stream_metrics_by_seed.csv
    │   ├── realistic_stream_summary.csv
    │   ├── realistic_stream_per_category.csv
    │   ├── realistic_stream_decisions.csv
    │   ├── adaptation_demo_results.csv
    │   └── adaptation_demo_summary.csv
    └── rag_outputs/                    ← created by Step 3 (RAG)
        ├── shift_quality_audit_report.md
        └── rag_shift_summary.json
```

## Saved vision outputs

**Daily use:** `current_anomaly_score.csv`. **For the report:** the `archive/` folder.

### `current_anomaly_score.csv` (daily use)
One row per **test image and method** (3 methods → each image appears 3 times), at the default threshold.

| Column | Meaning |
|---|---|
| `category` | Product category, e.g. `hazelnut` |
| `image_path` | Path relative to `MyDrive/Integrated/data/mvtec/`, e.g. `hazelnut/test/crack/003.png` |
| `true_defect_type` | Folder name: `good` or the defect type |
| `label` | Ground truth: `0` = good, `1` = defective |
| `method` | `knn_cosine`, `mahalanobis` or `ocsvm` |
| `anomaly_score` | Higher = more likely defective (scales differ between methods) |
| `threshold` | **Default** threshold for this category and method: 95th percentile of good training scores |
| `prediction` | `1` if `anomaly_score >= threshold`, else `0` (**use this one**) |
| `predicted_defect_type` | Defect type predicted by the defect-type classifier; falls back to `true_defect_type` when no prediction exists (good images, toothbrush, or Section 6 switched off) |
| `normalized_score` | Same score on a common **0–1 scale for all methods**: share of good training images that look less unusual (0.97 = more unusual than 97% of good parts; 1.0 = more unusual than all of them) |
| `normalized_threshold` | The default threshold on that 0–1 scale (≈ 0.95) |

`prediction` is always based on the raw `anomaly_score` and `threshold`; the normalised columns are for
comparing methods and for display.

Tip: filter the `method` column to `mahalanobis`, the chosen method (see `method_choice.json`), to get one row per image.

### `archive/anomaly_scores_old_vs_new_threshold.csv` (for the report)
All 3 methods and both thresholds side by side, so the report can compare them:
- **new** = the 95th percentile of good training scores (the default, no test labels)
- **old** = the best-F1 threshold tuned on the test set (optimistic)

Same rows and columns as `current_anomaly_score.csv`, with these columns renamed or added:

| Column | Meaning |
|---|---|
| `threshold_new` / `prediction_new` / `normalized_threshold_new` | `threshold` / `prediction` / `normalized_threshold` from the daily file |
| `threshold_old` | Best-F1 threshold on the test set |
| `prediction_old` | `1` if `anomaly_score >= threshold_old`, else `0` |
| `normalized_threshold_old` | The old threshold on the 0–1 scale |

### `metrics.json`
```json
{
  "hazelnut": {
    "knn_cosine":  {"auc": 0.97, "recall": 0.93, "false_alarm_rate": 0.05, "f1": 0.94,
                    "tp": 65, "fp": 2, "tn": 38, "fn": 5,
                    "threshold": 0.12, "normalized_threshold": 0.95,
                    "recall_f1thr": 0.95, "false_alarm_rate_f1thr": 0.03, "f1_f1thr": 0.96,
                    "tp_f1thr": 67, "fp_f1thr": 1, "tn_f1thr": 39, "fn_f1thr": 3,
                    "threshold_f1thr": 0.13, "normalized_threshold_f1thr": 0.98},
    "mahalanobis": {...},
    "ocsvm":       {...}
  },
  "grid": {...}
}
```
(numbers above are only an illustration). **RL scheduling** should use `recall` (chance that a defect
is caught) and `false_alarm_rate` (chance that a good item is flagged). These keys, without a suffix, are at the
default 95th-percentile threshold. Keys ending in `_f1thr` are the same numbers at the best-F1 threshold, for comparison.

### `method_choice.json`
Which method we continue with and why: `chosen_method` (highest mean AUC), the default threshold rule, the best
method per category, and mean AUC / recall / false-alarm rate per method at both thresholds.

### `defect_type_metrics.json` / `defect_type_summary.csv` (Section 6)
Per category:
- `accuracy`: the average over both halves (train on A and test on B, then the reverse)
- the accuracy of each half (`accuracy_train_a_test_b`, `accuracy_train_b_test_a`)
- `random_guess_accuracy`: what random guessing would give (1 ÷ number of defect types)
- the number of defect types and the size of each half

Toothbrush is skipped.

### `summary_table.csv` and `plots/`
Summary per category and method, plus `preprocessing_examples.png`, `roc_curves.png`,
`confusion_matrices.png`, `score_distributions.png` (all methods on the common 0–1 scale) and
`defect_type_confusion.png`.

## Step 3 – NLP / RAG reports in a nutshell

Step 3 bridges the quantitative outputs of vision and RL to actionable factory knowledge:

| Component | Function |
|---|---|
| Manufacturing Knowledge Base | Standard Operating Procedures (SOPs) for `grid`, `zipper`, `hazelnut`, `leather`, and `toothbrush` detailing root causes, severities, rework protocols, and scrap criteria |
| Ingestion & Chunking | Semantic section parser with category and defect-code metadata enrichment |
| Vector Embeddings | Dense similarity vectors (`all-MiniLM-L6-v2`) with resilient built-in subword feature encoder fallback |
| Hybrid Retriever | Cosine similarity ranking with metadata filtering by product category and defect type |
| Quality Shift Reports | Ingests `scheduler_inference_ready.csv` and `adaptation_demo_results.csv`, calculates RL cost reductions vs. 100% inspection baseline, identifies defect clusters, and embeds official SOP remediation guidance |
| Operator Copilot | Interactive assistant explaining why parts were escalated by RL and prescribing specific repair/scrap procedures |

Outputs saved to `MyDrive/Integrated/outputs/rag_outputs/`:
- `shift_quality_audit_report.md`: Formal executive quality assurance and incident shift audit report.
- `rag_shift_summary.json`: JSON payload ready for ingestion by Step 4 (BI Dashboard).

## Repository layout

```
notebooks/
├── step1_vision.ipynb          ← Step 1: Vision anomaly detection
└── step3_rag_reports.ipynb     ← Step 3: NLP / RAG reports & copilot
step2_RL_Scheduler.ipynb        ← Step 2: RL inspection scheduling
rag/                            ← Core modular RAG package
├── knowledge_base/             ← Industrial SOPs (Markdown)
├── ingestion.py                ← Markdown chunking & metadata parser
├── embeddings.py               ← Dense and subword embedding models
├── vector_store.py             ← Persistent vector indexing & cosine search
├── retriever.py                ← Hybrid semantic & metadata retriever
├── telemetry_loader.py         ← Step 1 & 2 CSV loader
├── report_generator.py         ← Executive shift report generator
└── copilot.py                  ← Interactive technician copilot
requirements.txt
.gitignore
```

Do not commit datasets, generated output directories, `.npy` feature dumps, or `.env`. Review notebook outputs before publishing; clear any local paths or private data.

## Local logic check

Outside Colab, the config switches to `../local_test/` (relative to `notebooks/`, git-ignored).
Put a tiny dummy dataset with the MVTec layout in `local_test/data/mvtec/` to check that the
notebook runs end to end. Results on dummy data are meaningless.
