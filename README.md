# Toulmin-VLM-Driving

Fine-tuning **Cosmos-Reason2-8B** (Qwen3-VL, 8B) to predict pedestrian crossing intention from driving video using a structured **Toulmin argument** format.

Every model prediction is exactly three lines:
```
grounds: <concrete visual observations about the target pedestrian>
warrant: <general principle linking observations to the conclusion>
answer: yes | no
```

The central hypothesis is that forcing the model to surface an explicit warrant (a bridging principle) causes it to draw on a learned world model rather than pattern-matching on raw visual features. Two post-hoc experiments — a warrant signal test and a counterfactual flip test — probe whether the warrant carries decision-relevant information.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [End-to-End Pipeline](#end-to-end-pipeline)
   - [Step 1 — Generate video clips](#step-1--generate-video-clips)
   - [Step 2 — Balanced sampling](#step-2--balanced-sampling)
   - [Step 3 — Toulmin debate annotation](#step-3--toulmin-debate-annotation-8-agent)
   - [Step 4 — SC CoT baseline](#step-4--sc-cot-baseline-matched-compute)
   - [Step 5 — Build training splits](#step-5--build-training-splits)
   - [Step 6 — Build test evaluation set](#step-6--build-test-evaluation-set)
5. [Training](#training)
6. [Inference & Evaluation](#inference--evaluation)
7. [Analysis](#analysis)
8. [Configuration Reference](#configuration-reference)
9. [File Reference](#file-reference)

---

## Repository Structure

```
toulmin-vlm-driving/
│
├── PSI_change/json_mode_90/          # Data pipeline (90-frame window)
│   ├── step_1_make_dataset.py        # PSI raw → 90-frame mp4 clips + JSONL
│   ├── select_1000.py                # Balanced 1000-record sample
│   ├── orchestrator_toulmin_v2.py    # ★ 8-agent dual-warrant debate pipeline
│   ├── sc_cot_baseline_pipeline.py   # SC CoT baseline (N=6, matched compute)
│   ├── build_sc_cot_canonical.py     # Canonicalize SC CoT outputs
│   ├── build_sft_dpo_datasets.py     # Build all train/val splits (v3)
│   ├── build_test_eval_v2.py         # Build canonical test set
│   ├── check_quality.py              # GPT quality checker
│   ├── filter_and_stats.py           # Filter by quality flag
│   ├── make_grpo_dataset.py          # GRPO set from quality failures
│   └── [legacy pipelines]            # see §File Reference
│
├── configs/                          # YAML hyperparameter configs
│   ├── train_sft_v3.yaml             # Toulmin-SFT v3
│   ├── train_sc_cot_v3.yaml          # SC CoT-SFT v3
│   ├── train_ipo_v3.yaml             # Toulmin-IPO v3
│   ├── train_kto_v3.yaml             # Toulmin-KTO v3
│   ├── infer_toulmin_sft_v3.yaml     # Inference config — Toulmin SFT
│   ├── infer_sc_cot_sft_v3.yaml      # Inference config — SC CoT
│   ├── infer_toulmin_ipo_v3.yaml     # Inference config — IPO
│   └── infer_toulmin_kto_v3.yaml     # Inference config — KTO
│
├── train_sft.py                      # SFT trainer (video LoRA, curriculum)
├── train_ipo.py                      # IPO preference trainer
├── train_kto.py                      # KTO preference trainer
│
├── infer_eval.py                     # Inference + accuracy / Brier metrics
├── analyze_warrant_signal.py         # SBERT warrant-vs-grounds signal test
├── warrant_flip_experiment.py        # Counterfactual warrant flip test
├── check_acc.py                      # Quick per-model accuracy summary
├── export_paper_metrics.py           # Aggregate metrics across all models
│
├── setup_conda_env.sh                # One-command conda environment setup
└── CLAUDE.md                         # Codebase notes and known issues
```

**Data and checkpoints are not tracked** (see `.gitignore`). Video clips, JSONL datasets, and LoRA adapter weights must be generated or obtained separately.

---

## Requirements

### Hardware

| Component | Minimum | Used in this project |
|-----------|---------|----------------------|
| GPU | 24 GB VRAM | NVIDIA RTX 6000 Ada (48 GB) |
| CPU RAM | 32 GB | — |
| Disk | 100 GB free | ~400 GB with all PSI videos |

All training scripts target a **single GPU**. Set `CUDA_VISIBLE_DEVICES=1` (or `=0`) before each command.

### Software

| Library | Version | Role |
|---------|---------|------|
| Python | 3.10 | — |
| PyTorch | 2.4.0 (CUDA 12.4) | Training backend |
| `transformers` | ≥ 4.45 | Qwen3-VL model + processor |
| `peft` | ≥ 0.13 | LoRA adapter |
| `trl` | ≥ 0.11 | SFTTrainer utilities |
| `bitsandbytes` | latest | 4-bit NF4 quantization |
| `qwen_vl_utils` | — | Video frame processing for Qwen-VL |
| `flash-attn` | ≥ 2 | Attention kernel (recommended) |
| `decord` | latest | Fast video decoding |
| `sentence-transformers` | latest | SBERT for analysis scripts |
| `scikit-learn` | latest | Logistic regression, stratified CV |
| `opencv-python` | latest | Frame extraction, bbox overlay |
| `ffmpeg` | system | Video encoding |
| `openai` | latest | GPT agents in data pipeline |
| `google-generativeai` | latest | Gemini agents in data pipeline |

### API Keys (data pipeline only)

The data pipeline (`orchestrator_toulmin_v2.py`, `sc_cot_baseline_pipeline.py`) requires:

```bash
export GEMINI_API_KEY=<your-key>   # Google AI Studio
export OPENAI_API_KEY=<your-key>   # OpenAI
```

Training and inference do **not** require any API keys.

---

## Installation

```bash
# Clone
git clone https://github.com/wszsycn/toulmin-vlm-driving.git
cd toulmin-vlm-driving

# Create conda environment (installs all dependencies)
bash setup_conda_env.sh
conda activate cosmos2
```

The script installs PyTorch 2.4.0 with CUDA 12.4, all Python packages, and the system `ffmpeg` codec. Edit `setup_conda_env.sh` if you need a different CUDA version.

> **Note:** All data paths in scripts are prefixed `/workspace/` (Docker container mount). If running outside Docker, replace `/workspace/` with the absolute path of this repository.

---

## End-to-End Pipeline

The full pipeline produces training datasets from the raw [PSI 2.0](https://github.com/PSI-Intention2022/PSI-Dataset) dataset. Steps 1–6 are **one-time preprocessing** — once the outputs exist you can jump straight to [Training](#training).

### Step 1 — Generate video clips

Slides a 90-frame window (step=45) over each pedestrian track, draws a green bounding box overlay, and encodes mp4 clips at 15 fps.

```bash
python PSI_change/json_mode_90/step_1_make_dataset.py
```

**Input:** PSI 2.0 `frames/` directory and `annotations/*/pedestrian_intent.json`  
**Output:** `PSI_change/json_mode_90/videos/*.mp4` + `psi_90f_raw.jsonl`

Edit `FRAME_DIR`, `ANNOT_DIR`, and `OUTPUT_VIDEO` at the top of the script to point to your PSI data.

---

### Step 2 — Balanced sampling

Filters ambiguous windows (`|agg_intent − 0.5| ≤ 0.2`), caps five records per track, and draws a balanced 50/50 yes/no sample of 1000 records.

```bash
python PSI_change/json_mode_90/select_1000.py
```

**Input:** `psi_90f_raw.jsonl`  
**Output:** `psi_90f_1000.jsonl`

---

### Step 3 — Toulmin debate annotation (8-agent)

The main data-generation pipeline. For each record it:
1. **Agent A** (Gemini) — pedestrian motion analysis
2. **Agent B** (Gemini) — scene context analysis
3. **Agent C** (Gemini) — synthesises grounds (never sees the GT answer)
4. **Agent D-yes / D-no** (Gemini) — independent YES and NO warrants
5. **Agent E** (Gemini) — debate judge; picks winner + confidence margin
6. **Agent F** (GPT) — format/consistency critic
7. **Orchestrator** (GPT) — retries failed agents (max 3 rounds)

```bash
export GEMINI_API_KEY=<your-key>
export OPENAI_API_KEY=<your-key>
python PSI_change/json_mode_90/orchestrator_toulmin_v2.py
```

**Input:** `psi_90f_1000.jsonl` + video files  
**Output:** `trf_train_v3/psi_toulmin_orchestrator_v2_1000.jsonl`

The pipeline is resumable — it skips records already present in the output file. All LLM calls are cached under `agent_cache_orch_v2/`.

---

### Step 4 — SC CoT baseline (matched compute)

Generates N=6 self-consistency chain-of-thought completions per record using Gemini, using the same total number of LLM calls as the Toulmin pipeline (matched compute for fair comparison).

```bash
export GEMINI_API_KEY=<your-key>
python PSI_change/json_mode_90/sc_cot_baseline_pipeline.py

# Canonicalize (select majority-vote completion as the single training target)
python PSI_change/json_mode_90/build_sc_cot_canonical.py
```

**Input:** `psi_90f_1000.jsonl` + video files  
**Output:** `trf_train_v3/psi_sc_cot_canonical_1000.jsonl`

---

### Step 5 — Build training splits

Applies Tricks 1–3 and 5, builds the matched intersection set (593 records in both Toulmin and SC CoT pools), and writes all train/val splits for SFT and DPO/IPO/KTO.

```bash
python PSI_change/json_mode_90/build_sft_dpo_datasets.py
```

**Input:** `psi_toulmin_orchestrator_v2_1000.jsonl` + `psi_sc_cot_canonical_1000.jsonl`  
**Output** (all under `trf_train_v3/`):

| File | Records | Purpose |
|------|---------|---------|
| `psi_sft_toulmin_matched_train.jsonl` | 536 | Toulmin SFT training |
| `psi_sft_toulmin_matched_val.jsonl` | 57 | Toulmin SFT validation |
| `psi_sft_sc_cot_matched_train.jsonl` | 552 | SC CoT SFT training |
| `psi_sft_sc_cot_matched_val.jsonl` | 41 | SC CoT SFT validation |
| `psi_dpo_toulmin_matched_train.jsonl` | 423 | IPO / KTO training pairs |
| `psi_dpo_toulmin_matched_val.jsonl` | 44 | IPO / KTO validation pairs |

**Data tricks applied:**
- **Trick 1** — asymmetric margin filter (yes-winner ≥ 0.4, no-winner ≥ 0.7)
- **Trick 2** — hard negative mining (debate-loser warrant appended with `_hardneg` id)
- **Trick 3** — curriculum stage field (stage 1 = high-confidence, stage 2 = remainder)
- **Trick 5** — per-record `sample_weight` from annotator agreement

---

### Step 6 — Build test evaluation set

Builds the canonical held-out test set from PSI test annotations.

```bash
python PSI_change/json_mode_90/build_test_eval_v2.py
# Smoke-check with first 5 records:
python PSI_change/json_mode_90/build_test_eval_v2.py --smoke
```

**Output:** `psi_test_eval_v2.jsonl` (572 records, multi-annotator aggregated intent)

---

## Training

All training scripts read hyperparameters exclusively from a YAML config file. Use `--smoke` to validate the pipeline on 5 steps before committing to a full run.

### Toulmin SFT (train first — IPO/KTO require this checkpoint)

```bash
# Smoke test (5 steps, ~7 min)
CUDA_VISIBLE_DEVICES=1 python train_sft.py \
    --config configs/train_sft_v3.yaml --smoke

# Full training (~3 epochs, ~6–8 hours)
CUDA_VISIBLE_DEVICES=1 python train_sft.py \
    --config configs/train_sft_v3.yaml
```

**Output:** `checkpoints/toulmin_sft_v3/` (LoRA adapter, `adapter_config.json`)

### SC CoT SFT (parallel to Toulmin SFT — independent checkpoint)

```bash
CUDA_VISIBLE_DEVICES=1 python train_sft.py \
    --config configs/train_sc_cot_v3.yaml
```

**Output:** `checkpoints/sc_cot_sft_v3/`

### IPO — Identity Preference Optimization

Requires `checkpoints/toulmin_sft_v3/` to exist.

```bash
# Smoke test
CUDA_VISIBLE_DEVICES=1 python train_ipo.py \
    --config configs/train_ipo_v3.yaml --smoke

# Full (1 epoch, ~2–3 hours)
CUDA_VISIBLE_DEVICES=1 python train_ipo.py \
    --config configs/train_ipo_v3.yaml
```

**Output:** `checkpoints/toulmin_ipo_v3/`

### KTO — Kahneman-Tversky Optimization

Requires `checkpoints/toulmin_sft_v3/` to exist.

```bash
CUDA_VISIBLE_DEVICES=1 python train_kto.py \
    --config configs/train_kto_v3.yaml --smoke

CUDA_VISIBLE_DEVICES=1 python train_kto.py \
    --config configs/train_kto_v3.yaml
```

**Output:** `checkpoints/toulmin_kto_v3/`

### Key training hyperparameters (v3)

| Config | LR | Epochs | LoRA rank | Early stopping |
|--------|----|--------|-----------|----------------|
| `train_sft_v3.yaml` | 1e-4 | 3 | 64 | patience=2 |
| `train_sc_cot_v3.yaml` | 1e-4 | 3 | 64 | patience=2 |
| `train_ipo_v3.yaml` | 2e-6 | 1 | 64 | — |
| `train_kto_v3.yaml` | 3e-6 | 1 | 64 | — |

---

## Inference & Evaluation

```bash
# Toulmin SFT
CUDA_VISIBLE_DEVICES=1 python infer_eval.py \
    --config configs/infer_toulmin_sft_v3.yaml

# SC CoT SFT
CUDA_VISIBLE_DEVICES=1 python infer_eval.py \
    --config configs/infer_sc_cot_sft_v3.yaml

# IPO
CUDA_VISIBLE_DEVICES=1 python infer_eval.py \
    --config configs/infer_toulmin_ipo_v3.yaml

# KTO
CUDA_VISIBLE_DEVICES=1 python infer_eval.py \
    --config configs/infer_toulmin_kto_v3.yaml

# Smoke test any config (first 5 records)
CUDA_VISIBLE_DEVICES=1 python infer_eval.py \
    --config configs/infer_toulmin_sft_v3.yaml --smoke
```

**Input:** `psi_test_eval_v2.jsonl` (572 records) + adapter checkpoint  
**Output per run:**
- `predictions/<model>_predictions.jsonl` — per-record predictions
- `predictions/<model>_metrics.json` — accuracy, balanced accuracy, Brier score, ROC-AUC

Inference is **resumable**: records already written to the output JSONL are skipped on restart.

### Prediction record format

```json
{
  "id": "video_0001_track_0_00135-00224",
  "video": "/workspace/.../video.mp4",
  "answer_hard": "yes",
  "answer_soft": 0.85,
  "predicted_hard": "yes",
  "predicted_prob": 0.92,
  "raw_samples": ["grounds: ...\nwarrant: ...\nanswer: yes"],
  "parse_status": "1/1_parsed",
  "meta": { "video_id": "...", "track_id": "...", "frame_span": [...] }
}
```

### Quick accuracy summary

```bash
# Print accuracy + per-class recall for one model
python check_acc.py toulmin_sft_v3

# Aggregate all models for paper table
python export_paper_metrics.py
```

---

## Analysis

### Warrant signal test (Metric 1 & 5)

Tests whether the warrant segment carries decision-relevant information beyond the grounds, using SBERT-feature logistic regression and yes/no centroid separation.

```bash
python analyze_warrant_signal.py \
    --pred-dir PSI_change/json_mode_90/predictions \
    --out-json PSI_change/json_mode_90/predictions/warrant_signal.json
```

**Output:** prints two tables (logistic regression accuracy and centroid distance per segment) and writes `warrant_signal.json`. Requires `sentence-transformers` and `scikit-learn`.

**Interpretation:**
- `delta(warrant_acc − grounds_acc) > 0` → warrant carries signal beyond grounds (supports world-model framing)
- `warrant_dist > full_dist` and `warrant_dist > 0.10` → warrant differentiates yes/no responses

### Counterfactual flip experiment

Measures whether replacing the warrant with an opposite-class warrant causes the model to flip its answer more often than a random warrant swap does.

```bash
# Smoke test (5 samples, ~5 min)
CUDA_VISIBLE_DEVICES=1 python warrant_flip_experiment.py \
    --target-model toulmin_sft_v3 \
    --pred-dir PSI_change/json_mode_90/predictions \
    --model-ckpt checkpoints/toulmin_sft_v3 \
    --n-samples 5

# Full run (100 stratified samples, ~2 hours)
CUDA_VISIBLE_DEVICES=1 python warrant_flip_experiment.py \
    --target-model toulmin_sft_v3 \
    --pred-dir PSI_change/json_mode_90/predictions \
    --model-ckpt checkpoints/toulmin_sft_v3 \
    --n-samples 100 \
    --out-json PSI_change/json_mode_90/predictions/warrant_flip.json
```

**Decision rule:**
- `flip_rate_c > 0.30` AND `flip_rate_c > 2 × flip_rate_b` → **SUPPORTED** (warrant encodes directional decision signal)
- `flip_rate_c ≤ flip_rate_b` → **CONTRADICTED** (opposite warrant is no more disruptive than random)
- otherwise → **MIXED**

---

## Configuration Reference

All configs live in `configs/`. Training and inference configs share the same structure:

```yaml
model_id:      nvidia/Cosmos-Reason2-8B
adapter_path:  /workspace/checkpoints/toulmin_sft_v3   # inference only
output_dir:    /workspace/checkpoints/toulmin_sft_v3   # training only

# Data
train_path:    /workspace/PSI_change/json_mode_90/trf_train_v3/psi_sft_toulmin_matched_train.jsonl
val_path:      /workspace/PSI_change/json_mode_90/trf_train_v3/psi_sft_toulmin_matched_val.jsonl

# Hyperparameters
num_train_epochs:            3
learning_rate:               1.0e-4
gradient_accumulation_steps: 8
warmup_ratio:                0.05
lora_dropout:                0.1
eval_steps:                  100
early_stopping_patience:     2
early_stopping_threshold:    0.005

# Video decoding (must match between training and inference)
num_video_frames:     90
frame_size:           112
max_pixels_per_frame: 12544
```

---

## File Reference

### v3 pipeline (active)

| File | Purpose |
|------|---------|
| `PSI_change/json_mode_90/step_1_make_dataset.py` | Raw PSI → 90-frame mp4 clips + `psi_90f_raw.jsonl` |
| `PSI_change/json_mode_90/select_1000.py` | Balanced 1000-record sample with intent filter |
| `PSI_change/json_mode_90/orchestrator_toulmin_v2.py` | 8-agent dual-warrant debate pipeline |
| `PSI_change/json_mode_90/sc_cot_baseline_pipeline.py` | SC CoT baseline, N=6 Gemini samples |
| `PSI_change/json_mode_90/build_sc_cot_canonical.py` | Majority-vote canonicalization of SC CoT |
| `PSI_change/json_mode_90/build_sft_dpo_datasets.py` | All train/val splits with tricks 1/2/3/5 |
| `PSI_change/json_mode_90/build_test_eval_v2.py` | Canonical test set (572 records) |
| `train_sft.py` | SFT with curriculum, sample weights, early stopping |
| `train_ipo.py` | IPO preference trainer (builds on SFT checkpoint) |
| `train_kto.py` | KTO preference trainer (builds on SFT checkpoint) |
| `infer_eval.py` | Inference + metrics on `psi_test_eval_v2.jsonl` |
| `analyze_warrant_signal.py` | SBERT logistic regression + centroid separation |
| `warrant_flip_experiment.py` | Counterfactual warrant swap experiment |
| `check_acc.py` | Quick accuracy/recall summary |
| `export_paper_metrics.py` | Aggregate metrics across all models |
| `configs/train_sft_v3.yaml` | Toulmin-SFT v3 hyperparameters |
| `configs/train_sc_cot_v3.yaml` | SC CoT-SFT v3 hyperparameters |
| `configs/train_ipo_v3.yaml` | IPO v3 hyperparameters |
| `configs/train_kto_v3.yaml` | KTO v3 hyperparameters |
| `configs/infer_toulmin_sft_v3.yaml` | Inference config — Toulmin SFT |
| `configs/infer_sc_cot_sft_v3.yaml` | Inference config — SC CoT |
| `configs/infer_toulmin_ipo_v3.yaml` | Inference config — IPO |
| `configs/infer_toulmin_kto_v3.yaml` | Inference config — KTO |

### Legacy / reference (not used for v3 results)

| File | Notes |
|------|-------|
| `PSI_change/json_mode_90/orchestrator_toulmin.py` | v1 5-agent pipeline (single warrant, answer leakage) |
| `PSI_change/json_mode_90/toulmin_agent.py` | 3-agent GPT-only pipeline |
| `PSI_change/json_mode_90/gemini_toulmin_pipeline.py` | 3-agent Gemini+GPT pipeline |
| `PSI_change/json_mode_90/cot_baseline_pipeline.py` | Single-sample CoT (N=1) |
| `PSI_change/json_mode_90/check_quality.py` | GPT quality checker for v1 pipeline |
| `PSI_change/json_mode_90/filter_and_stats.py` | Filter by quality flag (v1) |
| `PSI_change/json_mode_90/to_llava_format.py` | LLaVA format conversion (v1) |
| `PSI_change/json_mode_90/make_grpo_dataset.py` | GRPO candidates from quality failures |
| `PSI_change/json_mode_90/toulmin_agent_test20.py` | 3-agent run on first 20 test records |
| `train_psi_video_sft.py` | Original video SFT (hardcoded, no YAML) |
| `train_psi_multimage_sft.py` | Multi-image frame SFT (LoRA r=16) |
| `train_psi_grpo.py` | GRPO training with 7 reward functions |
| `train_psi_grpo_claude_new.py` | Cleaner GRPO variant (4 frames) |
| `train_psi_grpo_judge_claude.py` | GRPO with LLM judge reward |
| `train_dpo.py` | DPO (superseded by IPO/KTO) |
| `cosmos_video_compare_3models_binary.py` | Binary metric comparison script |
| `gemini_batch_video.py` | Gemini Flash inference baseline |
| `analyze_text.py` / `analyze_text_export.py` | Text-level output analysis |
| `regurgitation_test.py` | Test for training-set memorisation |
| `grpo_viewer.html` / `orchestrator_viewer.html` | Local HTML viewers for output inspection |
