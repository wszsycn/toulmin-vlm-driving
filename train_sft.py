#!/usr/bin/env python3
"""
train_sft.py — Single-GPU SFT for Cosmos-Reason2-8B on PSI pedestrian intent.

All hyperparameters come from a YAML config file.  The training JSONL already
carries the canonical system/prompt/completion fields from the data pipeline —
no text re-rendering happens here.  The video is injected into the user turn
following Qwen3-VL's content-list convention.

Usage:
    python train_sft.py --config configs/toulmin_sft.yaml
    python train_sft.py --config configs/cot_sft.yaml --smoke
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    set_seed as hf_set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

try:
    from transformers import Qwen3VLForConditionalGeneration as _VLModelCls
except ImportError:
    from transformers import AutoModelForVision2Seq as _VLModelCls

from qwen_vl_utils import process_vision_info


# ============================================================
# Config
# ============================================================

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# Video loading with /tmp disk cache
# ============================================================

_VIDEO_CACHE_DIR = Path("/tmp/video_cache")
_VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _video_cache_path(video_path: str) -> Path:
    md5 = hashlib.md5(video_path.encode()).hexdigest()
    return _VIDEO_CACHE_DIR / f"{md5}.npy"


def _decode_video_decord(path: str, n_frames: int, frame_size: int) -> np.ndarray:
    """Decode n_frames uniformly sampled from video; resize to frame_size×frame_size."""
    import decord
    vr = decord.VideoReader(path, num_threads=2)
    total = len(vr)
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = vr.get_batch(indices.tolist()).asnumpy()  # (T, H, W, 3) uint8 RGB
    if frame_size != frames.shape[1] or frame_size != frames.shape[2]:
        frames = np.stack([
            np.array(Image.fromarray(frames[i]).resize(
                (frame_size, frame_size), Image.BILINEAR
            ))
            for i in range(len(frames))
        ])
    return frames  # (T, frame_size, frame_size, 3) uint8 RGB


def load_video_cached(path: str, n_frames: int = 90, frame_size: int = 112) -> np.ndarray:
    """
    Return (n_frames, frame_size, frame_size, 3) uint8 RGB array.
    Decodes on first call; subsequent calls load from /tmp/video_cache/.
    """
    cache = _video_cache_path(path)
    if cache.exists():
        return np.load(str(cache))
    frames = _decode_video_decord(path, n_frames, frame_size)
    np.save(str(cache), frames)
    return frames


# ============================================================
# Dataset
# ============================================================

class PSIDataset(TorchDataset):
    """
    Reads the LLaVA-format JSONL produced by build_sft_dpo_datasets.py.
    Fields used verbatim from each line:
      record["system"]               → system message string
      record["prompt"][0]["content"] → user message string
      record["completion"][0]["content"] → assistant response string
      record["video"]                → path to the .mp4 file
    No text re-rendering; the pipeline's canonical strings are used as-is.
    """

    def __init__(self, jsonl_path: str, config: dict):
        self.records: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                self.records.append({
                    "video_path":  r["video"],
                    "system":      r["system"],
                    "user_text":   r["prompt"][0]["content"],
                    "target_text": r["completion"][0]["content"],
                })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


# ============================================================
# Label masking — mask all tokens up to (and including) the response template
# ============================================================

def _get_response_template_ids(tokenizer) -> list[int]:
    """
    Tokenize '<|im_start|>assistant\\n' without special-token wrapping.
    Searching for this sequence in input_ids locates the assistant turn boundary.
    """
    return tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)


def mask_labels(
    input_ids: torch.Tensor,
    response_template_ids: list[int],
    pad_token_id: int,
) -> torch.Tensor:
    """
    Returns labels tensor where:
      - system + user tokens (everything before the assistant turn) → -100
      - <|im_start|>assistant\\n wrapping tokens → kept  (model learns turn boundary)
      - assistant response content tokens → kept
      - <|im_end|> after response → kept  (model learns when to stop)
      - padding tokens → -100

    Masking boundary: the last occurrence of the response_template_ids sequence
    (i.e. <|im_start|>assistant\\n) in input_ids.  Everything strictly before
    that index is masked; the template and everything after it is kept.
    """
    labels = input_ids.clone()
    ids = input_ids.tolist()
    tlen = len(response_template_ids)

    # Scan backwards — last assistant turn in case of multi-turn (not used here,
    # but correct by default).
    prefix_end = -1
    for i in range(len(ids) - tlen, -1, -1):
        if ids[i: i + tlen] == response_template_ids:
            prefix_end = i   # index of <|im_start|>  (first token of assistant turn)
            break

    if prefix_end == -1:
        # Template not found — defensive: mask everything
        labels[:] = -100
        return labels

    labels[:prefix_end] = -100          # mask system + user prefix only
    labels[labels == pad_token_id] = -100   # mask padding wherever it appears
    return labels


# ============================================================
# Collator — decord + cache, process_vision_info, label masking
# ============================================================

class VideoCollator:
    def __init__(
        self,
        processor,
        response_template_ids: list[int],
        cfg: dict,
    ):
        self.processor = processor
        self.response_template_ids = response_template_ids
        self.pad_id = processor.tokenizer.pad_token_id or 0
        self.n_frames = cfg.get("num_video_frames", 90)
        self.frame_size = cfg.get("frame_size", 112)
        self.max_pixels = cfg.get("max_pixels_per_frame", None)
        self._debug_done = False

    def _build_messages(self, rec: dict) -> list[dict]:
        """
        Build Qwen3-VL messages for one example.

        System and assistant content are plain strings taken verbatim from the
        JSONL (no re-rendering).  The user turn uses the content-list format so
        the processor can inject video pad tokens (<|video_pad|>) before the
        user text.  process_vision_info picks up the video from that list and
        returns the pixel tensors + video_grid_thw.
        """
        frames = load_video_cached(rec["video_path"], self.n_frames, self.frame_size)
        # process_vision_info requires list/tuple, not raw ndarray
        pil_frames = [Image.fromarray(frames[i]) for i in range(len(frames))]
        video_content: dict = {"type": "video", "video": pil_frames}
        if self.max_pixels is not None:
            # min_pixels must be < max_pixels; library default is ~100352 which
            # exceeds our 112×112=12544 budget, so set an explicit lower bound.
            video_content["min_pixels"] = 4 * 28 * 28   # 3136 — one patch grid
            video_content["max_pixels"] = self.max_pixels

        return [
            {
                "role": "system",
                "content": rec["system"],          # string — verbatim from JSONL
            },
            {
                "role": "user",
                "content": [                       # list — Qwen3-VL video convention
                    video_content,
                    {"type": "text", "text": rec["user_text"]},
                ],
            },
            {
                "role": "assistant",
                "content": rec["target_text"],     # string — verbatim from JSONL
            },
        ]

    def __call__(self, features: list[dict]) -> dict:
        all_messages = [self._build_messages(f) for f in features]

        # ── Chat template (text side) ─────────────────────────────────────
        texts = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            for msgs in all_messages
        ]

        # ── Vision processing ─────────────────────────────────────────────
        # Handle both 2-return and 3-return versions of process_vision_info,
        # and the tuple (frames, metadata) wrapping introduced in some versions.
        try:
            raw = process_vision_info(
                all_messages,
                image_patch_size=14,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            image_inputs, video_inputs, video_kwargs = raw
        except (TypeError, ValueError):
            raw = process_vision_info(all_messages, return_video_kwargs=True)
            if len(raw) == 3:
                image_inputs, video_inputs, video_kwargs = raw
            else:
                image_inputs, video_inputs = raw
                video_kwargs = {}

        # Unpack (frames, metadata) tuples if present
        video_metadatas = None
        if video_inputs and isinstance(video_inputs[0], tuple):
            frames_list, meta_list = zip(*video_inputs)
            video_inputs = list(frames_list)
            video_metadatas = list(meta_list)

        # ── Processor (tokenize + pixel tensors) ─────────────────────────
        proc_kwargs: dict = dict(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if video_metadatas is not None:
            proc_kwargs["video_metadata"] = video_metadatas
        if video_kwargs:
            proc_kwargs.update(video_kwargs)

        batch = self.processor(**proc_kwargs)

        # ── Label masking ─────────────────────────────────────────────────
        batch["labels"] = torch.stack([
            mask_labels(
                batch["input_ids"][i],
                self.response_template_ids,
                self.pad_id,
            )
            for i in range(len(features))
        ])

        if not self._debug_done:
            n_active = (batch["labels"][0] != -100).sum().item()
            seq_len = batch["input_ids"].shape[1]
            print(
                f"[Collator] seq_len={seq_len}  "
                f"active_label_tokens={n_active}  "
                f"pixel_values_videos shape="
                f"{batch.get('pixel_values_videos', torch.tensor([])).shape}"
            )
            self._debug_done = True

        return batch


# ============================================================
# Per-step CSV loss logger
# ============================================================

class LossCSVCallback(TrainerCallback):
    def __init__(self, csv_path: str):
        import csv
        self.path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "epoch", "loss", "lr", "grad_norm"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        import csv
        if not logs:
            return
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                state.global_step,
                logs.get("epoch"),
                logs.get("loss"),
                logs.get("learning_rate"),
                logs.get("grad_norm"),
            ])


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single-GPU SFT for Cosmos-Reason2-8B on PSI pedestrian intent"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 5 training steps then exit (validates pipeline without full run)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Reproducibility ─────────────────────────────────────────────────────
    seed = cfg.get("seed", 42)
    hf_set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Output dir ───────────────────────────────────────────────────────────
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Config    : {args.config}")
    print(f"Format    : {cfg.get('format', 'toulmin')}")
    print(f"Model     : {cfg['model_id']}")
    print(f"Train     : {cfg['train_path']}")
    print(f"Val       : {cfg.get('val_path', '(none)')}")
    print(f"Output    : {output_dir}")
    print(f"Smoke test: {args.smoke}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_dataset = PSIDataset(cfg["train_path"], cfg)
    val_dataset = PSIDataset(cfg["val_path"], cfg) if cfg.get("val_path") else None
    print(f"\nTrain samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Val samples  : {len(val_dataset)}")

    if args.smoke:
        train_dataset.records = train_dataset.records[:8]
        if val_dataset:
            val_dataset.records = val_dataset.records[:4]
        print("[smoke] sliced to 8 train / 4 val records")

    # ── Processor ────────────────────────────────────────────────────────────
    model_id = cfg["model_id"]
    print(f"\nLoading processor from {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    response_template_ids = _get_response_template_ids(processor.tokenizer)
    print(f"Response template IDs: {response_template_ids}")

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"Loading model {model_id} with 4-bit NF4 ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = _VLModelCls.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False   # incompatible with gradient checkpointing

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Collator ─────────────────────────────────────────────────────────────
    collator = VideoCollator(processor, response_template_ids, cfg)

    # ── Training args ─────────────────────────────────────────────────────────
    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"

    _eval_strat = cfg.get("eval_strategy", "epoch") if val_dataset is not None else "no"
    _save_strat = cfg.get("save_strategy", "epoch")
    _eval_steps = cfg.get("eval_steps", 100)
    _save_total_limit = cfg.get("save_total_limit", 2)
    _load_best = cfg.get("load_best_model_at_end", False) and val_dataset is not None
    _metric_best = cfg.get("metric_for_best_model", "eval_loss")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        optim="paged_adamw_8bit",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.get("logging_steps", 10),
        save_strategy=_save_strat,
        save_total_limit=_save_total_limit,
        eval_strategy=_eval_strat,
        eval_steps=_eval_steps if _eval_strat == "steps" else None,
        save_steps=_eval_steps if _save_strat == "steps" else None,
        load_best_model_at_end=_load_best,
        metric_for_best_model=_metric_best if _load_best else None,
        greater_is_better=False if _metric_best == "eval_loss" else True,
        report_to=report_to,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        seed=seed,
        # smoke: hard-cap at 5 steps; -1 means "run all steps normally"
        max_steps=5 if args.smoke else -1,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    csv_callback = LossCSVCallback(f"{output_dir}/step_losses.csv")
    callbacks = [csv_callback]
    if cfg.get("early_stopping_patience") and _load_best:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=cfg["early_stopping_patience"],
            early_stopping_threshold=cfg.get("early_stopping_threshold", 0.0),
        ))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    # ── GPU stats ─────────────────────────────────────────────────────────────
    gpu = torch.cuda.get_device_properties(0)
    mem_before = round(torch.cuda.max_memory_reserved(0) / 1024 ** 3, 2)
    print(
        f"\nGPU: {gpu.name}  "
        f"({round(gpu.total_memory / 1024 ** 3, 1)} GB total)  "
        f"reserved before train: {mem_before} GB"
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.train()

    peak_mem = round(torch.cuda.max_memory_reserved(0) / 1024 ** 3, 2)
    print(f"\nPeak GPU memory: {peak_mem} GB")

    if args.smoke:
        print("[smoke] 5-step run complete — pipeline OK")
        return

    # ── Save LoRA adapters ─────────────────────────────────────────────────
    trainer.save_model(output_dir)
    with open(f"{output_dir}/log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"LoRA adapters saved → {output_dir}")


if __name__ == "__main__":
    main()
