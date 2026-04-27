#!/usr/bin/env python3
"""
train_kto.py — KTO (Kahneman-Tversky Optimization) fine-tuning on PSI.

KTO (Ethayarajh et al. 2024) aligns models without requiring paired
(chosen, rejected) comparisons.  Each example is labeled individually as
"desirable" or "undesirable" and the asymmetric prospect-theoretic value
function pushes the policy UP for desirable and DOWN for undesirable.

DPO pairs → 2x KTO records:
    chosen   → label="desirable"
    rejected → label="undesirable"

ref_kl_estimate: beta-scaled EMA of KL[π||π_ref] tracked per class;
the OTHER class's EMA is used as the KL anchor in the loss, matching the
KTO paper's cross-KL formulation.

Architecture identical to train_dpo.py except:
  - KTODataset / KTOCollator (single response, not chosen+rejected)
  - kto_loss replaces dpo_loss
  - per-class KL EMA state maintained in the training loop

Usage:
    python train_kto.py --config configs/toulmin_kto.yaml
    python train_kto.py --config configs/toulmin_kto.yaml --smoke
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
    set_seed as hf_set_seed,
)
from peft import PeftModel, prepare_model_for_kbit_training

try:
    from transformers import Qwen3VLForConditionalGeneration as _VLModelCls
except ImportError:
    from transformers import AutoModelForVision2Seq as _VLModelCls

from qwen_vl_utils import process_vision_info
import bitsandbytes as bnb


# ============================================================
# Config
# ============================================================

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# Video loading — identical to train_dpo.py
# ============================================================

_VIDEO_CACHE_DIR = Path("/tmp/video_cache")
_VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _video_cache_path(video_path: str) -> Path:
    md5 = hashlib.md5(video_path.encode()).hexdigest()
    return _VIDEO_CACHE_DIR / f"{md5}.npy"


def _decode_video_decord(path: str, n_frames: int, frame_size: int) -> np.ndarray:
    import decord
    vr = decord.VideoReader(path, num_threads=2)
    indices = np.linspace(0, len(vr) - 1, n_frames, dtype=int)
    frames = vr.get_batch(indices.tolist()).asnumpy()
    if frame_size != frames.shape[1] or frame_size != frames.shape[2]:
        frames = np.stack([
            np.array(Image.fromarray(frames[i]).resize(
                (frame_size, frame_size), Image.BILINEAR
            ))
            for i in range(len(frames))
        ])
    return frames


def load_video_cached(path: str, n_frames: int = 90, frame_size: int = 112) -> np.ndarray:
    cache = _video_cache_path(path)
    if cache.exists():
        return np.load(str(cache))
    frames = _decode_video_decord(path, n_frames, frame_size)
    np.save(str(cache), frames)
    return frames


# ============================================================
# Dataset
# ============================================================

class KTODataset(TorchDataset):
    """
    Reads DPO-format JSONL and expands each pair into two independent records:
      chosen   -> label="desirable"
      rejected -> label="undesirable"
    The 2x expansion is performed at load time; DataLoader shuffle interleaves them.
    """

    def __init__(self, jsonl_path: str):
        self.records: list[dict] = []
        n_d = n_u = 0
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                base = {
                    "video_path": r["video"],
                    "system":     r["system"],
                    "user_text":  r["prompt"][0]["content"],
                }
                self.records.append({
                    **base,
                    "response_text": r["chosen"][0]["content"],
                    "label":         "desirable",
                })
                self.records.append({
                    **base,
                    "response_text": r["rejected"][0]["content"],
                    "label":         "undesirable",
                })
                n_d += 1
                n_u += 1
        self.n_desirable   = n_d
        self.n_undesirable = n_u

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


# ============================================================
# Label masking — identical to train_dpo.py
# ============================================================

def _get_response_template_ids(tokenizer) -> list[int]:
    return tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)


def mask_labels(
    input_ids: torch.Tensor,
    response_template_ids: list[int],
    pad_token_id: int,
) -> torch.Tensor:
    labels = input_ids.clone()
    ids    = input_ids.tolist()
    tlen   = len(response_template_ids)
    prefix_end = -1
    for i in range(len(ids) - tlen, -1, -1):
        if ids[i: i + tlen] == response_template_ids:
            prefix_end = i
            break
    if prefix_end == -1:
        labels[:] = -100
        return labels
    labels[:prefix_end] = -100
    labels[labels == pad_token_id] = -100
    return labels


# ============================================================
# Vision-info helper — identical to train_dpo.py
# ============================================================

def _call_process_vision_info(messages):
    try:
        raw = process_vision_info(
            messages,
            image_patch_size=14,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        image_inputs, video_inputs, video_kwargs = raw
    except (TypeError, ValueError):
        raw = process_vision_info(messages, return_video_kwargs=True)
        if len(raw) == 3:
            image_inputs, video_inputs, video_kwargs = raw
        else:
            image_inputs, video_inputs = raw
            video_kwargs = {}

    video_metadatas = None
    if video_inputs and isinstance(video_inputs[0], tuple):
        frames_list, meta_list = zip(*video_inputs)
        video_inputs    = list(frames_list)
        video_metadatas = list(meta_list)

    return image_inputs, video_inputs, video_metadatas, video_kwargs


# ============================================================
# KTO Collator — single response per example (not paired)
# ============================================================

class KTOCollator:
    """
    Encodes one response per example.  Batch contains a mix of desirable
    and undesirable records drawn from the shuffled 2x dataset.
    """

    def __init__(self, processor, response_template_ids: list[int], cfg: dict):
        self.processor             = processor
        self.response_template_ids = response_template_ids
        self.pad_id    = processor.tokenizer.pad_token_id or 0
        self.n_frames  = cfg.get("num_video_frames", 90)
        self.frame_size = cfg.get("frame_size", 112)
        self.max_pixels = cfg.get("max_pixels_per_frame", None)
        self._debug_done = False

    def _video_content(self, video_path: str) -> dict:
        frames = load_video_cached(video_path, self.n_frames, self.frame_size)
        pil_frames = [Image.fromarray(frames[i]) for i in range(len(frames))]
        vc: dict = {"type": "video", "video": pil_frames}
        if self.max_pixels is not None:
            vc["min_pixels"] = 4 * 28 * 28
            vc["max_pixels"] = self.max_pixels
        return vc

    def _build_messages(self, rec: dict, response_text: str, vc: dict) -> list[dict]:
        return [
            {"role": "system",    "content": rec["system"]},
            {"role": "user",      "content": [vc, {"type": "text", "text": rec["user_text"]}]},
            {"role": "assistant", "content": response_text},
        ]

    def _encode_sequence(self, messages: list[dict]) -> dict:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs, video_metadatas, video_kwargs = \
            _call_process_vision_info(messages)
        proc_kwargs = dict(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if video_metadatas is not None:
            proc_kwargs["video_metadata"] = video_metadatas
        if video_kwargs:
            proc_kwargs.update(video_kwargs)
        return self.processor(**proc_kwargs)

    def __call__(self, features: list[dict]) -> dict:
        ids_list   = []
        mask_list  = []
        lbls_list  = []
        pv_list    = []
        thw_list   = []
        is_d_list  = []

        for rec in features:
            vc  = self._video_content(rec["video_path"])
            enc = self._encode_sequence(self._build_messages(rec, rec["response_text"], vc))
            lbl = mask_labels(enc["input_ids"][0], self.response_template_ids, self.pad_id)

            ids_list.append(enc["input_ids"][0])
            mask_list.append(enc["attention_mask"][0])
            lbls_list.append(lbl)
            is_d_list.append(rec["label"] == "desirable")

            if "pixel_values_videos" in enc:
                pv_list.append(enc["pixel_values_videos"])
            if "video_grid_thw" in enc:
                thw_list.append(enc["video_grid_thw"])

        def pad_stack(tensors: list[torch.Tensor], pad_val: int) -> torch.Tensor:
            max_len = max(t.shape[0] for t in tensors)
            padded  = []
            for t in tensors:
                if t.shape[0] < max_len:
                    p = torch.full((max_len - t.shape[0],), pad_val, dtype=t.dtype)
                    t = torch.cat([t, p])
                padded.append(t)
            return torch.stack(padded)

        batch: dict = {
            "input_ids":      pad_stack(ids_list,  self.pad_id),
            "attention_mask": pad_stack(mask_list, 0),
            "labels":         pad_stack(lbls_list, -100),
            "is_desirable":   torch.tensor(is_d_list, dtype=torch.bool),
        }
        if pv_list:
            batch["pixel_values_videos"] = torch.cat(pv_list, dim=0)
        if thw_list:
            batch["video_grid_thw"] = torch.cat(thw_list, dim=0)

        if not self._debug_done:
            active   = (batch["labels"][0] != -100).sum().item()
            pv_shape = batch.get("pixel_values_videos", torch.tensor([])).shape
            lbl_str  = "desirable" if is_d_list[0] else "undesirable"
            print(
                f"[KTOCollator] seq_len={batch['input_ids'].shape[1]}"
                f" active={active}"
                f" label={lbl_str}"
                f" pv={pv_shape}"
            )
            self._debug_done = True

        return batch


# ============================================================
# Shared log-prob helper — identical to train_dpo.py
# ============================================================

def get_response_logps(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    pixel_values_videos=None,
    video_grid_thw=None,
) -> torch.Tensor:
    """Sum of log-probs over assistant response tokens. Shape: (batch,)."""
    kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
    if pixel_values_videos is not None:
        kwargs["pixel_values_videos"] = pixel_values_videos
    if video_grid_thw is not None:
        kwargs["video_grid_thw"] = video_grid_thw

    outputs  = model(**kwargs)
    logits   = outputs.logits[:, :-1, :].float()
    targets  = labels[:, 1:]
    mask     = (targets != -100)

    log_probs    = F.log_softmax(logits, dim=-1)
    per_tok_logp = torch.gather(
        log_probs, 2, targets.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)

    return (per_tok_logp * mask).sum(dim=-1)


# ============================================================
# KTO loss
# ============================================================

def kto_loss(
    policy_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    is_desirable: bool,
    ref_kl_estimate: float,     # beta-scaled EMA from the OTHER class
    beta: float,
    lambda_d: float = 1.0,
    lambda_u: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    KTO value function (Ethayarajh et al. 2024).

    logratio = β * (log π(y|x) - log π_ref(y|x))
    ref_kl   = β * EMA[log π/π_ref] from the other class

    Desirable  loss = λ_d * (1 - σ(logratio - ref_kl))
    Undesirable loss = λ_u * (1 - σ(ref_kl - logratio))

    Gradient pulls policy UP for desirable (logratio → large positive)
    and DOWN for undesirable (logratio → large negative).

    Returns (loss, reward) where reward = β*(log π - log π_ref).detach()
    """
    logratio = beta * (policy_logp - ref_logp)          # (batch,)
    kl_t     = torch.tensor(ref_kl_estimate,
                            dtype=logratio.dtype,
                            device=logratio.device)

    if is_desirable:
        v = lambda_d * (1.0 - torch.sigmoid(logratio - kl_t))
    else:
        v = lambda_u * (1.0 - torch.sigmoid(kl_t - logratio))

    loss   = v.mean()
    reward = logratio.detach().mean()
    return loss, reward


# ============================================================
# Eval loop
# ============================================================

def run_eval(
    model,
    val_loader,
    beta: float,
    lambda_d: float,
    lambda_u: float,
    device,
    global_step: int,
    ema_kl_d: float,
    ema_kl_u: float,
):
    model.eval()
    total_loss = 0.0
    sum_d = sum_u = n_d = n_u = 0.0

    with torch.no_grad():
        for batch in val_loader:
            pv  = batch.get("pixel_values_videos")
            thw = batch.get("video_grid_thw")
            if pv  is not None: pv  = pv.to(device)
            if thw is not None: thw = thw.to(device)

            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)
            is_d = batch["is_desirable"][0].item()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pi_logp = get_response_logps(model, ids, mask, lbls, pv, thw)
                with model.disable_adapter():
                    ref_logp = get_response_logps(model, ids, mask, lbls, pv, thw)
                ref_kl = ema_kl_u if is_d else ema_kl_d
                loss, reward = kto_loss(pi_logp, ref_logp, is_d, ref_kl, beta, lambda_d, lambda_u)

            total_loss += loss.item()
            if is_d:
                sum_d += reward.item(); n_d += 1
            else:
                sum_u += reward.item(); n_u += 1

    n_total  = n_d + n_u
    mean_d   = sum_d / max(n_d, 1)
    mean_u   = sum_u / max(n_u, 1)
    margin   = mean_d - mean_u
    print(
        f"  [eval step={global_step}]"
        f"  loss={total_loss / max(n_total, 1):.4f}"
        f"  reward_d={mean_d:.4f}"
        f"  reward_u={mean_u:.4f}"
        f"  margin={margin:.4f}"
        f"  kl_d={ema_kl_d:.4f}  kl_u={ema_kl_u:.4f}"
    )
    model.train()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single-GPU KTO fine-tuning of Toulmin SFT model on PSI"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="16 examples (~5 optimizer steps), then exit")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    seed = cfg.get("seed", 42)
    hf_set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    beta     = float(cfg.get("beta", 0.1))
    lambda_u = float(cfg.get("lambda_undesirable", 1.0))
    ema_alpha = float(cfg.get("kl_ema_alpha", 0.1))

    print(f"Config          : {args.config}")
    print(f"Model           : {cfg['model_id']}")
    print(f"SFT adapter     : {cfg['sft_adapter_path']}")
    print(f"Train           : {cfg['train_path']}")
    print(f"Val             : {cfg.get('val_path', '(none)')}")
    print(f"Output          : {output_dir}")
    print(f"Beta            : {beta}")
    print(f"KL EMA alpha    : {ema_alpha}")
    print(f"Smoke test      : {args.smoke}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = KTODataset(cfg["train_path"])
    val_dataset   = KTODataset(cfg["val_path"]) if cfg.get("val_path") else None

    # lambda_desirable: auto = n_undesirable / n_desirable
    ld_cfg = cfg.get("lambda_desirable", "auto")
    if str(ld_cfg).lower() == "auto":
        lambda_d = train_dataset.n_undesirable / max(train_dataset.n_desirable, 1)
    else:
        lambda_d = float(ld_cfg)

    print(f"lambda_desirable: {lambda_d:.4f}  "
          f"(n_d={train_dataset.n_desirable}, n_u={train_dataset.n_undesirable})")
    print(f"lambda_undesirable: {lambda_u:.4f}")
    print(f"\nTrain examples: {len(train_dataset)} "
          f"({train_dataset.n_desirable} d + {train_dataset.n_undesirable} u)")
    if val_dataset:
        print(f"Val examples  : {len(val_dataset)}")

    # Smoke: use enough records to guarantee max_smoke_steps optimizer updates
    max_smoke_steps = 5
    if args.smoke:
        grad_accum_smoke = int(cfg.get("gradient_accumulation_steps", 8))
        n_smoke = min(16, len(train_dataset.records))
        train_dataset.records = train_dataset.records[:n_smoke]
        # Rebalance: count d/u after slice
        train_dataset.n_desirable   = sum(1 for r in train_dataset.records if r["label"] == "desirable")
        train_dataset.n_undesirable = sum(1 for r in train_dataset.records if r["label"] == "undesirable")
        if val_dataset:
            val_dataset.records = val_dataset.records[:8]
        # To get max_smoke_steps opt-steps, we need n_smoke batches = max_smoke_steps * grad_accum
        # If n_smoke < that, reduce effective grad_accum for smoke
        n_steps_achievable = n_smoke // grad_accum_smoke
        if n_steps_achievable < max_smoke_steps:
            print(f"[smoke] {n_smoke} records / grad_accum={grad_accum_smoke} → "
                  f"only {n_steps_achievable} opt-steps; "
                  f"overriding grad_accum=2 for smoke to get ~{n_smoke//2} steps")
            cfg["gradient_accumulation_steps"] = 2
        else:
            print(f"[smoke] sliced to {n_smoke} train / {len(val_dataset.records) if val_dataset else 0} val")

    # ── Processor ─────────────────────────────────────────────────────────────
    model_id = cfg["model_id"]
    print(f"\nLoading processor from {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    response_template_ids = _get_response_template_ids(processor.tokenizer)
    print(f"Response template IDs: {response_template_ids}")

    # ── Base model (4-bit NF4) ─────────────────────────────────────────────────
    print(f"Loading base model {model_id} with 4-bit NF4 ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = _VLModelCls.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    base = prepare_model_for_kbit_training(base)

    # ── Load SFT LoRA adapter — continue training it ──────────────────────────
    print(f"Loading SFT adapter from {cfg['sft_adapter_path']} ...")
    model = PeftModel.from_pretrained(
        base,
        cfg["sft_adapter_path"],
        is_trainable=True,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {total_params:,} || "
        f"trainable%: {100 * trainable_params / total_params:.4f}"
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    collator = KTOCollator(processor, response_template_ids, cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collator,
    ) if val_dataset else None

    # ── Hyperparameters ────────────────────────────────────────────────────────
    lr               = float(cfg.get("learning_rate", 5e-6))
    num_epochs       = int(cfg.get("num_train_epochs", 1))
    grad_accum       = int(cfg.get("gradient_accumulation_steps", 8))
    warmup_steps_    = int(cfg.get("warmup_steps", 20))
    max_grad_norm    = float(cfg.get("max_grad_norm", 1.0))
    logging_steps    = int(cfg.get("logging_steps", 5)) if not args.smoke else 1
    eval_every_steps = int(cfg.get("eval_every_steps", 50))

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    steps_per_epoch = math.ceil(len(train_dataset) / grad_accum)
    total_opt_steps = (max_smoke_steps if args.smoke
                       else steps_per_epoch * num_epochs)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps_,
        num_training_steps=total_opt_steps,
    )

    # ── CSV log ────────────────────────────────────────────────────────────────
    log_path = Path(output_dir) / "kto_step_losses.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "opt_step",
            "loss_total",          # loss for this step's example(s)
            "loss_d_avg",          # window avg over desirable examples
            "loss_u_avg",          # window avg over undesirable examples
            "kl_estimate_d",       # beta-scaled EMA KL for desirable
            "kl_estimate_u",       # beta-scaled EMA KL for undesirable
            "reward_d",            # window avg reward for desirable
            "reward_u",            # window avg reward for undesirable
            "reward_margin",       # reward_d_avg - reward_u_avg
            "reward_accuracy",     # float(reward_d_avg > reward_u_avg)
            "lr",
        ])

    # ── GPU stats ──────────────────────────────────────────────────────────────
    gpu = torch.cuda.get_device_properties(0)
    mem_before = round(torch.cuda.max_memory_reserved(0) / 1024 ** 3, 2)
    print(
        f"\nGPU: {gpu.name}  "
        f"({round(gpu.total_memory / 1024 ** 3, 1)} GB total)  "
        f"reserved before train: {mem_before} GB"
    )

    # ── KL EMA state ──────────────────────────────────────────────────────────
    # Separate beta-scaled KL estimates per class.
    # Updated from the CURRENT class; used as anchor for the OTHER class.
    ema_kl_d = 0.0   # running KL from desirable examples
    ema_kl_u = 0.0   # running KL from undesirable examples

    # ── Per-window accumulators (reset every logging_steps opt_steps) ─────────
    win_loss_d_sum = win_loss_u_sum = 0.0
    win_loss_d_n   = win_loss_u_n   = 0
    win_reward_d_sum = win_reward_u_sum = 0.0
    win_reward_d_n   = win_reward_u_n   = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0
    opt_step    = 0
    model.train()

    for epoch in range(num_epochs):
        print(f"\n{'='*60}\nEpoch {epoch + 1}/{num_epochs}\n{'='*60}")
        optimizer.zero_grad()

        for batch in train_loader:
            if args.smoke and opt_step >= max_smoke_steps:
                break

            device = next(model.parameters()).device
            pv  = batch.get("pixel_values_videos")
            thw = batch.get("video_grid_thw")
            if pv  is not None: pv  = pv.to(device)
            if thw is not None: thw = thw.to(device)

            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)
            is_d = batch["is_desirable"][0].item()   # batch_size=1

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Policy forward
                pi_logp = get_response_logps(model, ids, mask, lbls, pv, thw)

                # Reference forward (base, adapter disabled)
                with torch.no_grad(), model.disable_adapter():
                    ref_logp = get_response_logps(model, ids, mask, lbls, pv, thw)

                # Beta-scaled KL for this batch's class
                kl_now = (beta * (pi_logp - ref_logp)).detach().mean().item()

                # Update EMA for current class; use OTHER class as anchor
                if is_d:
                    ema_kl_d = ema_alpha * kl_now + (1.0 - ema_alpha) * ema_kl_d
                    ref_kl   = ema_kl_u
                else:
                    ema_kl_u = ema_alpha * kl_now + (1.0 - ema_alpha) * ema_kl_u
                    ref_kl   = ema_kl_d

                loss, reward = kto_loss(
                    pi_logp, ref_logp, is_d, ref_kl, beta, lambda_d, lambda_u
                )

            # First-batch sanity check (smoke only)
            if args.smoke and global_step == 0:
                print(
                    f"\n[smoke sanity step=0]"
                    f"  label={'desirable' if is_d else 'undesirable'}"
                    f"  pi_logp={pi_logp.item():.4f}"
                    f"  ref_logp={ref_logp.item():.4f}"
                    f"  kl_now={kl_now:.4f}"
                    f"  ref_kl={ref_kl:.4f}"
                    f"  reward={reward.item():.4f}"
                    f"  loss={loss.item():.4f}"
                )
                assert not math.isnan(loss.item()), "loss is NaN at step 0"
                assert not math.isinf(loss.item()), "loss is Inf at step 0"
                if abs(pi_logp.item() - ref_logp.item()) < 1e-6:
                    print("  WARNING: policy == ref — SFT adapter may not be loaded")

            # Per-window accumulation
            if is_d:
                win_loss_d_sum += loss.item();   win_loss_d_n   += 1
                win_reward_d_sum += reward.item(); win_reward_d_n += 1
            else:
                win_loss_u_sum += loss.item();   win_loss_u_n   += 1
                win_reward_u_sum += reward.item(); win_reward_u_n += 1

            (loss / grad_accum).backward()
            global_step += 1

            # Optimizer step every grad_accum batches
            if global_step % grad_accum == 0:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                lr_now = scheduler.get_last_lr()[0]

                if opt_step % logging_steps == 0:
                    loss_d_avg = win_loss_d_sum / max(win_loss_d_n, 1)
                    loss_u_avg = win_loss_u_sum / max(win_loss_u_n, 1)
                    rew_d_avg  = win_reward_d_sum / max(win_reward_d_n, 1)
                    rew_u_avg  = win_reward_u_sum / max(win_reward_u_n, 1)
                    margin     = rew_d_avg - rew_u_avg
                    rew_acc    = float(rew_d_avg > rew_u_avg)
                    peak_mem   = round(torch.cuda.max_memory_allocated(0) / 1e9, 2)

                    print(
                        f"  step={opt_step:4d}"
                        f"  loss={loss.item():.4f}"
                        f"  loss_d={loss_d_avg:.4f}"
                        f"  loss_u={loss_u_avg:.4f}"
                        f"  kl_d={ema_kl_d:.4f}"
                        f"  kl_u={ema_kl_u:.4f}"
                        f"  rew_d={rew_d_avg:.4f}"
                        f"  rew_u={rew_u_avg:.4f}"
                        f"  margin={margin:.4f}"
                        f"  acc={rew_acc:.2f}"
                        f"  lr={lr_now:.2e}"
                        f"  peak={peak_mem}GB"
                    )

                    with open(log_path, "a", newline="") as f:
                        csv.writer(f).writerow([
                            opt_step,
                            round(float(loss.item()), 6),
                            round(loss_d_avg, 6),
                            round(loss_u_avg, 6),
                            round(ema_kl_d, 6),
                            round(ema_kl_u, 6),
                            round(rew_d_avg, 6),
                            round(rew_u_avg, 6),
                            round(margin, 6),
                            round(rew_acc, 4),
                            lr_now,
                        ])

                    # Reset window accumulators
                    win_loss_d_sum = win_loss_u_sum = 0.0
                    win_loss_d_n   = win_loss_u_n   = 0
                    win_reward_d_sum = win_reward_u_sum = 0.0
                    win_reward_d_n   = win_reward_u_n   = 0

                if val_loader and opt_step % eval_every_steps == 0:
                    run_eval(model, val_loader, beta, lambda_d, lambda_u,
                             device, opt_step, ema_kl_d, ema_kl_u)

                if args.smoke and opt_step >= max_smoke_steps:
                    break

        if args.smoke:
            break

        if cfg.get("save_at_epoch_end", True):
            print(f"\nSaving adapter -> {output_dir}")
            model.save_pretrained(output_dir)

    # ── Final report ───────────────────────────────────────────────────────────
    peak_mem = round(torch.cuda.max_memory_allocated(0) / 1e9, 2)
    print(f"\nPeak GPU memory: {peak_mem} GB")
    print(f"Final EMA KL estimates — desirable: {ema_kl_d:.4f}  undesirable: {ema_kl_u:.4f}")

    if args.smoke:
        print(f"\n[smoke] KTO run complete — pipeline OK")
        print(f"  Expected behaviour:")
        print(f"    reward_d > 0:   policy favours chosen over ref")
        print(f"    reward_u < 0:   policy disfavours rejected vs ref")
        print(f"    kl_estimate:    should stabilize, not diverge")
        print(f"    reward_acc:     starts ~0.5, grows toward 1.0")
        print(f"\n  Full training launch command:")
        print(f"    CUDA_VISIBLE_DEVICES=1 python train_kto.py --config configs/toulmin_kto.yaml")
        return

    model.save_pretrained(output_dir)
    print(f"LoRA adapter saved -> {output_dir}")


if __name__ == "__main__":
    main()
