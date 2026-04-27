#!/usr/bin/env python3
"""
train_ipo.py — Single-GPU IPO fine-tuning of the Toulmin SFT model on PSI.

Identity Preference Optimization (Azar et al., 2023) replaces DPO's sigmoid
loss with a squared loss that has an explicit finite preference margin
  target = 1 / (2 * beta)
preventing the unbounded reward growth that caused DPO over-alignment.

With beta=0.1, target margin = 5.0.  The model is penalised for exceeding
this target — so reward_margin converges to ~5 rather than blowing past 8+.

Architecture is 95% identical to train_dpo.py.  Only the loss function changes.
Reference model: same approximation as DPO (base 4-bit weights, adapter disabled).

Usage:
    python train_ipo.py --config configs/toulmin_ipo.yaml
    python train_ipo.py --config configs/toulmin_ipo.yaml --smoke
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
    total = len(vr)
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
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
# Dataset — identical to train_dpo.py
# ============================================================

class DPODataset(TorchDataset):
    """
    Reads DPO JSONL (schema: id, video, system, prompt, chosen, rejected, meta).
    chosen/rejected are each [{role: assistant, content: "..."}].
    """

    def __init__(self, jsonl_path: str):
        self.records: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                self.records.append({
                    "video_path":    r["video"],
                    "system":        r["system"],
                    "user_text":     r["prompt"][0]["content"],
                    "chosen_text":   r["chosen"][0]["content"],
                    "rejected_text": r["rejected"][0]["content"],
                })

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
    ids = input_ids.tolist()
    tlen = len(response_template_ids)
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
        video_inputs = list(frames_list)
        video_metadatas = list(meta_list)

    return image_inputs, video_inputs, video_metadatas, video_kwargs


# ============================================================
# DPO Collator — identical to train_dpo.py
# ============================================================

class DPOCollator:
    def __init__(self, processor, response_template_ids: list[int], cfg: dict):
        self.processor = processor
        self.response_template_ids = response_template_ids
        self.pad_id = processor.tokenizer.pad_token_id or 0
        self.n_frames   = cfg.get("num_video_frames", 90)
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

    def _build_messages(self, rec: dict, response_text: str, video_content: dict) -> list[dict]:
        return [
            {"role": "system",    "content": rec["system"]},
            {"role": "user",      "content": [video_content,
                                               {"type": "text", "text": rec["user_text"]}]},
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
        chosen_ids_list   = []
        chosen_mask_list  = []
        chosen_lbls_list  = []
        rejected_ids_list  = []
        rejected_mask_list = []
        rejected_lbls_list = []
        pv_list  = []
        thw_list = []

        for rec in features:
            vc = self._video_content(rec["video_path"])

            chosen_enc = self._encode_sequence(
                self._build_messages(rec, rec["chosen_text"], vc)
            )
            rejected_enc = self._encode_sequence(
                self._build_messages(rec, rec["rejected_text"], vc)
            )

            chosen_lbls  = mask_labels(chosen_enc["input_ids"][0],
                                       self.response_template_ids, self.pad_id)
            rejected_lbls = mask_labels(rejected_enc["input_ids"][0],
                                        self.response_template_ids, self.pad_id)

            chosen_ids_list.append(chosen_enc["input_ids"][0])
            chosen_mask_list.append(chosen_enc["attention_mask"][0])
            chosen_lbls_list.append(chosen_lbls)
            rejected_ids_list.append(rejected_enc["input_ids"][0])
            rejected_mask_list.append(rejected_enc["attention_mask"][0])
            rejected_lbls_list.append(rejected_lbls)

            if "pixel_values_videos" in chosen_enc:
                pv_list.append(chosen_enc["pixel_values_videos"])
            if "video_grid_thw" in chosen_enc:
                thw_list.append(chosen_enc["video_grid_thw"])

        def pad_stack(tensors: list[torch.Tensor], pad_val: int) -> torch.Tensor:
            max_len = max(t.shape[0] for t in tensors)
            padded = []
            for t in tensors:
                if t.shape[0] < max_len:
                    p = torch.full((max_len - t.shape[0],), pad_val, dtype=t.dtype)
                    t = torch.cat([t, p])
                padded.append(t)
            return torch.stack(padded)

        batch: dict = {
            "chosen_input_ids":        pad_stack(chosen_ids_list,  self.pad_id),
            "chosen_attention_mask":   pad_stack(chosen_mask_list, 0),
            "chosen_labels":           pad_stack(chosen_lbls_list, -100),
            "rejected_input_ids":      pad_stack(rejected_ids_list,  self.pad_id),
            "rejected_attention_mask": pad_stack(rejected_mask_list, 0),
            "rejected_labels":         pad_stack(rejected_lbls_list, -100),
        }
        if pv_list:
            batch["pixel_values_videos"] = torch.cat(pv_list, dim=0)
        if thw_list:
            batch["video_grid_thw"] = torch.cat(thw_list, dim=0)

        if not self._debug_done:
            c_active = (batch["chosen_labels"][0] != -100).sum().item()
            r_active = (batch["rejected_labels"][0] != -100).sum().item()
            pv_shape = batch.get("pixel_values_videos", torch.tensor([])).shape
            print(
                f"[DPOCollator] chosen seq_len={batch['chosen_input_ids'].shape[1]}"
                f" active={c_active} | "
                f"rejected seq_len={batch['rejected_input_ids'].shape[1]}"
                f" active={r_active} | "
                f"pixel_values_videos shape={pv_shape}"
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

    outputs = model(**kwargs)
    logits  = outputs.logits[:, :-1, :].float()
    targets = labels[:, 1:]
    mask    = (targets != -100)

    log_probs = F.log_softmax(logits, dim=-1)
    per_tok_logp = torch.gather(
        log_probs, 2, targets.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)

    return (per_tok_logp * mask).sum(dim=-1)


# ============================================================
# IPO loss (replaces DPO sigmoid loss)
# ============================================================

def ipo_loss(
    pi_chosen_logp: torch.Tensor,
    pi_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """
    IPO loss (Azar et al., 2023).

    Squared loss with explicit finite target margin = 1 / (2 * beta).
    Prevents unbounded reward growth: model is penalised for exceeding target.

    With beta=0.1 → target = 5.0  (vs DPO which reached 8+ unchecked).
    """
    pi_logratios  = pi_chosen_logp  - pi_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    delta  = pi_logratios - ref_logratios
    target = 1.0 / (2.0 * beta)       # explicit finite margin target
    loss   = (delta - target).pow(2).mean()

    chosen_rewards   = beta * (pi_chosen_logp   - ref_chosen_logp).detach()
    rejected_rewards = beta * (pi_rejected_logp - ref_rejected_logp).detach()
    reward_margin = (chosen_rewards - rejected_rewards).mean()
    reward_acc    = (chosen_rewards > rejected_rewards).float().mean()

    return loss, chosen_rewards.mean(), rejected_rewards.mean(), \
           reward_margin, reward_acc


# ============================================================
# Eval loop
# ============================================================

def run_eval(model, val_loader, beta: float, device, global_step: int):
    target_margin = 1.0 / (2.0 * beta)
    model.eval()
    total_loss = total_margin = n = 0
    with torch.no_grad():
        for batch in val_loader:
            pv  = batch.get("pixel_values_videos")
            thw = batch.get("video_grid_thw")
            if pv  is not None: pv  = pv.to(device)
            if thw is not None: thw = thw.to(device)

            c_ids  = batch["chosen_input_ids"].to(device)
            c_mask = batch["chosen_attention_mask"].to(device)
            c_lbls = batch["chosen_labels"].to(device)
            r_ids  = batch["rejected_input_ids"].to(device)
            r_mask = batch["rejected_attention_mask"].to(device)
            r_lbls = batch["rejected_labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pi_c_logp = get_response_logps(model, c_ids, c_mask, c_lbls, pv, thw)
                pi_r_logp = get_response_logps(model, r_ids, r_mask, r_lbls, pv, thw)
                with torch.no_grad(), model.disable_adapter():
                    ref_c_logp = get_response_logps(model, c_ids, c_mask, c_lbls, pv, thw)
                    ref_r_logp = get_response_logps(model, r_ids, r_mask, r_lbls, pv, thw)
                loss, c_rew, r_rew, margin_t, _ = ipo_loss(
                    pi_c_logp, pi_r_logp, ref_c_logp, ref_r_logp, beta
                )

            total_loss   += loss.item()
            total_margin += margin_t.item()
            n += 1

    print(
        f"  [eval step={global_step}]  "
        f"loss={total_loss / max(n, 1):.4f}  "
        f"reward_margin={total_margin / max(n, 1):.4f}  "
        f"(target={target_margin:.2f})"
    )
    model.train()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single-GPU IPO fine-tuning of Toulmin SFT model on PSI"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="8 examples, 5 optimizer steps, then exit")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    seed = cfg.get("seed", 42)
    hf_set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    beta          = float(cfg.get("beta", 0.1))
    target_margin = 1.0 / (2.0 * beta)

    print(f"Config          : {args.config}")
    print(f"Model           : {cfg['model_id']}")
    print(f"SFT adapter     : {cfg['sft_adapter_path']}")
    print(f"Train           : {cfg['train_path']}")
    print(f"Val             : {cfg.get('val_path', '(none)')}")
    print(f"Output          : {output_dir}")
    print(f"Beta            : {beta}")
    print(f"Target margin   : {target_margin:.2f}  (= 1 / 2β)")
    print(f"Smoke test      : {args.smoke}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = DPODataset(cfg["train_path"])
    val_dataset   = DPODataset(cfg["val_path"]) if cfg.get("val_path") else None
    print(f"\nTrain samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Val samples  : {len(val_dataset)}")

    if args.smoke:
        train_dataset.records = train_dataset.records[:8]
        if val_dataset:
            val_dataset.records = val_dataset.records[:4]
        print("[smoke] sliced to 8 train / 4 val records")

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
    collator = DPOCollator(processor, response_template_ids, cfg)
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
    logging_steps    = int(cfg.get("logging_steps", 5))
    eval_every_steps = int(cfg.get("eval_every_steps", 50))
    max_smoke_steps  = 5

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
    log_path = Path(output_dir) / "ipo_step_losses.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "opt_step", "loss",
            "chosen_logp", "rejected_logp",
            "chosen_reward", "rejected_reward",
            "reward_margin", "target_margin", "reward_accuracy", "lr",
        ])

    # ── GPU stats ──────────────────────────────────────────────────────────────
    gpu = torch.cuda.get_device_properties(0)
    mem_before = round(torch.cuda.max_memory_reserved(0) / 1024 ** 3, 2)
    print(
        f"\nGPU: {gpu.name}  "
        f"({round(gpu.total_memory / 1024 ** 3, 1)} GB total)  "
        f"reserved before train: {mem_before} GB"
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0    # raw data batches seen
    opt_step    = 0    # optimizer updates applied
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

            c_ids  = batch["chosen_input_ids"].to(device)
            c_mask = batch["chosen_attention_mask"].to(device)
            c_lbls = batch["chosen_labels"].to(device)
            r_ids  = batch["rejected_input_ids"].to(device)
            r_mask = batch["rejected_attention_mask"].to(device)
            r_lbls = batch["rejected_labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Policy forward (SFT adapter active)
                pi_c_logp = get_response_logps(model, c_ids, c_mask, c_lbls, pv, thw)
                pi_r_logp = get_response_logps(model, r_ids, r_mask, r_lbls, pv, thw)

                # Reference forward (SFT adapter disabled → 4-bit base)
                with torch.no_grad(), model.disable_adapter():
                    ref_c_logp = get_response_logps(model, c_ids, c_mask, c_lbls, pv, thw)
                    ref_r_logp = get_response_logps(model, r_ids, r_mask, r_lbls, pv, thw)

                loss, c_rew, r_rew, r_margin_t, r_acc_t = ipo_loss(
                    pi_c_logp, pi_r_logp,
                    ref_c_logp, ref_r_logp,
                    beta,
                )

            # First-batch sanity check
            if args.smoke and global_step == 0:
                print(
                    f"\n[smoke sanity]"
                    f"  pi_chosen_logp={pi_c_logp.item():.4f}"
                    f"  pi_rejected_logp={pi_r_logp.item():.4f}"
                    f"  ref_chosen_logp={ref_c_logp.item():.4f}"
                    f"  delta={(pi_c_logp - pi_r_logp - ref_c_logp + ref_r_logp).item():.4f}"
                    f"  target={target_margin:.2f}"
                    f"  raw_loss={loss.item():.4f}"
                )
                assert not math.isnan(loss.item()), "loss is NaN at step 0"
                assert not math.isinf(loss.item()), "loss is Inf at step 0"
                if abs(pi_c_logp.item() - pi_r_logp.item()) < 1e-6:
                    print("  WARNING: pi_chosen == pi_rejected — degenerate pair?")
                if abs(pi_c_logp.item() - ref_c_logp.item()) < 1e-6:
                    print("  WARNING: policy == ref — SFT adapter may not be loaded")
                r_margin_0 = r_margin_t.item()
                if r_margin_0 > target_margin * 1.5:
                    print(f"  WARNING: initial reward_margin={r_margin_0:.3f} already above "
                          f"1.5x target={target_margin:.2f} — check SFT adapter")

            (loss / grad_accum).backward()
            global_step += 1

            # Optimizer step every grad_accum raw batches
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

                r_margin = r_margin_t.item()
                r_acc    = r_acc_t.item()
                lr_now   = scheduler.get_last_lr()[0]

                if opt_step % logging_steps == 0:
                    peak_mem = round(torch.cuda.max_memory_allocated(0) / 1e9, 2)
                    print(
                        f"  step={opt_step:4d}"
                        f"  loss={loss.item():.4f}"
                        f"  c_logp={pi_c_logp.item():.2f}"
                        f"  r_logp={pi_r_logp.item():.2f}"
                        f"  margin={r_margin:.3f}/{target_margin:.1f}"
                        f"  acc={r_acc:.2f}"
                        f"  lr={lr_now:.2e}"
                        f"  peak={peak_mem}GB"
                    )

                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        opt_step,
                        round(float(loss.item()), 6),
                        round(float(pi_c_logp.item()), 4),
                        round(float(pi_r_logp.item()), 4),
                        round(float(c_rew.item()), 4),
                        round(float(r_rew.item()), 4),
                        round(r_margin, 4),
                        round(target_margin, 4),
                        round(r_acc, 4),
                        lr_now,
                    ])

                if val_loader and opt_step % eval_every_steps == 0:
                    run_eval(model, val_loader, beta, device, opt_step)

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

    if args.smoke:
        print(f"\n[smoke] 5-step IPO run complete — pipeline OK")
        print(f"  Expected behaviour:")
        print(f"    reward_acc:    should start ~0.5, grow toward 1.0")
        print(f"    reward_margin: should grow toward {target_margin:.2f} (target=1/2β), NOT blow past 8+")
        print(f"    loss:          should be finite and decreasing")
        print(f"\n  Full training launch command:")
        print(f"    CUDA_VISIBLE_DEVICES=1 python train_ipo.py --config configs/toulmin_ipo.yaml")
        return

    model.save_pretrained(output_dir)
    print(f"LoRA adapter saved -> {output_dir}")


if __name__ == "__main__":
    main()
