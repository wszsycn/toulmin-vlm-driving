import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

SFT_LOG  = "/workspace/outputs/Cosmos-Reason2-8B-psi-video-90f-sft/log_history.json"
GRPO_LOG = "/workspace/outputs/Cosmos-Reason2-8B-psi-video-90f-grpo/log_history.json"
OUT_DIR  = "/workspace/eval_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def load_log(path):
    with open(path) as f:
        return json.load(f)

def extract(log, key):
    steps, vals = [], []
    for entry in log:
        if key in entry and entry[key] is not None:
            steps.append(entry.get("step", len(steps)))
            vals.append(entry[key])
    return steps, vals

def moving_average(vals, window=10):
    if len(vals) < window:
        return vals
    result = []
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        result.append(sum(vals[start:i+1]) / (i - start + 1))
    return result

def plot_metric(ax, steps, vals, title, color, window=10):
    """画原始值（淡色）+ 移动平均（深色）"""
    ax.plot(steps, vals, color=color, linewidth=0.8, alpha=0.3)
    smoothed = moving_average(vals, window)
    ax.plot(steps, smoothed, color=color, linewidth=2.0, label=f"MA({window})")
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

# ── SFT ──────────────────────────────────────────────────
sft_log = load_log(SFT_LOG)

sft_loss_steps,  sft_loss  = extract(sft_log, "loss")
sft_lr_steps,    sft_lr    = extract(sft_log, "learning_rate")
sft_grad_steps,  sft_grad  = extract(sft_log, "grad_norm")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("SFT Training — Cosmos-Reason2-8B", fontsize=13, fontweight="bold")

axes[0].plot(sft_loss_steps, sft_loss, color="#534AB7", linewidth=0.8, alpha=0.3)
axes[0].plot(sft_loss_steps, moving_average(sft_loss), color="#534AB7", linewidth=2.0, label="MA(10)")
axes[0].set_title("Loss"); axes[0].set_xlabel("Step"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)

axes[1].plot(sft_lr_steps, sft_lr, color="#0F6E56", linewidth=1.5)
axes[1].set_title("Learning Rate"); axes[1].set_xlabel("Step"); axes[1].grid(alpha=0.3)
axes[1].ticklabel_format(style="sci", axis="y", scilimits=(0,0))

axes[2].plot(sft_grad_steps, sft_grad, color="#BA7517", linewidth=0.8, alpha=0.3)
axes[2].plot(sft_grad_steps, moving_average(sft_grad), color="#BA7517", linewidth=2.0, label="MA(10)")
axes[2].set_title("Grad Norm"); axes[2].set_xlabel("Step"); axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)

plt.tight_layout()
sft_out = f"{OUT_DIR}/sft_training_log.png"
plt.savefig(sft_out, dpi=150, bbox_inches="tight")
plt.close()
print(f"SFT plot saved: {sft_out}")

# ── GRPO ─────────────────────────────────────────────────
grpo_log = load_log(GRPO_LOG)

grpo_loss_steps,   grpo_loss   = extract(grpo_log, "loss")
grpo_lr_steps,     grpo_lr     = extract(grpo_log, "learning_rate")
grpo_reward_steps, grpo_reward = extract(grpo_log, "reward")
grpo_kl_steps,     grpo_kl     = extract(grpo_log, "kl")

# 只取 mean 的 reward 子项
reward_mean_keys = [
    ("answer_correct",  "rewards/answer_correct_reward_func/mean"),
    ("case_reasoning",  "rewards/case_reasoning_reward/mean"),
    ("balanced_acc",    "rewards/balanced_acc_reward/mean"),
    ("direction",       "rewards/direction_reward_func/mean"),
    ("format",          "rewards/format_reward_func/mean"),
    ("non_redundancy",  "rewards/non_redundancy_reward_func/mean"),
    ("probabilistic",   "rewards/probabilistic_reward_func/mean"),
]
reward_colors = ["#534AB7","#0F6E56","#185FA5","#BA7517","#A32D2D","#3B6D11","#D4537E"]

# 第一行：loss / lr / total reward / kl
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
fig.suptitle("GRPO Training — Cosmos-Reason2-8B", fontsize=13, fontweight="bold")

def plot_ax(ax, steps, vals, title, color, window=10):
    ax.plot(steps, vals, color=color, linewidth=0.8, alpha=0.3)
    ax.plot(steps, moving_average(vals, window), color=color, linewidth=2.0, label=f"MA({window})")
    ax.set_title(title); ax.set_xlabel("Step"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

plot_ax(axes[0,0], grpo_loss_steps,   grpo_loss,   "Loss",          "#534AB7")
axes[0,1].plot(grpo_lr_steps, grpo_lr, color="#0F6E56", linewidth=1.5)
axes[0,1].set_title("Learning Rate"); axes[0,1].set_xlabel("Step")
axes[0,1].grid(alpha=0.3); axes[0,1].ticklabel_format(style="sci", axis="y", scilimits=(0,0))
plot_ax(axes[0,2], grpo_reward_steps, grpo_reward, "Total Reward",  "#185FA5")
plot_ax(axes[0,3], grpo_kl_steps,     grpo_kl,     "KL Divergence", "#A32D2D")

# 第二行：7个 reward 子项（放6个，最后一格空）
for i, (label, key) in enumerate(reward_mean_keys[:6]):
    steps, vals = extract(grpo_log, key)
    row, col = 1, i % 4 if i < 4 else i - 4 + 1
    row = 1
    col = i
    if i >= 4:
        break
    plot_ax(axes[1, i], steps, vals, label, reward_colors[i])

# 最后把剩余3个放在一张图
ax_last = axes[1, 3]
for i in range(4, len(reward_mean_keys)):
    label, key = reward_mean_keys[i]
    steps, vals = extract(grpo_log, key)
    ax_last.plot(steps, vals, color=reward_colors[i], linewidth=0.8, alpha=0.3)
    ax_last.plot(steps, moving_average(vals), color=reward_colors[i],
                 linewidth=2.0, label=label)
ax_last.set_title("other rewards"); ax_last.set_xlabel("Step")
ax_last.grid(alpha=0.3); ax_last.legend(fontsize=7)

plt.tight_layout()
grpo_out = f"{OUT_DIR}/grpo_training_log.png"
plt.savefig(grpo_out, dpi=150, bbox_inches="tight")
plt.close()
print(f"GRPO plot saved: {grpo_out}")