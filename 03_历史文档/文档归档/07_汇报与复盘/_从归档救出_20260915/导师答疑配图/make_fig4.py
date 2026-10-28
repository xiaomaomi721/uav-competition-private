#!/usr/bin/env python3
"""Draw the data-flow diagram explaining 6996 images -> 5597 train + 1399 val -> epoch -> 56000 steps."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent

fig, ax = plt.subplots(figsize=(10, 7.2), dpi=150)
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

def box(x, y, w, h, text, fc="#1f77b4", tc="white", fs=12, bold=True):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       linewidth=1.2, edgecolor="#1f3b73", facecolor=fc)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold" if bold else "normal")

def note(x, y, text, fs=10, color="#333333", ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color)

def arrow(x1, y1, x2, y2, color="#1f3b73"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=16, linewidth=1.6, color=color)
    ax.add_patch(a)

# 顶层：6996
box(3.4, 8.6, 3.2, 0.9, "官方训练集\n6996 张图", fc="#1f77b4", fs=13)

# 分叉
arrow(5.0, 8.6, 2.6, 6.6)
arrow(5.0, 8.6, 7.4, 6.6)

box(0.6, 5.8, 4.0, 0.85, "训练集：5597 张\n（喂给模型学习）", fc="#2ca02c", fs=12)
box(5.4, 5.8, 4.0, 0.85, "验证集：1399 张\n（不学习，只打分）", fc="#ff7f0e", fs=12)

note(2.6, 7.5, "拆成两份", fs=10, color="#555555")

# 训练集 -> epoch
arrow(2.6, 5.8, 2.6, 4.4)
box(0.6, 3.55, 4.0, 0.85, "1 个 Epoch\n= 把 5597 张完整看一遍", fc="#1f77b4", fs=12)

note(5.3, 4.0, "怎么算步数？\n5597 ÷ 4张/次 ≈ 1400 步\n(1 Epoch = 1400 步)", fs=10, color="#444444")

# epoch -> 40 遍
arrow(2.6, 3.55, 2.6, 2.15)
box(0.6, 1.3, 4.0, 0.85, "反复看 40 遍\n= 40 个 Epoch", fc="#d62728", fs=12)

note(5.3, 1.7, "总步数：\n1400 步 × 40 = 56000 步\n\n（56000 是『步数』\n不是『图数』！）", fs=10, color="#d62728")

ax.text(5.0, 0.35, "一次参数更新 = 消化 4 张图（Batch Size 2 × 梯度累积 2 = 等效 4）",
        ha="center", fontsize=10, color="#333333")

ax.text(5.0, 10.0, "数据流全貌：图数 → 训练/验证 → Epoch → 步数", ha="center", fontsize=15, fontweight="bold", color="#1f3b73")

fig.tight_layout()
fig.savefig(OUT / "fig4_数据流讲解.png", bbox_inches="tight")
plt.close(fig)
print("saved fig4")
