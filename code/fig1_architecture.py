"""Figure 1: architecture of SEVERA (vector PDF + PNG)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "out")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
NAVY, BLUE, GREY, GREEN, ORANGE = "#17375E", "#DCE6F2", "#F2F2F2", "#E2EFDA", "#FCE4D6"

fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
boxes = []


def box(x, y, w, h, title, lines, fc):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=NAVY, lw=0.9)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.2, fontweight="bold", color=NAVY)
    ax.text(x + w / 2, y + h - 6.0, "\n".join(lines), ha="center", va="top", fontsize=6.9, linespacing=1.35)
    boxes.append((x, y, w, h))


def arrow(x0, y0, x1, y1, label=None, dy=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=NAVY))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center", va="bottom", fontsize=6.3, color="#444444")


box(1, 27, 16, 20, "New CVE record", ["description", "CNA CWE ids", "CNA CVSS v3.1", "(if any)"], GREY)
box(22, 27, 18, 20, "Perception", ["LLM backbone", "(Qwen2.5-7B, QLoRA)", "+ 11 metric heads", r"$p_m(v_m \mid x)$"], BLUE)
box(45, 27, 25, 20, "Inference engine", ["per-metric combination", "exact joint posterior over", "104,976 base vectors", "Bayes decision under the", "official v4.0 score"], BLUE)
box(75, 27, 23, 20, "Abstention gate", ["band confidence c(x)", "threshold by Learn-then-", "Test, per provenance", "group: P(risk ≤ α) ≥ 1 − δ"], ORANGE)
box(45, 2, 25, 19, "Knowledge base", ["K1 CWE → metric priors", "K2 v3.1 → v4.0 alignment", "K3 metric compatibility", "official score table", "(36 MacroVectors)"], GREEN)
box(75, 2, 23, 19, "Output", ["answer: vector, CVSS-B,", "band, explanation", "abstain: route to analyst", "with the most uncertain", "metric"], GREY)

arrow(17.6, 37, 21.4, 37)
arrow(40.6, 37, 44.4, 37)
arrow(70.6, 37, 74.4, 37)
arrow(57.5, 21.6, 57.5, 26.4)
arrow(86.5, 26.4, 86.5, 21.6)
ax.text(59, 23.6, "rules and\npriors", fontsize=6.3, color="#444444", va="center")
fig.tight_layout(pad=0.2)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig1_architecture.{ext}"), dpi=300, bbox_inches="tight")
print("saved")
