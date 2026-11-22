"""Figure 1: architecture of SEVERA (vector PDF + PNG). Left band: any perception module; right band: the decision
layer contributed by the paper; numbered cards with coloured header strips; outcomes on the right."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
import figstyle as fs

fs.apply()
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "out")
os.makedirs(OUT, exist_ok=True)

fig = plt.figure(figsize=(7.2, 3.75))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")
STRIP = 3.4


def band(x, y, w, h, label, fc, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec="none"))
    ax.text(x + 1.2, y + h - 1.3, label, fontsize=7.6, fontfamily=fs.TEXT_M, color=color, va="top", ha="left")


def card(x, y, w, h, num, title, lines, color, size=7.3):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="#BDBDBD", lw=0.6, zorder=2))
    ax.add_patch(Rectangle((x, y + h - STRIP), w, STRIP, fc=color, ec="none", zorder=3))
    tx = x + 1.2
    if num:
        ax.add_patch(Circle((x + 2.2, y + h - STRIP / 2), 1.25, fc="white", ec="none", zorder=4))
        ax.text(x + 2.2, y + h - STRIP / 2 - 0.05, str(num), fontsize=7, fontfamily=fs.TEXT_M, color=color,
                ha="center", va="center", zorder=5)
        tx = x + 4.2
    ax.text(tx, y + h - STRIP / 2 - 0.05, title, fontsize=8.2, fontfamily=fs.TEXT_M, color="white", va="center",
            ha="left", zorder=5)
    ax.text(x + 1.2, y + h - STRIP - 1.4, "\n".join(lines), fontsize=size, color=fs.INK, va="top", ha="left",
            linespacing=1.38, zorder=5)


def arrow(p0, p1, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=fs.AXIS, zorder=6,
                                 connectionstyle="arc3,rad=%s" % rad, shrinkA=0, shrinkB=0))


def note(x, y, text, va="bottom", ha="center", size=7.2, color=None):
    ax.text(x, y, text, fontsize=size, color=color or fs.MUTED, ha=ha, va=va, zorder=7)


BLUE, GREEN, ORANGE = fs.OI["blue"], fs.OI["green"], fs.OI["orange"]
band(0.6, 1.2, 34.6, 49.6, "PERCEPTION  ·  any per-metric classifier", "#F1F3F5", fs.MUTED)
band(36.4, 1.2, 63.0, 49.6, "SEVERA DECISION LAYER  ·  this work", fs.ACCENT_TINT, fs.ACCENT)

card(2.2, 29, 14.6, 14.5, None, "CVE record", ["description", "CNA's CWE ids", "CNA's v3.1 vector", "(when present)"], "#7F7F7F")
card(19.2, 29, 14.2, 14.5, 1, "Perception", ["TF-IDF, DeBERTa-v3", "or Qwen2.5-7B,", "11 metric heads", r"$p_m(v_m \mid x)$"], BLUE)
card(38.2, 29, 21.6, 14.5, 3, "Inference engine", ["exact posterior over all", "104,976 base vectors;", "Bayes decision under", "the official v4.0 score"], fs.ACCENT)
card(38.2, 5.0, 21.6, 13.5, 2, "Knowledge base", ["K1  CWE priors", "K2  v3.1 to v4.0 alignment", "K3  metric compatibility"], GREEN)
card(63.0, 29, 16.8, 14.5, 4, "Abstention gate", ["band confidence c(x),", "Learn-then-Test", "threshold per group", "of CNAs"], ORANGE)
card(63.0, 5.0, 16.8, 13.5, 5, "Explanation", ["the metric whose", "uncertainty moves most", "band probability"], "#7F7F7F", size=7.0)

note(2.4, 24.5, "Trained once, on CVEs published\nbefore March 2026; returns one\nprobability vector per metric.\nSEVERA never retrains it.",
     ha="left", va="top", size=7.0)
# outcomes
ax.add_patch(Rectangle((83.0, 34.0), 15.0, 9.5, fc="white", ec=GREEN, lw=1.0, zorder=2))
ax.text(84.2, 42.2, "Answer", fontsize=8.2, fontfamily=fs.TEXT_M, color=GREEN, va="top")
ax.text(84.2, 39.0, "vector, CVSS-B,\nband, explanation", fontsize=7.0, va="top", linespacing=1.35)
ax.add_patch(Rectangle((83.0, 20.5), 15.0, 9.5, fc="white", ec=fs.ACCENT, lw=1.0, zorder=2))
ax.text(84.2, 28.7, "Refer to analyst", fontsize=8.2, fontfamily=fs.TEXT_M, color=fs.ACCENT, va="top")
ax.text(84.2, 25.5, "with the metric\nto check first", fontsize=7.0, va="top", linespacing=1.35)
note(90.5, 9.0, r"$P(\mathrm{error} > \alpha) \leq \delta$", size=8.6, color=fs.ACCENT)
note(90.5, 5.6, "for each group of CNAs", size=6.9, color=fs.ACCENT)

arrow((16.8, 36.0), (19.2, 36.0))
arrow((33.4, 36.0), (38.2, 36.0))
arrow((49.0, 18.5), (49.0, 29.0))
note(50.0, 23.6, "priors, alignment", ha="left", va="center", size=6.9)
arrow((59.8, 37.5), (63.0, 37.5))
note(61.4, 38.2, r"$\hat{v}$", size=8)
arrow((59.8, 30.5), (63.0, 17.5))
arrow((79.8, 38.75), (83.0, 38.75))
note(81.5, 39.7, r"$c \geq \tau_g$", size=7.4)
arrow((79.8, 31.0), (83.0, 25.25))
note(81.0, 24.0, r"$c < \tau_g$", va="top", size=7.4)
arrow((79.8, 12.0), (88.0, 20.5), rad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig1_architecture.{ext}"), dpi=300)
print("saved")
