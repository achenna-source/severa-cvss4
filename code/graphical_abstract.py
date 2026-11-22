"""Graphical abstract (JISA): the gap, the method, the result, in the figure identity of the manuscript.
Reads every number from manuscript/fills.json."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import figstyle as fs

fs.apply()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "out")
F = json.load(open(os.path.join(ROOT, "manuscript", "fills.json"), encoding="utf-8"))

W, H = 100, 47
fig = plt.figure(figsize=(7.0, 3.3))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
STRIP = 3.6


def card(x, w, title, color):
    ax.add_patch(Rectangle((x, 1.0), w, H - 2.0, fc="white", ec="#C8C8C8", lw=0.6))
    ax.add_patch(Rectangle((x, H - 1.0 - STRIP), w, STRIP, fc=color, ec="none"))
    ax.text(x + 1.4, H - 1.0 - STRIP / 2, title, fontsize=8.6, fontfamily=fs.TEXT_M, color="white", va="center")


def txt(x, y, s, size=7.6, color=fs.INK, font=fs.TEXT, ha="left", va="top", ls=1.3):
    ax.text(x, y, s, fontsize=size, color=color, fontfamily=font, ha=ha, va=va, linespacing=ls)


def rule(x0, x1, y):
    ax.plot([x0, x1], [y, y], color="#D9D9D9", lw=0.7)


# ---- the gap
card(1.0, 31.0, "THE GAP", fs.INK)
txt(2.8, 40.2, F["SHARE_V4_2026"], size=24, color=fs.ACCENT, font=fs.TEXT_M)
txt(2.8, 33.4, "of the CVEs published in 2026\ncarry a CVSS v4.0 vector")
rule(2.8, 30.2, 24.8)
txt(2.8, 22.8, "Since April 2026 the NVD no longer\nscores CVEs that their CNA has\nalready scored.")
rule(2.8, 30.2, 12.9)
txt(2.8, 10.9, "Who scores keeps changing:\nVulnCheck went from %s of training\nlabels to %s of test labels." % (F["PROV_VULNCHECK_TRAIN"], F["PROV_VULNCHECK_TEST"]))

# ---- the method
card(34.5, 31.0, "SEVERA", fs.ACCENT)
steps = [(fs.OI["blue"], "Any per-metric classifier", "TF-IDF, DeBERTa-v3 or Qwen2.5-7B"),
         (fs.OI["green"], "Knowledge base", "CWE priors, v3.1 to v4.0 alignment"),
         (fs.ACCENT, "Exact Bayes decision", "all 104,976 vectors, about %s ms" % F["DECODE_MS"]),
         (fs.OI["orange"], "Certified abstention", "Learn-then-Test, per group of CNAs")]
y = 39.6
for k, (col, head, sub) in enumerate(steps, 1):
    ax.add_patch(Rectangle((36.2, y - 7.0), 27.6, 7.0, fc="#FAFAFA", ec="none"))
    ax.add_patch(Rectangle((36.2, y - 7.0), 0.9, 7.0, fc=col, ec="none"))
    ax.add_patch(Circle((39.6, y - 3.5), 1.6, fc=col, ec="none"))
    txt(39.6, y - 3.55, str(k), size=7.6, color="white", font=fs.TEXT_M, ha="center", va="center")
    txt(42.2, y - 1.1, head, size=7.9, font=fs.TEXT_M)
    txt(42.2, y - 4.0, sub, size=6.9, color=fs.MUTED)
    if k < 4:
        ax.add_patch(FancyArrowPatch((50.0, y - 7.0), (50.0, y - 9.0), arrowstyle="-|>", mutation_scale=6, lw=0.7,
                                     color=fs.AXIS))
    y -= 9.0

# ---- the result
card(68.0, 31.0, "RESULTS  ·  %s new CVEs" % F["N_TEST"], fs.OI["blue"])
txt(69.8, 40.2, "−%s to −%s" % (F["QWEN_MAE_REL"], F["TFIDF_MAE_REL"]), size=16, color=fs.ACCENT, font=fs.TEXT_M)
txt(69.8, 34.6, "score error, on three classifiers")
bx = fig.add_axes([0.80, 0.40, 0.175, 0.25])
mods = ["TF-IDF", "DeBERTa-v3", "Qwen 7B"]
base = [float(F["TFIDF_BASE_MAE"]), float(F["DEB_BASE_MAE"]), float(F["QWEN_BASE_MAE"])]
sev = [float(F["TFIDF_SEV_MAE"]), float(F["DEB_SEV_MAE"]), float(F["QWEN_SEV_MAE"])]
for i, (b0, s0) in enumerate(zip(base, sev)):
    bx.plot([s0, b0], [i, i], color="#CFCFCF", lw=2.2, solid_capstyle="round", zorder=1)
    bx.plot(b0, i, "o", ms=5.5, mfc="#9E9E9E", mec="white", mew=0.8, zorder=2)
    bx.plot(s0, i, "o", ms=6.5, mfc=fs.ACCENT, mec="white", mew=0.8, zorder=3)
    bx.text(s0 - 0.02, i, "%.2f" % s0, ha="right", va="center", fontsize=6.6, color=fs.ACCENT)
    bx.text(b0 + 0.02, i, "%.2f" % b0, ha="left", va="center", fontsize=6.6, color=fs.MUTED)
bx.set_yticks(range(3)); bx.set_yticklabels(mods, fontsize=6.9)
bx.set_ylim(2.5, -0.5); bx.set_xlim(0.36, 1.0)
bx.set_xticks([]); bx.spines["bottom"].set_visible(False); bx.spines["left"].set_visible(False)
bx.tick_params(axis="y", length=0)
for xx, label, col in ((-0.55, "score MAE:", fs.MUTED), (-0.08, "● per-metric argmax", "#9E9E9E"),
                       (0.62, "● SEVERA", fs.ACCENT)):
    bx.text(xx, -0.12, label, transform=bx.transAxes, fontsize=6.3, color=col, ha="left", va="top")
rule(69.8, 97.2, 15.4)
txt(69.8, 13.6, "%s%%" % F["ABST_COV_010"], size=16, color=fs.ACCENT, font=fs.TEXT_M)
txt(81.8, 13.2, "answered at %s%% band error\n(target 10%%); GitHub\nadvisories go to analysts" % F["ABST_RISK_010"],
    size=6.9, ls=1.25)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "graphical_abstract." + ext), dpi=400)
print("written", os.path.join(OUT, "graphical_abstract.png"))
