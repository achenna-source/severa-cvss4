"""Graphical abstract (JISA): the gap, the method, the result. Reads the numbers from manuscript/fills.json."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "out")
F = json.load(open(os.path.join(ROOT, "manuscript", "fills.json"), encoding="utf-8"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
NAVY, BLUE, GREY, GREEN, ORANGE, INK = "#17375E", "#DCE6F2", "#F2F2F2", "#E2EFDA", "#FCE4D6", "#222222"

fig = plt.figure(figsize=(7.0, 3.3))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 47); ax.axis("off")


def panel(x, w, title):
    ax.add_patch(FancyBboxPatch((x, 1.2), w, 44, boxstyle="round,pad=0.2,rounding_size=1.0", fc="white", ec=NAVY, lw=0.9))
    ax.text(x + w / 2, 43.2, title, ha="center", va="top", fontsize=9.2, fontweight="bold", color=NAVY)


def box(x, y, w, h, text, fc, bold=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec=NAVY, lw=0.7))
    if bold:
        ax.text(x + w / 2, y + h - 1.0, bold, ha="center", va="top", fontsize=7.4, fontweight="bold", color=NAVY)
        ax.text(x + w / 2, y + h - 3.9, text, ha="center", va="top", fontsize=6.8, color=INK, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.8, color=INK, linespacing=1.25)


def down(x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=8, lw=0.8, color=NAVY))


# ---- panel 1: the gap
panel(1, 29, "The gap")
box(2.5, 30.0, 26, 10.0, "Since April 2026 the NVD no\nlonger scores CVEs that their\nCNA has already scored", GREY, "Scoring moved to CNAs")
box(2.5, 17.0, 26, 10.6, "only %s of the CVEs published\nin 2026 carry a v4.0 vector" % F["SHARE_V4_2026"], ORANGE, "CVSS v4.0 is mostly missing")
box(2.5, 3.0, 26, 11.6, "VulnCheck: %s of training labels,\n%s of test labels; the error of\na model differs by CNA" % (F["PROV_VULNCHECK_TRAIN"], F["PROV_VULNCHECK_TEST"]),
    GREY, "Who scores keeps changing")

# ---- panel 2: SEVERA
panel(33, 32, "SEVERA")
box(35, 33.2, 28, 7.0, "Any per-metric classifier\n(TF-IDF, DeBERTa, Qwen2.5-7B)", BLUE)
down(49, 33.0, 30.4)
box(35, 23.0, 28, 7.0, "Knowledge base: CWE priors,\nv3.1 to v4.0 alignment", GREEN)
down(49, 22.8, 20.2)
box(35, 12.6, 28, 7.4, "Exact Bayes decision over all\n104,976 vectors (about %s ms)" % F["DECODE_MS"], BLUE)
down(49, 12.4, 9.8)
box(35, 2.6, 28, 7.0, "Certified abstention per group\nof CNAs (Learn-then-Test)", ORANGE)

# ---- panel 3: results
panel(68, 31, "Results")
ax.text(83.5, 39.6, "%s CVEs published Jul to Nov 2026" % F["N_TEST"], ha="center", va="top", fontsize=6.8, color=INK)
bx = fig.add_axes([0.735, 0.43, 0.235, 0.36])
mods = ["TF-IDF", "DeBERTa", "Qwen 7B"]
base = [float(F["TFIDF_BASE_MAE"]), float(F["DEB_BASE_MAE"]), float(F["QWEN_BASE_MAE"])]
sev = [float(F["TFIDF_SEV_MAE"]), float(F["DEB_SEV_MAE"]), float(F["QWEN_SEV_MAE"])]
xs = range(len(mods)); w = 0.38
bx.bar([i - w / 2 for i in xs], base, w, color="#B7C3D0", label="per-metric argmax")
bx.bar([i + w / 2 for i in xs], sev, w, color=NAVY, label="SEVERA")
for i in xs:
    bx.text(i + w / 2, sev[i] + 0.02, "%.2f" % sev[i], ha="center", va="bottom", fontsize=6.0, color=NAVY)
bx.set_xticks(list(xs)); bx.set_xticklabels(mods, fontsize=6.6)
bx.set_ylabel("score MAE", fontsize=6.8); bx.tick_params(axis="y", labelsize=6.2)
bx.set_ylim(0, max(base) * 1.25)
for s in ("top", "right"):
    bx.spines[s].set_visible(False)
bx.legend(fontsize=5.8, frameon=False, loc="upper right", handlelength=1.0)
box(69.5, 2.6, 28, 9.6, "%s%% answered at %s%% band error;\nGitHub advisories go to analysts" % (F["ABST_COV_010"], F["ABST_RISK_010"]),
    ORANGE, "Certified answers (target 10%)")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "graphical_abstract." + ext), dpi=400)
print("written", os.path.join(OUT, "graphical_abstract.png"))
