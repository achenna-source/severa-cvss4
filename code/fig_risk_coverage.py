"""Figure: selective band error vs coverage on the test split, per provenance group, with the operating
points of the global and the per-group (Mondrian) Learn-then-Test thresholds at alpha = 0.10."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT, ScoreTable
import abstention as AB

name = sys.argv[1] if len(sys.argv) > 1 else "deberta_s1"
t = ScoreTable()
d = np.load(os.path.join(ROOT, "results", f"{name}_FINAL_arrays.npz"), allow_pickle=True)
R = json.load(open(os.path.join(ROOT, "results", f"{name}_FINAL.json")))
Y, P, bp, grp = d["test_y"], d["test_ours_band"], d["test_bandp"], d["test_group"]
err = (t.band[P.astype(np.int64) @ t.radix] != t.band[Y.astype(np.int64) @ t.radix]).astype(int)
conf = bp.max(1)
ab = R["abstention"]["0.1"]
colors = {"VulDB": "#1f4e79", "VulnCheck": "#c55a11", "GitHub": "#548235", "Other": "#7f7f7f"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
for ax, tag, title in ((axes[0], "global_bonf", "Global threshold"), (axes[1], "mondrian_bonf", "Per-group thresholds")):
    for g in ("VulDB", "VulnCheck", "Other", "GitHub"):
        m = grp == g
        cov, risk, _ = AB.risk_coverage(conf[m], err[m])
        ax.plot(cov, risk, color=colors[g], lw=1.1, label=g)
        thr = ab[f"thr_{tag}"]
        th = thr[g] if isinstance(thr, dict) else thr
        acc = conf[m] >= th
        if acc.any():
            ax.plot(acc.mean(), err[m][acc].mean(), "o", color=colors[g], ms=5, mec="black", mew=0.6)
        else:
            ax.plot(0, 0, "x", color=colors[g], ms=6)
    ax.axhline(0.10, color="black", ls="--", lw=0.8)
    ax.text(0.02, 0.105, "α = 0.10", fontsize=7, va="bottom")
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel("Coverage")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.55)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Selective band error")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.06, 1, 1))
out = os.path.join(ROOT, "figures", "out")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"fig3_risk_coverage.{ext}"), dpi=300, bbox_inches="tight")
print("saved for", name)
