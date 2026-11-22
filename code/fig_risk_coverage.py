"""Figure 3: selective band error against coverage on the test split, per provenance group, with the operating
points of (a) one global and (b) per-group (Mondrian) Learn-then-Test thresholds at alpha = 0.10."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from common import ROOT, ScoreTable
import abstention as AB
import figstyle as fs

fs.apply()
name = sys.argv[1] if len(sys.argv) > 1 else "deberta_s1"
t = ScoreTable()
d = np.load(os.path.join(ROOT, "results", f"{name}_FINAL_arrays.npz"), allow_pickle=True)
R = json.load(open(os.path.join(ROOT, "results", f"{name}_FINAL.json")))
Y, P, bp, grp = d["test_y"], d["test_ours_band"], d["test_bandp"], d["test_group"]
err = (t.band[P.astype(np.int64) @ t.radix] != t.band[Y.astype(np.int64) @ t.radix]).astype(int)
conf = bp.max(1)
ab = R["abstention"]["0.1"]
ALPHA, YMAX = 0.10, 0.55

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, gridspec_kw=dict(wspace=0.16))
points = {}
for ax, tag, ttl, letter in ((axes[0], "global_bonf", "One global threshold", "a"),
                             (axes[1], "mondrian_bonf", "One threshold per group of CNAs", "b")):
    ax.axhspan(ALPHA, YMAX, color=fs.ACCENT_TINT, lw=0, zorder=0)
    ax.axhline(ALPHA, color=fs.ACCENT, lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.text(0.985, ALPHA + 0.008, "target  α = 10%", ha="right", va="bottom", fontsize=7.4, color=fs.ACCENT)
    ax.text(0.985, YMAX - 0.012, "error above target", ha="right", va="top", fontsize=7.2, color=fs.ACCENT,
            style="italic")
    for g in ("Other", "GitHub", "VulnCheck", "VulDB"):
        m = grp == g
        cov, risk, _ = AB.risk_coverage(conf[m], err[m])
        keep = cov >= 0.01  # below 1% coverage the error rests on a handful of CVEs
        ax.plot(cov[keep], risk[keep], color=fs.GROUP_COLOR[g], lw=1.6, zorder=3, solid_capstyle="round")
        thr = ab[f"thr_{tag}"]
        th = thr[g] if isinstance(thr, dict) else thr
        acc = conf[m] >= th
        if acc.any():
            points[(letter, g)] = (acc.mean(), err[m][acc].mean())
            ax.plot(*points[(letter, g)], "o", ms=7.5, mfc=fs.GROUP_COLOR[g], mec="white", mew=1.4, zorder=5,
                    clip_on=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, YMAX)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Coverage (share of CVEs answered)")
    fs.title(ax, ttl)
    fs.panel_label(ax, letter, x=-0.075 if letter == "b" else -0.15, y=1.025)
axes[0].set_ylabel("Band error among answered CVEs")

gx, gy = points[("a", "GitHub")]
axes[0].annotate("GitHub advisories answered\nwith %.1f%% band error" % (100 * gy), xy=(gx, gy), xytext=(0.36, 0.475),
                 fontsize=7.3, color=fs.INK, va="center",
                 arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.7, shrinkA=2, shrinkB=5))
axes[1].text(0.05, 0.475, "GitHub advisories: none answered,\nall referred to analysts", fontsize=7.3, color=fs.INK,
             va="center")
handles = [Line2D([], [], color=fs.GROUP_COLOR[g], lw=1.8, label=g) for g in ("VulDB", "VulnCheck", "Other", "GitHub")]
handles.append(Line2D([], [], ls="", marker="o", ms=7, mfc="#8C8C8C", mec="white", mew=1.2,
                      label="operating point at α = 10%"))
fig.legend(handles=handles, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.1), handlelength=1.6,
           columnspacing=1.5)
out = os.path.join(ROOT, "figures", "out")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"fig3_risk_coverage.{ext}"), dpi=300, bbox_inches="tight")
print("saved for", name, {k: tuple(round(x, 3) for x in v) for k, v in points.items()})
