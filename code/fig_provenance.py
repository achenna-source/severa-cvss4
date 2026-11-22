"""Figure: monthly share of CVSS v4.0 labels by provenance group, with split boundaries."""
import collections, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT, load_rows

rows = load_rows()
months = sorted({r["published"][:7] for r in rows if r["published"][:7] >= "2024-07"})
cnt = {m: collections.Counter() for m in months}
for r in rows:
    m = r["published"][:7]
    if m in cnt:
        cnt[m][r["prov"]] += 1
groups = ["VulDB", "VulnCheck", "GitHub", "Other"]
colors = {"VulDB": "#1f4e79", "VulnCheck": "#c55a11", "GitHub": "#548235", "Other": "#a5a5a5"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig, ax = plt.subplots(figsize=(7.0, 2.8))
bottom = [0.0] * len(months)
for g in groups:
    vals = [cnt[m][g] / max(1, sum(cnt[m].values())) for m in months]
    ax.bar(range(len(months)), vals, bottom=bottom, color=colors[g], label=g, width=0.85, edgecolor="white", lw=0.3)
    bottom = [b + v for b, v in zip(bottom, vals)]
for boundary in ("2026-03", "2026-04", "2026-07"):
    ax.axvline(months.index(boundary) - 0.5, color="black", lw=0.8, ls="--")
for centre, lab in ((months.index("2026-03"), "dev"), (months.index("2026-05"), "calibration"),
                    ((months.index("2026-07") + len(months) - 1) / 2, "test")):
    ax.text(centre, 1.02, lab, fontsize=7, va="bottom", ha="center")
ax.text((months.index("2026-03") - 1) / 2, 1.02, "training", fontsize=7, va="bottom", ha="center")
ax.set_xticks(range(len(months)))
ax.set_xticklabels([m[2:] for m in months], rotation=90, fontsize=6.5)
ax.set_ylabel("Share of v4.0 labels")
ax.set_ylim(0, 1.0)
ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
ax2 = ax.twinx()
ax2.plot(range(len(months)), [sum(cnt[m].values()) for m in months], color="black", lw=1.0, marker="o", ms=2)
ax2.set_ylabel("v4.0 labels per month")
for s in ("top",):
    ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
fig.tight_layout()
out = os.path.join(ROOT, "figures", "out")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"fig2_provenance.{ext}"), dpi=300, bbox_inches="tight")
print("saved", len(months), "months")
