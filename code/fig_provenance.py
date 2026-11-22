"""Figure 2: (a) CVSS v4.0 labels per month, (b) share of each provenance group, with the temporal splits as
background bands."""
import collections, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT, load_rows
import figstyle as fs

fs.apply()
rows = load_rows()
months = sorted({r["published"][:7] for r in rows if r["published"][:7] >= "2024-07"})
cnt = {m: collections.Counter() for m in months}
for r in rows:
    m = r["published"][:7]
    if m in cnt:
        cnt[m][r["prov"]] += 1
groups = ["VulDB", "VulnCheck", "GitHub", "Other"]
X = range(len(months))
idx = months.index
spans = [("Training", -0.5, idx("2026-03") - 0.5, None), ("Dev", idx("2026-03") - 0.5, idx("2026-04") - 0.5, "#F1F3F5"),
         ("Calibration", idx("2026-04") - 0.5, idx("2026-07") - 0.5, "#E4E8EC"),
         ("Test", idx("2026-07") - 0.5, len(months) - 0.5, fs.ACCENT_TINT)]

fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.9], hspace=0.12))
for ax in (a, b):
    for name, x0, x1, col in spans:
        if col:
            ax.axvspan(x0, x1, color=col, lw=0, zorder=0)
tot = [sum(cnt[m].values()) for m in months]
a.bar(X, tot, width=0.72, color=["#8C8C8C" if m < "2026-07" else fs.ACCENT for m in months], zorder=2)
a.set_ylabel("v4.0 labels\nper month")
a.yaxis.grid(True, zorder=1); a.set_axisbelow(True)
a.set_ylim(0, max(tot) * 1.32)
for name, x0, x1, col in spans:
    a.text((x0 + x1) / 2, max(tot) * 1.30, name, ha="center", va="top", fontsize=7.6, fontfamily=fs.TEXT_M,
           color=fs.ACCENT if name == "Test" else fs.MUTED)
fs.panel_label(a, "a", x=-0.115, y=0.98)

bottom = [0.0] * len(months)
for g in groups:
    vals = [cnt[m][g] / max(1, sum(cnt[m].values())) for m in months]
    b.bar(X, vals, bottom=bottom, width=0.72, color=fs.GROUP_COLOR[g], label=g, zorder=2, lw=0)
    bottom = [u + v for u, v in zip(bottom, vals)]
b.set_ylim(0, 1)
b.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); b.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
b.set_ylabel("Share of v4.0 labels")
fs.panel_label(b, "b", x=-0.115, y=0.99)
b.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), handlelength=1.1, columnspacing=1.6)
labels = []
for m in months:
    y, mo = m.split("-")
    labels.append(["Jan", "", "", "Apr", "", "", "Jul", "", "", "Oct", "", ""][int(mo) - 1] + ("\n" + y if mo in ("01", "07") or m == months[0] else ""))
b.set_xticks(list(X)); b.set_xticklabels(labels, fontsize=7.4)
b.set_xlim(-0.6, len(months) - 0.4)
out = os.path.join(ROOT, "figures", "out")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"fig2_provenance.{ext}"), dpi=300, bbox_inches="tight")
print("saved", len(months), "months")
