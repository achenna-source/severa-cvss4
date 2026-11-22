"""Common figure identity, matched to the manuscript: Franklin Gothic text, Palatino mathematics, the Okabe-Ito
colour-blind-safe palette for the provenance groups, and the deep vermillion accent of the manuscript for SEVERA."""
import matplotlib

INK, MUTED, AXIS, GRID = "#1A1A1A", "#6B6B6B", "#4D4D4D", "#E6E6E6"
ACCENT, ACCENT_TINT = "#B4441C", "#FBEDE6"
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", sky="#56B4E9", vermillion="#D55E00",
          purple="#CC79A7", yellow="#F0E442")
GROUP_COLOR = {"VulDB": OI["blue"], "VulnCheck": OI["orange"], "GitHub": OI["green"], "Other": "#A6A6A6"}
TEXT, TEXT_M = "Franklin Gothic Book", "Franklin Gothic Medium"


def apply():
    matplotlib.rcParams.update({
        "font.family": TEXT,
        "font.size": 8.5,
        "text.color": INK,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Palatino Linotype",
        "mathtext.it": "Palatino Linotype:italic",
        "mathtext.bf": "Palatino Linotype:bold",
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK,
        "axes.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def panel_label(ax, letter, x=-0.075, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12, fontfamily=TEXT_M, color=INK, va="bottom", ha="left")


def title(ax, text):
    ax.set_title(text, fontsize=9.5, fontfamily=TEXT_M, color=INK, loc="left", pad=8)
