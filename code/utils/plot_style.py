"""
plot_style.py — Consistent figure styling for all manuscript figures.
Shakrallah (2026) Acta Tropica — Influenza MS-AR Saudi Arabia
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

COLORS = {
    "regime1": "#2980B9",  # Blue  — Pre-pandemic
    "regime2": "#C0392B",  # Red   — Pandemic
    "regime3": "#27AE60",  # Green — Post-pandemic
    "msar":    "#27AE60",
    "ar2":     "#C0392B",
    "sarima":  "#2980B9",
    "prophet": "#E67E22",
    "lstm":    "#8E44AD",
    "dark":    "#2C3E50",
}

def set_style():
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.28, "grid.linestyle": "--",
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "legend.framealpha": 0.92,
        "lines.linewidth": 1.8, "savefig.dpi": 200,
        "savefig.bbox": "tight", "savefig.facecolor": "white",
    })

def save_fig(fig, filename, dpi=200):
    import os; os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/{filename}", dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  Saved: figures/{filename}")
    plt.close(fig)
