#!/usr/bin/env python3
"""
05_regime_probabilities.py
===========================
Generates:
  - Figure 4A: Regime overlay on observed time series
  - Figure 4B: Smoothed regime membership probabilities

Requires: results/msar_regime_probabilities.csv (from script 04)
"""
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from scipy.ndimage import uniform_filter1d
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from utils.plot_style import COLORS, set_style, save_fig

set_style()
os.makedirs("figures", exist_ok=True)

df   = pd.read_csv("data/flunet_saudi_clean.csv", parse_dates=["date"])
prob = pd.read_csv("results/msar_regime_probabilities.csv", parse_dates=["date"])
df   = df.sort_values("date").reset_index(drop=True)
prob = prob.sort_values("date").reset_index(drop=True)
pr = df["positivity_rate"].fillna(0.0).values
p1 = uniform_filter1d(prob["P_regime_1"].values, 9)
p2 = uniform_filter1d(prob["P_regime_2"].values, 9)
p3 = uniform_filter1d(prob["P_regime_3"].values, 9)
tot = p1+p2+p3; p1/=tot; p2/=tot; p3/=tot
most_likely = np.argmax(np.stack([p1,p2,p3],1), axis=1)
regime_colors = [COLORS["regime1"], COLORS["regime2"], COLORS["regime3"]]

# ── FIGURE 4A: Regime overlay ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
fig.suptitle(
    "Figure 4A: Observed Positivity Rate with Most-Likely Regime Background\n"
    "NOTE: No causal interpretation is implied. Background colour = dominant regime at each time point.",
    fontsize=10, fontweight="bold", color=COLORS["dark"])

dates = df["date"].values
for i in range(len(dates)-1):
    ax.axvspan(dates[i], dates[i+1], color=regime_colors[most_likely[i]], alpha=0.20)

ax.plot(df["date"], pr, color="black", lw=1.8, label="Observed positivity rate", zorder=5)
roll = df["positivity_rate"].rolling(4, center=True, min_periods=1).mean()
ax.plot(df["date"], roll, color="black", lw=2.5, ls="--", alpha=0.65,
        label="4-week rolling mean", zorder=4)

patches = [
    mpatches.Patch(facecolor=COLORS["regime1"], alpha=0.35, label="Regime 1 — Pre-pandemic"),
    mpatches.Patch(facecolor=COLORS["regime2"], alpha=0.35, label="Regime 2 — Pandemic"),
    mpatches.Patch(facecolor=COLORS["regime3"], alpha=0.35, label="Regime 3 — Post-pandemic"),
    plt.Line2D([0],[0],color="black",lw=1.8, label="Observed"),
    plt.Line2D([0],[0],color="black",lw=2,ls="--", label="Rolling mean"),
]
ax.legend(handles=patches, loc="upper right", fontsize=9, framealpha=0.92)
ax.set_ylabel("Weekly Positivity Rate", fontsize=11)
ax.set_xlabel("Epidemiological Week", fontsize=11)
ax.set_xlim(df["date"].min(), df["date"].max())
ax.set_ylim(-0.01, 0.46)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0%}"))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout(rect=[0,0,1,0.93])
save_fig(fig, "fig4a_regime_overlay.png")
print("Figure 4A saved")

# ── FIGURE 4B: Smoothed probabilities ─────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 11), facecolor="white")
fig.suptitle(
    "Figure 4B: Smoothed Regime Membership Probabilities — MS-AR(2), K = 3\n"
    "Saudi Arabia Weekly Influenza Positivity Rate, 2017–2026",
    fontsize=11, fontweight="bold", color=COLORS["dark"], y=0.99)

ax0 = axes[0]
ax0.plot(df["date"], pr, color="black", lw=1.5)
ax0.fill_between(df["date"], 0, pr, color="gray", alpha=0.2)
ax0.set_ylabel("Positivity Rate", fontsize=9)
ax0.set_title("Observed Weekly Positivity Rate", fontsize=9)

for ax_r, p, label, col in [
    (axes[1], p1, "Regime 1: Pre-pandemic — High Seasonal Transmission", COLORS["regime1"]),
    (axes[2], p2, "Regime 2: Pandemic — Suppressed Transmission",        COLORS["regime2"]),
    (axes[3], p3, "Regime 3: Post-pandemic — Elevated Baseline",         COLORS["regime3"])]:
    ax_r.fill_between(df["date"], 0, p, color=col, alpha=0.5)
    ax_r.plot(df["date"], p, color=col, lw=1.5)
    ax_r.set_ylim(0, 1.05)
    ax_r.set_ylabel("P(Regime)", fontsize=9)
    ax_r.set_title(label, fontsize=9, color=col)
    for yr in ["2020-01-01","2022-01-01"]:
        ax_r.axvline(pd.Timestamp(yr), color="black", lw=1.5, ls="--", alpha=0.7)

axes[-1].set_xlabel("Epidemiological Week", fontsize=10)
for ax in axes:
    ax.grid(True, alpha=0.22); ax.tick_params(labelsize=8)
plt.tight_layout(rect=[0,0,1,0.97])
save_fig(fig, "fig4b_smoothed_probabilities.png")
print("Figure 4B saved\nScript 05 complete.")
