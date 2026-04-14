#!/usr/bin/env python3
"""
02_descriptive_analysis.py
===========================
Generates:
  - Table 1: Descriptive statistics by epidemic period
  - Figure 1: Weekly positivity rate time series with regime shading
  - Figure 2: STL-style seasonal decomposition

Requires: data/flunet_saudi_clean.csv (from script 01)
"""

import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from scipy import stats
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from utils.plot_style import COLORS, set_style, save_fig

set_style()
os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/flunet_saudi_clean.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
pr = df["positivity_rate"].fillna(0.0).values
pre  = df[df["period"] == "pre_pandemic"]
pan  = df[df["period"] == "pandemic"]
post = df[df["period"] == "post_pandemic"]
print(f"Loaded n = {len(df)} observations")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — Descriptive statistics
# ══════════════════════════════════════════════════════════════════════════════
def period_stats(sub, label):
    r = sub["positivity_rate"]
    return {
        "Period": label,
        "n_weeks": len(r),
        "mean_pct": round(r.mean()*100, 1),
        "median_pct": round(r.median()*100, 1),
        "sd_pct": round(r.std()*100, 1),
        "min_pct": round(r.min()*100, 1),
        "max_pct": round(r.max()*100, 1),
        "q25_pct": round(r.quantile(0.25)*100, 1),
        "q75_pct": round(r.quantile(0.75)*100, 1),
        "total_specimens": int(sub["SPEC_PROCESSED_NB"].sum()),
        "total_positives": int(sub["INF_ALL"].sum()),
    }

rows = [
    period_stats(pre,  "Pre-pandemic (2017–2019)"),
    period_stats(pan,  "Pandemic (2020–2021)"),
    period_stats(post, "Post-pandemic (2022–2026)"),
    period_stats(df,   "Overall (2017–2026)"),
]

# One-way ANOVA p-value
f_stat, anova_p = stats.f_oneway(
    pre["positivity_rate"].dropna(),
    pan["positivity_rate"].dropna(),
    post["positivity_rate"].dropna()
)
print(f"One-way ANOVA: F = {f_stat:.2f}, p = {anova_p:.4e}")

t1 = pd.DataFrame(rows)
t1.to_csv("tables/table1_descriptive_stats.csv", index=False)
print("Table 1 saved: tables/table1_descriptive_stats.csv")
print(t1.to_string(index=False))

# Mann-Kendall trend test
def mann_kendall(x):
    n = len(x); s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(x[j]-x[i])
    var_s = n*(n-1)*(2*n+5)/18
    z = (s-1)/np.sqrt(var_s) if s>0 else ((s+1)/np.sqrt(var_s) if s<0 else 0)
    tau = s/(0.5*n*(n-1))
    return tau, z, 2*(1-stats.norm.cdf(abs(z)))

def sens_slope(yrs, vals):
    slopes = [(vals[j]-vals[i])/(yrs[j]-yrs[i])
              for i in range(len(yrs)-1) for j in range(i+1,len(yrs))]
    return np.median(slopes)*10  # per decade

x_all = np.arange(len(df))
years = df["ISO_YEAR"].values + df["ISO_WEEK"].values/52.0
mk_tau, mk_z, mk_p = mann_kendall(pr)
slope_decade = sens_slope(years, pr)
print(f"\nMann-Kendall: tau={mk_tau:.3f}, z={mk_z:.2f}, p={mk_p:.4f}")
print(f"Sen's slope: {slope_decade:.4f} per decade")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Weekly positivity time series
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(14, 10), facecolor="white",
                          gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle(
    "Figure 1: Weekly Influenza Positivity Rate — Saudi Arabia, 2017–2026\n"
    "WHO FluNet Sentinel Surveillance (n = 476 analysable weeks; 2 excluded: SPEC = 0 or missing)",
    fontsize=11, fontweight="bold", color=COLORS["dark"], y=0.99)

ax = axes[0]
for subset, col in [(pre, COLORS["regime1"]), (pan, COLORS["regime2"]), (post, COLORS["regime3"])]:
    ax.fill_between(subset["date"], 0, subset["positivity_rate"], color=col, alpha=0.28)
    ax.plot(subset["date"], subset["positivity_rate"], color=col, lw=1.4)

roll = df["positivity_rate"].rolling(4, center=True, min_periods=1).mean()
ax.plot(df["date"], roll, color="black", lw=2.5, alpha=0.65, label="4-week rolling mean")

ax.axvline(pd.Timestamp("2020-01-01"), color="black", lw=2, ls="--", alpha=0.85)
ax.axvline(pd.Timestamp("2022-01-01"), color="black", lw=2, ls="--", alpha=0.85)
ax.text(pd.Timestamp("2020-02-01"), 0.41, "Pandemic\nonset", fontsize=8.5,
        color="black", va="top")
ax.text(pd.Timestamp("2022-02-01"), 0.41, "Post-pandemic\ntransition", fontsize=8.5,
        color="black", va="top")

patches = [
    mpatches.Patch(color=COLORS["regime1"], alpha=0.5, label="Regime 1 — Pre-pandemic (2017–2019)"),
    mpatches.Patch(color=COLORS["regime2"], alpha=0.5, label="Regime 2 — Pandemic (2020–2021)"),
    mpatches.Patch(color=COLORS["regime3"], alpha=0.5, label="Regime 3 — Post-pandemic (2022–2026)"),
    plt.Line2D([0],[0], color="black", lw=2.5, alpha=0.65, label="4-week rolling mean"),
]
ax.legend(handles=patches, loc="upper right", fontsize=8.5, framealpha=0.92)
ax.set_ylabel("Weekly Positivity Rate", fontsize=11)
ax.set_xlim(df["date"].min(), df["date"].max())
ax.set_ylim(-0.01, 0.46)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xticklabels([])
ax.set_title("NOTE: No causal interpretation implied by identified regimes. "
             "Positivity rates reflect both infection dynamics and testing behaviour.",
             fontsize=8, color="gray", style="italic", pad=4)

# Specimen volume panel
ax2 = axes[1]
ax2.bar(df["date"], df["SPEC_PROCESSED_NB"]/1000, width=5,
        color="steelblue", alpha=0.55, label="Specimens processed (×1000)")
ax2.set_ylabel("Specimens (×1000)", fontsize=9)
ax2.set_xlabel("Epidemiological Week", fontsize=11)
ax2.set_xlim(df["date"].min(), df["date"].max())
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.legend(fontsize=8.5)

plt.tight_layout(rect=[0, 0, 1, 0.97])
save_fig(fig, "fig1_timeseries.png")
print("Figure 1 saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Seasonal decomposition
# ══════════════════════════════════════════════════════════════════════════════
def moving_avg(x, w=52):
    out = np.full(len(x), np.nan)
    hw = w // 2
    for i in range(hw, len(x)-hw):
        out[i] = np.mean(x[i-hw:i+hw+1])
    out[:hw] = out[hw]; out[-hw:] = out[-hw-1]
    return out

trend_c = moving_avg(pr, 52)
detrended = pr - trend_c
df2 = df.copy(); df2["detrended"] = detrended
seas_avg = df2.groupby("ISO_WEEK")["detrended"].mean()
seasonal_c = df2["ISO_WEEK"].map(seas_avg).values
residual_c = detrended - seasonal_c

fig, axes = plt.subplots(4, 1, figsize=(14, 11), facecolor="white")
fig.suptitle(
    "Figure 2: Seasonal Decomposition of Weekly Influenza Positivity Rate\n"
    "Saudi Arabia, 2017–2026 (52-week centred moving-average; Cleveland et al., 1990)",
    fontsize=11, fontweight="bold", color=COLORS["dark"], y=0.99)

comps = [(pr, "Observed", COLORS["regime1"]),
         (trend_c, "Trend", COLORS["regime2"]),
         (seasonal_c, "Seasonal", COLORS["regime3"]),
         (residual_c, "Residual", "#E67E22")]

for ax, (data, label, col) in zip(axes, comps):
    ax.plot(df["date"], data, color=col, lw=1.5)
    if label != "Residual":
        ax.fill_between(df["date"], 0, data, color=col, alpha=0.12)
    else:
        ax.axhline(0, color="gray", lw=1, ls="--", alpha=0.5)
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2022-01-01"),
               color="#E74C3C", alpha=0.07)
    ax.set_ylabel(label, fontsize=9)
    ax.tick_params(labelsize=8)

axes[-1].set_xlabel("Epidemiological Week", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.97])
save_fig(fig, "fig2_stl_decomposition.png")
print("Figure 2 saved")
print("\nScript 02 complete.")
