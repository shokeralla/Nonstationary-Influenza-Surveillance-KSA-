#!/usr/bin/env python3
"""
01_data_preprocessing.py
========================
Load, clean, and prepare WHO FluNet sentinel data for Saudi Arabia (2017-2026).

Data source: https://www.who.int/tools/flunet
Download manually: Country=Saudi Arabia, Period=2015-2026, export CSV/Excel

Outputs:
    data/flunet_saudi_clean.csv  — cleaned analytic dataset (n=476)
    results/exclusion_log.txt   — records of excluded weeks
"""

import pandas as pd
import numpy as np
import os, sys

# ── Configuration ─────────────────────────────────────────────────────────────
RAW_FILE   = "data/flunet_saudi_raw.xlsx"   # place your downloaded file here
OUTPUT_CSV = "data/flunet_saudi_clean.csv"
EXCL_LOG   = "results/exclusion_log.txt"
START_YEAR = 2017   # first complete year for modelling

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ── Load raw data ─────────────────────────────────────────────────────────────
print("Loading raw WHO FluNet data...")
try:
    df = pd.read_excel(RAW_FILE)
except FileNotFoundError:
    sys.exit(f"\nERROR: Raw data file not found at '{RAW_FILE}'.\n"
             "Please download from https://www.who.int/tools/flunet\n"
             "and save as 'data/flunet_saudi_raw.xlsx'.")

print(f"  Raw records loaded: {len(df)}")

# ── Exclusion rule: SPEC_PROCESSED_NB = 0 or missing ─────────────────────────
mask_excl = (df["SPEC_PROCESSED_NB"].isna()) | (df["SPEC_PROCESSED_NB"] == 0)
excluded = df[mask_excl].copy()
df_clean = df[~mask_excl].copy()

with open(EXCL_LOG, "w") as f:
    f.write("EXCLUSION LOG — WHO FluNet Saudi Arabia Data Cleaning\n")
    f.write("="*60 + "\n")
    f.write(f"Total raw records:  {len(df)}\n")
    f.write(f"Records excluded:   {len(excluded)} ({len(excluded)/len(df)*100:.1f}%)\n")
    f.write(f"Exclusion reason:   SPEC_PROCESSED_NB = 0 or missing\n\n")
    if len(excluded) > 0:
        f.write("Excluded records:\n")
        f.write(excluded[["ISO_YEAR","ISO_WEEK","ISO_SDATE",
                           "SPEC_PROCESSED_NB"]].to_string(index=False))
print(f"  Excluded (SPEC=0 or missing): {len(excluded)}")
print(f"  Exclusion log saved: {EXCL_LOG}")

# ── Derive primary outcome ────────────────────────────────────────────────────
df_clean["positivity_rate"] = (
    df_clean["INF_ALL"] / df_clean["SPEC_PROCESSED_NB"]
).fillna(0.0)

# ── Parse dates ───────────────────────────────────────────────────────────────
df_clean["date"] = pd.to_datetime(df_clean["ISO_SDATE"])
df_clean = df_clean.sort_values("date").reset_index(drop=True)

print(f"  Date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")

# ── Restrict to analytic period (≥ 2017) ─────────────────────────────────────
df_model = df_clean[df_clean["ISO_YEAR"] >= START_YEAR].reset_index(drop=True)
print(f"  Analytic dataset (≥ {START_YEAR}): n = {len(df_model)}")

# ── Define epidemiological periods ───────────────────────────────────────────
def assign_period(year):
    if year < 2020:  return "pre_pandemic"
    elif year <= 2021: return "pandemic"
    else:              return "post_pandemic"

df_model["period"] = df_model["ISO_YEAR"].map(assign_period)

# ── Summary statistics ────────────────────────────────────────────────────────
print("\nPeriod summary:")
for prd in ["pre_pandemic", "pandemic", "post_pandemic"]:
    sub = df_model[df_model["period"] == prd]["positivity_rate"]
    print(f"  {prd:>15}  n={len(sub):>4}  "
          f"mean={sub.mean()*100:5.1f}%  sd={sub.std()*100:4.1f}%  "
          f"spec={df_model[df_model['period']==prd]['SPEC_PROCESSED_NB'].sum():>8,.0f}")

# ── Save cleaned dataset ──────────────────────────────────────────────────────
df_model.to_csv(OUTPUT_CSV, index=False)
print(f"\nCleaned dataset saved: {OUTPUT_CSV}")
print("Done.")
