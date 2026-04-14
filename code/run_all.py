#!/usr/bin/env python3
"""
run_all.py
==========
Master pipeline script — runs all analysis steps in sequence.

Shakrallah (2026) Acta Tropica — Influenza MS-AR Saudi Arabia
DOI: 10.5281/zenodo.12345678
GitHub: https://github.com/shakrallah/influenza-msar-saudi

Usage:
    python code/run_all.py

Prerequisites:
    pip install -r requirements_python.txt
    Place raw WHO FluNet data at: data/flunet_saudi_raw.xlsx
    Download from: https://www.who.int/tools/flunet

Estimated runtime: ~10 minutes (S=50 simulation replicates)
For publication: set S=500 in code/07_simulation_study.py (~30 min)
"""

import subprocess, sys, time, os

SCRIPTS = [
    ("Step 1/8: Data preprocessing & exclusion",      "code/01_data_preprocessing.py"),
    ("Step 2/8: Descriptive analysis (Table 1, Figs 1-2)", "code/02_descriptive_analysis.py"),
    ("Step 3/8: Identifiability analysis (Table 2, Fig 3)", "code/03_identifiability_analysis.py"),
    ("Step 4/8: MS-AR(K=3,p=2) estimation (Table 3)", "code/04_msar_estimation.py"),
    ("Step 5/8: Regime probability plots (Figs 4A-4B)", "code/05_regime_probabilities.py"),
    ("Step 6/8: Forecast benchmarks (Table 4, Fig 5)",  "code/06_forecast_benchmarks.py"),
    ("Step 7/8: Simulation study (identifiability)",    "code/07_simulation_study.py"),
    ("Step 8/8: Sensitivity & robustness analysis",     "code/08_sensitivity_analysis.py"),
]

print("="*65)
print("  Influenza MS-AR Saudi Arabia — Full Analysis Pipeline")
print("  Shakrallah (2026) Acta Tropica")
print("="*65)
print()

errors = []
for step_label, script in SCRIPTS:
    print(f"\n{'─'*65}")
    print(f"  {step_label}")
    print(f"  Script: {script}")
    print(f"{'─'*65}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  ✅ Completed in {elapsed:.1f}s")
    else:
        print(f"  ❌ FAILED (exit code {result.returncode})")
        errors.append(script)

print(f"\n{'='*65}")
if errors:
    print(f"  Pipeline finished with {len(errors)} error(s):")
    for e in errors: print(f"    ✗ {e}")
else:
    print("  ✅ All 8 steps completed successfully!")
    print()
    print("  Output locations:")
    print("    figures/   — Figures 1–5 (PNG, 200 dpi)")
    print("    tables/    — Tables 1–4 (CSV)")
    print("    results/   — Metrics JSON, regime probabilities")
    print()
    print("  ⚠️  DISCLAIMER: No causal interpretation is implied by")
    print("  identified statistical regimes. All identifiability thresholds")
    print("  are empirically supported guidelines, not formal theorems.")
print("="*65)
