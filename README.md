# Influenza Surveillance MS-AR Framework — Saudi Arabia (2017–2026)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12345678.svg)](https://doi.org/10.5281/zenodo.12345678)
[![R](https://img.shields.io/badge/R-%3E%3D4.3.2-blue)](https://www.r-project.org/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)

Reproducibility repository for:

> **Shakrallah, S.A.A.** (2026). Identifiability and Estimation in Nonstationary
> Time Series with Structural Regime Transitions: A Statistical Framework Applied
> to Influenza Surveillance Data from Saudi Arabia (2017–2026).
> *Acta Tropica* (under review). DOI: 10.5281/zenodo.12345678

Corresponding author: **Ashokralla@bu.edu.sa** — Al-Baha University, KSA

---

## Repository Structure

```
influenza-msar-saudi/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements_R.txt
├── requirements_python.txt
├── data/
│   ├── README_data.md
│   └── data_preprocessing.py
├── code/
│   ├── 01_data_preprocessing.py
│   ├── 02_descriptive_analysis.py
│   ├── 03_identifiability_analysis.py
│   ├── 04_msar_estimation.py
│   ├── 05_regime_probabilities.py
│   ├── 06_forecast_benchmarks.py
│   ├── 07_simulation_study.py
│   ├── 08_sensitivity_analysis.py
│   └── utils/
│       ├── gev_functions.py
│       ├── forecast_metrics.py
│       └── plot_style.py
├── figures/
├── tables/
├── results/
└── docs/
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Dataset | 476 weekly obs., 2017–2026 (2 excluded: SPEC=0 or missing) |
| Regimes | K = 3 |
| Pre-pandemic mean positivity | 16.1% |
| Pandemic mean positivity | 3.7% |
| Post-pandemic mean positivity | 9.5% |
| Identifiability threshold Δμ/σ̄ | ≥ 1.5 (empirical guideline) |
| Empirical Δμ/σ̄ | 1.63 ✅ |
| MS-AR RMSE (4-week) | **0.0656** vs AR(2) 0.0856 → −23.4% |
| MS-AR RMSE (8-week) | **0.0721** vs AR(2) 0.0961 → −25.0% |
| MS-AR CRPS (4-week) | **0.0349** vs AR(2) 0.0484 |
| DM test p-value (4-week) | **p = 0.009** |

> **Note:** No causal interpretation is implied by the identified statistical regimes.
> These are empirically supported guidelines, not formal theorems.

---

## Quick Start

```bash
git clone https://github.com/shakrallah/influenza-msar-saudi.git
cd influenza-msar-saudi
pip install -r requirements_python.txt
python code/01_data_preprocessing.py  # downloads & cleans WHO FluNet data
python code/02_descriptive_analysis.py
python code/03_identifiability_analysis.py
python code/04_msar_estimation.py
python code/05_regime_probabilities.py
python code/06_forecast_benchmarks.py
python code/07_simulation_study.py
python code/08_sensitivity_analysis.py
```

Or use Docker:
```bash
docker pull shakrallah/influenza-msar-saudi:latest
docker run -v $(pwd)/results:/app/results shakrallah/influenza-msar-saudi:latest
```

---

## Data

Source: **WHO FluNet** (public domain)
URL: https://www.who.int/tools/flunet
Period: 2015–2026, weekly sentinel surveillance, Saudi Arabia

**Exclusion rule:** Weeks with `SPEC_PROCESSED_NB = 0` or missing excluded
(2 records, 0.4% of raw data).

Primary outcome: `positivity_rate = INF_ALL / SPEC_PROCESSED_NB`

---

## Citation

```bibtex
@article{shakrallah2026influenza,
  author  = {Shakrallah, Sheikh Abdulbaqi Ahmed},
  title   = {Identifiability and Estimation in Nonstationary Time Series with
             Structural Regime Transitions: A Statistical Framework Applied to
             Influenza Surveillance Data from Saudi Arabia (2017--2026)},
  journal = {Acta Tropica},
  year    = {2026},
  note    = {Under review},
  doi     = {10.5281/zenodo.12345678}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

## Disclaimer

Results are observational. No causal interpretation is implied.
All identifiability thresholds are empirically supported practical guidelines,
not formal mathematical theorems. Positivity rates reflect both infection
dynamics and testing behaviour.
