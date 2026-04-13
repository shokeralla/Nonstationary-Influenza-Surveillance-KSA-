# Influenza MS-AR Saudi Arabia

## Overview
This repository provides a fully reproducible analysis pipeline for modeling nonstationary influenza surveillance data in Saudi Arabia (2017–2026) using Markov-switching autoregressive (MS-AR) models.

The study focuses on regime-aware epidemiological dynamics and forecasting under structural breaks, with application to WHO FluNet surveillance data.

---

## Data Source
Data were obtained from the WHO FluNet platform:

https://www.who.int/tools/flunet

The dataset includes weekly influenza-positive cases (INF_ALL) and total processed specimens (SPEC_PROCESSED_NB).

---

## Methods

The following models are implemented:

- Markov-Switching Autoregressive (MS-AR) model (K = 3, p = 2)
- SARIMA
- Prophet
- LSTM (with bootstrap-based uncertainty)
- Seasonal naïve benchmark

Forecast evaluation uses rolling-window validation.

---

## Reproducibility

- Fixed random seeds
- Identical training windows across models
- Consistent forecast horizons
- Bootstrap-based uncertainty estimation
- Fully reproducible pipeline

---

## How to Run

### Python

```bash
pip install -r requirements.txt
python scripts/run_analysis.py
