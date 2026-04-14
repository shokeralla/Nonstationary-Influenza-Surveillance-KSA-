# Data Documentation

## Source
- **Platform:** WHO FluNet Global Influenza Surveillance System
- **URL:** https://www.who.int/tools/flunet
- **System:** WHO GISRS sentinel surveillance
- **Country:** Saudi Arabia (ISO code: SAU)
- **Region:** Eastern Mediterranean (EMR)
- **Surveillance type:** SENTINEL
- **Temporal resolution:** Weekly (ISO epidemiological weeks)
- **Raw period:** January 2015 – March 2026 (481 raw records)
- **Analytic period:** ISO Week 1, 2017 – ISO Week 11, 2026 (476 valid records)

## Data Access Instructions

```python
# Option 1: Direct download via WHO FluNet
# Navigate to: https://www.who.int/tools/flunet
# Select: Country = Saudi Arabia | Period = 2015-2026 | Export CSV

# Option 2: Use our preprocessing script (auto-downloads if possible)
python code/01_data_preprocessing.py
```

## Variables

| Variable | Type | Description | Units |
|----------|------|-------------|-------|
| `WHO_REGION` | str | WHO geographic region | — |
| `ITZ` | str | Influenza transmission zone | — |
| `COUNTRY_CODE` | str | ISO 3-letter country code | — |
| `ISO_YEAR` | int | ISO epidemiological year | — |
| `ISO_WEEK` | int | ISO epidemiological week (1–53) | — |
| `ISO_SDATE` | datetime | Start date of epidemiological week | — |
| `SPEC_RECEIVED_NB` | int | Specimens received | count |
| `SPEC_PROCESSED_NB` | int | Specimens processed (tested) | count |
| `AH1` | float | Influenza A/H1 (seasonal) positives | count |
| `AH1N12009` | float | Influenza A/H1N1pdm09 positives | count |
| `AH3` | float | Influenza A/H3N2 positives | count |
| `AH5` | float | Influenza A/H5 positives | count |
| `ANOTSUBTYPED` | float | Influenza A not subtyped | count |
| `INF_A` | float | Total influenza A positives | count |
| `BVIC` | float | Influenza B/Victoria lineage positives | count |
| `BYAM` | float | Influenza B/Yamagata lineage positives | count |
| `BNOTDETERMINED` | float | Influenza B lineage not determined | count |
| `INF_B` | float | Total influenza B positives | count |
| `INF_ALL` | float | Total influenza positives (A + B) | count |
| `INF_NEGATIVE` | float | Influenza-negative specimens | count |
| `ILI_ACTIVITY` | int | ILI activity level (1 = active) | binary |
| **`positivity_rate`** | **float** | **DERIVED: INF_ALL / SPEC_PROCESSED_NB** | **[0, 1]** |

## Exclusion Rules

```
IF SPEC_PROCESSED_NB = 0 OR SPEC_PROCESSED_NB IS NULL:
    → Exclude week from analysis
    → Document in exclusion log

Result: 2 records excluded (0.4% of raw dataset)
```

## Epidemiological Periods

| Period | ISO Years | n weeks | Definition basis |
|--------|-----------|---------|-----------------|
| Pre-pandemic | 2017–2019 | 156 | Prior to COVID-19 NPI introduction |
| Pandemic | 2020–2021 | 100 | NPI active period, suppressed influenza |
| Post-pandemic | 2022–2026 | 220 | Progressive NPI removal, Hajj reopened |

## Important Caveat

> Positivity rates reflect **both infection dynamics and testing behaviour**.
> Structural changes in testing intensity (e.g., the large expansion of specimen
> volume from ~7,000/year in 2021 to ~42,000–48,000/year in 2024–2025) may
> induce apparent regime shifts that do not correspond to true epidemiological
> transitions. Results should be interpreted with this limitation in mind.

## Subtype Summary (2017–2026)

| Subtype | Total detections | % of typed |
|---------|-----------------|------------|
| A/H1N1pdm09 | 5,459 | 25.2% |
| A/H3N2 | 3,730 | 17.3% |
| B/Victoria | 1,375 | 6.4% |
| B/Yamagata | 0 | 0.0% |
| A/H1 (seasonal) | 4 | 0.0% |
