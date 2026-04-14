# =============================================================================
# Script 01: Data Preprocessing and Descriptive Analysis
# Project: Identifiability and Estimation in Nonstationary Time Series with
#          Structural Regime Transitions
# Applied to: Saudi Arabia Influenza Surveillance Data (2017-2026)
# Journal: Acta Tropica (Elsevier)
# Author: Sheikh Abdulbaqi Ahmed Shakrallah
# Email: Ashokralla@bu.edu.sa | Al-Baha University, KSA
# Date: 2026
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Load Required Packages
# -----------------------------------------------------------------------------
library(readxl)       # Data import
library(dplyr)        # Data manipulation
library(tidyr)        # Data reshaping
library(lubridate)    # Date handling
library(ggplot2)      # Visualization
library(Kendall)      # Mann-Kendall trend test
library(trend)        # Sen's slope estimator
library(car)          # VIF computation
library(moments)      # Skewness and kurtosis

set.seed(2026)

# -----------------------------------------------------------------------------
# 1. Data Import and Initial Cleaning
# -----------------------------------------------------------------------------
# Load raw WHO FluNet sentinel surveillance data
# Source: https://www.who.int/tools/flunet
raw_data <- read_excel("data/influenza_saudi_arabia_2015_2026.xlsx")

cat("=== RAW DATA OVERVIEW ===\n")
cat("Total raw records:", nrow(raw_data), "\n")
cat("Columns:", paste(names(raw_data), collapse = ", "), "\n")

# -----------------------------------------------------------------------------
# 2. Exclusion Criteria (MANDATORY per study protocol)
# RULE: If SPEC_PROCESSED_NB = 0 or missing, exclude the week from analysis
# -----------------------------------------------------------------------------
excluded <- raw_data %>%
  filter(is.na(SPEC_PROCESSED_NB) | SPEC_PROCESSED_NB == 0)

cat("\n=== EXCLUSION REPORT ===\n")
cat("Records excluded (SPEC_PROCESSED_NB = 0 or NA):", nrow(excluded), "\n")
cat("Percentage excluded:", round(nrow(excluded) / nrow(raw_data) * 100, 2), "%\n")

clean_data <- raw_data %>%
  filter(!is.na(SPEC_PROCESSED_NB) & SPEC_PROCESSED_NB > 0)

cat("Records retained after exclusion:", nrow(clean_data), "\n")

# -----------------------------------------------------------------------------
# 3. Variable Construction
# -----------------------------------------------------------------------------
clean_data <- clean_data %>%
  mutate(
    # Primary outcome: weekly positivity rate (Equation 1 in manuscript)
    positivity_rate = ifelse(
      is.na(INF_ALL), 0,
      INF_ALL / SPEC_PROCESSED_NB
    ),
    # Parse date
    date = as.Date(ISO_SDATE),
    # Epidemiological period classification (a priori definition)
    period = case_when(
      ISO_YEAR < 2020                          ~ "Pre-pandemic (2017-2019)",
      ISO_YEAR >= 2020 & ISO_YEAR <= 2021      ~ "Pandemic (2020-2021)",
      ISO_YEAR >= 2022                          ~ "Post-pandemic (2022-2026)",
      TRUE                                     ~ NA_character_
    ),
    period = factor(period, levels = c(
      "Pre-pandemic (2017-2019)",
      "Pandemic (2020-2021)",
      "Post-pandemic (2022-2026)"
    ))
  ) %>%
  arrange(date)

# Restrict to main analytic period (2017+)
analytic_data <- clean_data %>%
  filter(ISO_YEAR >= 2017)

cat("\n=== ANALYTIC DATASET ===\n")
cat("Main analytic observations (2017-2026):", nrow(analytic_data), "\n")
cat("Date range:", format(min(analytic_data$date)), "to",
    format(max(analytic_data$date)), "\n")

# -----------------------------------------------------------------------------
# 4. Descriptive Statistics by Epidemiological Period (Table 1)
# -----------------------------------------------------------------------------
desc_stats <- analytic_data %>%
  group_by(period) %>%
  summarise(
    n_weeks         = n(),
    mean_pos        = round(mean(positivity_rate) * 100, 1),
    median_pos      = round(median(positivity_rate) * 100, 1),
    sd_pos          = round(sd(positivity_rate) * 100, 1),
    min_pos         = round(min(positivity_rate) * 100, 1),
    max_pos         = round(max(positivity_rate) * 100, 1),
    q25             = round(quantile(positivity_rate, 0.25) * 100, 1),
    q75             = round(quantile(positivity_rate, 0.75) * 100, 1),
    total_specimens = sum(SPEC_PROCESSED_NB, na.rm = TRUE),
    total_positive  = sum(INF_ALL, na.rm = TRUE),
    skewness_val    = round(skewness(positivity_rate), 3),
    kurtosis_val    = round(kurtosis(positivity_rate) - 3, 3),  # Excess kurtosis
    .groups = "drop"
  )

cat("\n=== TABLE 1: DESCRIPTIVE STATISTICS BY PERIOD ===\n")
print(desc_stats, width = 120)

# Overall statistics
overall_stats <- analytic_data %>%
  summarise(
    period          = "Overall (2017-2026)",
    n_weeks         = n(),
    mean_pos        = round(mean(positivity_rate) * 100, 1),
    median_pos      = round(median(positivity_rate) * 100, 1),
    sd_pos          = round(sd(positivity_rate) * 100, 1),
    min_pos         = round(min(positivity_rate) * 100, 1),
    max_pos         = round(max(positivity_rate) * 100, 1),
    q25             = round(quantile(positivity_rate, 0.25) * 100, 1),
    q75             = round(quantile(positivity_rate, 0.75) * 100, 1),
    total_specimens = sum(SPEC_PROCESSED_NB, na.rm = TRUE),
    total_positive  = sum(INF_ALL, na.rm = TRUE),
    skewness_val    = round(skewness(positivity_rate), 3),
    kurtosis_val    = round(kurtosis(positivity_rate) - 3, 3)
  )

cat("\n=== OVERALL STATISTICS ===\n")
print(overall_stats)

# Save Table 1 to CSV
table1 <- bind_rows(desc_stats, overall_stats)
write.csv(table1, "tables/Table1_DescriptiveStatistics.csv", row.names = FALSE)
cat("\nTable 1 saved to tables/Table1_DescriptiveStatistics.csv\n")

# -----------------------------------------------------------------------------
# 5. Trend Tests (Mann-Kendall + Sen's Slope)
# -----------------------------------------------------------------------------
cat("\n=== TREND ANALYSIS ===\n")

for (per in c("Pre-pandemic (2017-2019)", "Pandemic (2020-2021)",
              "Post-pandemic (2022-2026)", "Overall")) {
  if (per == "Overall") {
    d <- analytic_data
  } else {
    d <- analytic_data %>% filter(period == per)
  }
  pr <- d$positivity_rate
  mk  <- MannKendall(pr)
  sen <- sens.slope(ts(pr))
  cat(sprintf("[%s] MK tau=%.3f, p=%.4f | Sen's slope=%.4f/week (%.3f/decade)\n",
              per, mk$tau, mk$sl, sen$estimates, sen$estimates * 10 * 52))
}

# -----------------------------------------------------------------------------
# 6. Sensitivity Analysis: Testing Volume vs Positivity Rate
# -----------------------------------------------------------------------------
cat("\n=== SENSITIVITY: SPECIMEN VOLUME VS POSITIVITY RATE ===\n")
sp_test <- cor.test(analytic_data$SPEC_PROCESSED_NB,
                    analytic_data$positivity_rate,
                    method = "spearman")
cat(sprintf("Spearman rho = %.4f, p-value = %.4f\n",
            sp_test$estimate, sp_test$p.value))
cat("Interpretation: Positivity rates reflect both infection dynamics and",
    "testing behaviour.\nStructural changes in testing intensity may induce",
    "apparent regime shifts.\nResidual testing-effort confounding cannot be",
    "ruled out from observational data alone.\n")

# -----------------------------------------------------------------------------
# 7. Subtype Distribution
# -----------------------------------------------------------------------------
cat("\n=== SUBTYPE DISTRIBUTION ===\n")
subtypes <- c("AH1N12009", "AH3", "AH1", "BVIC", "BYAM")
total_typed <- sum(analytic_data$INF_ALL, na.rm = TRUE)
for (s in subtypes) {
  if (s %in% names(analytic_data)) {
    n <- sum(analytic_data[[s]], na.rm = TRUE)
    cat(sprintf("  %s: %.0f (%.1f%% of total positives)\n",
                s, n, n / total_typed * 100))
  }
}

# -----------------------------------------------------------------------------
# 8. Save Clean Dataset
# -----------------------------------------------------------------------------
saveRDS(analytic_data, "data/analytic_data_clean.rds")
write.csv(analytic_data %>%
            select(ISO_YEAR, ISO_WEEK, date, period,
                   SPEC_PROCESSED_NB, INF_ALL, INF_A, INF_B,
                   positivity_rate, AH1N12009, AH3, BVIC),
          "data/analytic_data_clean.csv", row.names = FALSE)

cat("\n=== DATA SAVED ===\n")
cat("RDS: data/analytic_data_clean.rds\n")
cat("CSV: data/analytic_data_clean.csv\n")
cat("\nScript 01 complete.\n")
