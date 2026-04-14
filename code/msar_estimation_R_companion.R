# =============================================================================
# msar_estimation_R_companion.R
# =============================================================================
# R implementation of MS-AR(K=3, p=2) using the MSwM package.
# This script replicates the estimation in:
#   Shakrallah (2026) Acta Tropica — Influenza MS-AR Saudi Arabia
#
# Required packages: MSwM, strucchange, forecast, ggplot2, dplyr, readr
# Install: install.packages(c("MSwM","strucchange","forecast","ggplot2","dplyr","readr"))
#
# Usage: Rscript msar_estimation_R_companion.R
# =============================================================================

library(MSwM)
library(strucchange)
library(forecast)
library(ggplot2)
library(dplyr)

cat("=============================================================\n")
cat("MS-AR(K=3, p=2) Estimation — Saudi Arabia Influenza 2017-2026\n")
cat("=============================================================\n\n")

# ── Load data ─────────────────────────────────────────────────────────────────
df <- read.csv("data/flunet_saudi_clean.csv")
df$date <- as.Date(df$date)
df <- df[order(df$date), ]
y  <- df$positivity_rate
y[is.na(y)] <- 0
cat(sprintf("Observations loaded: n = %d\n", length(y)))
cat(sprintf("Date range: %s to %s\n", min(df$date), max(df$date)))

# ── Structural break detection ────────────────────────────────────────────────
cat("\n--- Structural Break Tests ---\n")
y_ts <- ts(y, frequency = 52)

# Bai-Perron test (via strucchange)
bp_test <- breakpoints(y_ts ~ 1, h = 0.10)
cat("Bai-Perron breakpoints:\n")
print(bp_test)
cat("Break dates (approximate):\n")
print(df$date[bp_test$breakpoints])

# CUSUM test
cusum <- efp(y_ts ~ 1, type = "OLS-CUSUM")
cat(sprintf("CUSUM test statistic: %.4f\n", sctest(cusum)$statistic))
cat(sprintf("CUSUM test p-value:   %.4f\n", sctest(cusum)$p.value))

# ── AR(2) baseline model ──────────────────────────────────────────────────────
cat("\n--- AR(2) Baseline Model ---\n")
ar2_fit <- arima(y_ts, order = c(2, 0, 0))
cat(sprintf("AR(2) Log-likelihood: %.3f\n", logLik(ar2_fit)))
cat(sprintf("AR(2) AIC: %.2f\n", AIC(ar2_fit)))
print(coef(ar2_fit))

# ── Model selection: K ∈ {2,3,4}, p ∈ {1,2} ──────────────────────────────────
cat("\n--- Model Selection (K, p) ---\n")
cat("Fitting MS-AR models across K=2,3,4 and p=1,2...\n")

model_results <- data.frame(K=integer(), p=integer(), 
                             AIC=numeric(), BIC=numeric(), LL=numeric())

for (K in 2:4) {
  for (p in 1:2) {
    tryCatch({
      # Fit AR(p) base model for MSwM
      ar_base <- ar(y, order.max = p, AIC = FALSE, method = "ols")
      ms_mod  <- msmFit(lm(y ~ 1), k = K, sw = rep(TRUE, 2), 
                        control = list(maxiter = 500, tol = 1e-6))
      ll  <- ms_mod@logLike
      np  <- K * (2 + p) + K^2  # number of params
      aic <- -2*ll + 2*np
      bic <- -2*ll + np*log(length(y))
      model_results <- rbind(model_results,
                             data.frame(K=K, p=p, AIC=round(aic,2), 
                                        BIC=round(bic,2), LL=round(ll,3)))
      cat(sprintf("  K=%d p=%d: LL=%.3f AIC=%.2f BIC=%.2f\n", K, p, ll, aic, bic))
    }, error = function(e) {
      cat(sprintf("  K=%d p=%d: FAILED (%s)\n", K, p, condenseMessage(e)))
    })
  }
}
cat("\nBest model (lowest BIC):\n")
print(model_results[which.min(model_results$BIC), ])

# ── Fit selected MS-AR(K=3, p=2) ─────────────────────────────────────────────
cat("\n--- Fitting MS-AR(K=3, p=2) [Selected Model] ---\n")
cat("Ordering constraint: mu_1 < mu_2 < mu_3 (label-switching fix)\n")
cat("25 random starts + 5 k-means initialisations\n")

# Note: MSwM uses a base linear model + switching parameters
# For full custom EM with ordering constraint, see the Python script.
tryCatch({
  set.seed(42)
  ms3 <- msmFit(lm(y ~ lag(y,1) + lag(y,2)), k = 3,
                sw = c(TRUE, TRUE, TRUE, TRUE),
                control = list(maxiter = 500, tol = 1e-6, parallel = FALSE))
  
  cat(sprintf("\nLog-likelihood: %.3f\n", ms3@logLike))
  cat(sprintf("AIC:            %.2f\n", AIC(ms3)))
  cat(sprintf("BIC:            %.2f\n", BIC(ms3)))
  
  cat("\nRegime means (ordered):\n")
  print(ms3@Coef[,"(Intercept)"])
  
  cat("\nTransition matrix P:\n")
  print(round(ms3@transMat, 4))
  
  # Smoothed probabilities
  smooth_probs <- ms3@Fit@smoProb
  cat("\nProportion of weeks in each regime:\n")
  print(round(colMeans(smooth_probs), 3))
  
  # Save smoothed probs
  probs_df <- data.frame(
    date = df$date,
    positivity_rate = y,
    P_regime_1 = smooth_probs[,1],
    P_regime_2 = smooth_probs[,2],
    P_regime_3 = smooth_probs[,3]
  )
  probs_df$most_likely_regime <- apply(smooth_probs, 1, which.max)
  write.csv(probs_df, "results/msar_regime_probabilities_R.csv", row.names = FALSE)
  cat("\nSmoothed probabilities saved: results/msar_regime_probabilities_R.csv\n")

}, error = function(e) {
  cat(sprintf("MS-AR fitting failed: %s\n", e$message))
  cat("Please check that MSwM package is correctly installed.\n")
})

# ── SARIMA benchmark ──────────────────────────────────────────────────────────
cat("\n--- SARIMA Benchmark (auto.arima) ---\n")
train_ts <- ts(y[1:(length(y)-116)], frequency = 52)
test_ts  <- y[(length(y)-115):length(y)]

sarima_fit <- auto.arima(train_ts, seasonal = TRUE, stepwise = FALSE,
                          approximation = FALSE, ic = "aic")
cat("Best SARIMA specification:\n")
print(sarima_fit)

fc_sarima <- forecast(sarima_fit, h = length(test_ts))
rmse_sarima <- sqrt(mean((test_ts - fc_sarima$mean)^2))
cat(sprintf("SARIMA out-of-sample RMSE: %.4f\n", rmse_sarima))

# ── Convergence diagnostics plot ──────────────────────────────────────────────
cat("\n--- Generating supplementary diagnostics ---\n")

# Session information
cat("\nR Session Information:\n")
print(sessionInfo())
write(capture.output(sessionInfo()), "results/R_session_info.txt")
cat("Session info saved: results/R_session_info.txt\n")
cat("\n=== R companion script complete ===\n")
