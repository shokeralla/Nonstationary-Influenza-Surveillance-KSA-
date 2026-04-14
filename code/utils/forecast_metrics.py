"""
forecast_metrics.py — RMSE, MAE, CRPS, PI coverage utilities.
Shakrallah (2026) Acta Tropica
"""
import numpy as np
from scipy import stats

def rmse(obs, fc): return float(np.sqrt(np.mean((np.asarray(obs)-np.asarray(fc))**2)))
def mae(obs, fc):  return float(np.mean(np.abs(np.asarray(obs)-np.asarray(fc))))

def crps_gaussian(obs, mu, sigma):
    """CRPS under Gaussian forecast distribution."""
    obs = np.asarray(obs); mu = np.asarray(mu); sigma = np.asarray(sigma)
    z = (obs - mu) / sigma
    crps_val = sigma*(z*(2*stats.norm.cdf(z)-1) + 2*stats.norm.pdf(z) - 1/np.sqrt(np.pi))
    return float(np.mean(crps_val))

def pi_coverage(obs, fc, sigma, level=0.95):
    """Empirical prediction interval coverage."""
    z = stats.norm.ppf((1+level)/2)
    lo = np.asarray(fc) - z*np.asarray(sigma)
    hi = np.asarray(fc) + z*np.asarray(sigma)
    return float(np.mean((np.asarray(obs) >= lo) & (np.asarray(obs) <= hi)))

def coverage_deviation(obs, fc, sigma, level=0.95):
    """Empirical coverage minus nominal level."""
    return pi_coverage(obs, fc, sigma, level) - level

def diebold_mariano(e1, e2, h=1):
    """
    Diebold-Mariano test for equal predictive accuracy (squared loss).
    Returns test statistic and two-sided p-value.
    H0: equal expected squared loss.
    """
    d = np.asarray(e1)**2 - np.asarray(e2)**2
    n = len(d)
    d_bar = np.mean(d)
    # HAC variance with Bartlett kernel
    gamma0 = np.var(d, ddof=1)
    gamma_h = np.array([np.cov(d[j:], d[:-j])[0,1] if j>0 else gamma0 for j in range(h)])
    v = (gamma0 + 2*np.sum(gamma_h[1:])) / n
    dm_stat = d_bar / np.sqrt(max(v, 1e-12))
    p_val = 2*(1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_val)

def summary_table(models_dict):
    """
    Print a summary table of forecast metrics.
    models_dict: {name: {"obs":[], "fc":[], "sigma":[]}}
    """
    print(f"{'Model':<18} {'RMSE':>7} {'MAE':>7} {'CRPS':>7} {'Cov%':>7} {'CovDev':>8}")
    print("-"*60)
    for name, d in models_dict.items():
        r = rmse(d["obs"], d["fc"])
        m = mae(d["obs"], d["fc"])
        c = crps_gaussian(d["obs"], d["fc"], d["sigma"])
        cov = pi_coverage(d["obs"], d["fc"], d["sigma"])
        dev = coverage_deviation(d["obs"], d["fc"], d["sigma"])
        print(f"{name:<18} {r:>7.4f} {m:>7.4f} {c:>7.4f} {cov*100:>6.1f}% {dev*100:>+7.1f}%")
