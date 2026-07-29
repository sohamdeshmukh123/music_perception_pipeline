# deep_validation.py
#
# Implements paper Sections 3.3.4-3.3.8, previously missing entirely
# despite being referenced in the paper's Data Availability section
# as "a second script... included as supplementary files":
#   3.3.4 Exact permutation test (all 9! relabelings)
#   3.3.5 Bootstrap confidence intervals (20,000 resamples)
#   3.3.6 Systematic LOOCV comparison across all four engineered features
#   3.3.7 Simulation-based power analysis
#   3.3.8 Bayesian linear model + Gaussian Process, compared under LOOCV
#
# Run after modeling.py (uses the same merged clip-level table).

import itertools
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

RNG = np.random.default_rng(42)

CLIP_ORDER = [
    'Pattern', 'Pattern Break', 'Symmetry', 'Symmetry Break',
    'Sequence', 'Sequence Break', 'Fifth Ratio', 'Tritone Ratio', 'Random',
]

clip_ratings = pd.read_csv('aggregated_clip_ratings.csv').set_index('Clip').reindex(CLIP_ORDER)
clip_features = pd.read_csv('clip_features.csv').set_index('Clip').reindex(CLIP_ORDER)
merged = clip_ratings.join(clip_features)

h1 = merged['H1'].values
y = merged['Pleasantness_mean'].values
n = len(y)

results_log = []


def log(line=""):
    print(line)
    results_log.append(line)


# ---------------------------------------------------------------
# 3.3.4 Exact permutation test (all 9! = 362,880 relabelings)
# ---------------------------------------------------------------
log("=== 3.3.4 Exact permutation test ===")
observed_r, _ = spearmanr(h1, y)

perm_rs = np.empty(math.factorial(n))
idx = np.arange(n)
for i, perm in enumerate(itertools.permutations(idx)):
    perm_rs[i], _ = spearmanr(h1, y[list(perm)])

exact_p = np.mean(np.abs(perm_rs) >= np.abs(observed_r))
log(f"Observed Spearman r(H1, pleasantness) = {observed_r:.3f}")
log(f"Exact two-sided permutation p-value (9! = {len(perm_rs)} relabelings): {exact_p:.5f}")
log(f"(paper reports p = .00066)")

np.save('permutation_null_distribution.npy', perm_rs)

# ---------------------------------------------------------------
# 3.3.5 Bootstrap confidence intervals (20,000 resamples)
# ---------------------------------------------------------------
log("\n=== 3.3.5 Bootstrap CIs (20,000 resamples) ===")
n_boot = 20000
boot_slopes = np.empty(n_boot)
boot_intercepts = np.empty(n_boot)
boot_r2 = np.empty(n_boot)

for b in range(n_boot):
    sample_idx = RNG.integers(0, n, size=n)
    Xb = h1[sample_idx].reshape(-1, 1)
    yb = y[sample_idx]
    if np.all(Xb == Xb[0]):  # degenerate resample, skip (near-zero prob)
        boot_slopes[b] = np.nan
        boot_intercepts[b] = np.nan
        boot_r2[b] = np.nan
        continue
    m = LinearRegression().fit(Xb, yb)
    boot_slopes[b] = m.coef_[0]
    boot_intercepts[b] = m.intercept_
    boot_r2[b] = m.score(Xb, yb)

valid = ~np.isnan(boot_slopes)
slope_ci = np.percentile(boot_slopes[valid], [2.5, 97.5])
intercept_ci = np.percentile(boot_intercepts[valid], [2.5, 97.5])
r2_ci = np.percentile(boot_r2[valid], [2.5, 97.5])
pct_positive = np.mean(boot_slopes[valid] > 0) * 100

log(f"Slope beta = {np.nanmean(boot_slopes):.3f}, 95% bootstrap CI [{slope_ci[0]:.2f}, {slope_ci[1]:.2f}]")
log(f"Intercept = {np.nanmean(boot_intercepts):.3f}, 95% bootstrap CI [{intercept_ci[0]:.2f}, {intercept_ci[1]:.2f}]")
log(f"In-sample R^2 95% bootstrap CI [{r2_ci[0]:.2f}, {r2_ci[1]:.2f}]")
log(f"Slope positive in {pct_positive:.1f}% of resamples")
log(f"(paper reports slope 95% CI [1.70, 3.75], intercept 95% CI [6.06, 6.93], R2 CI [0.53, 0.98], 100% positive)")

pd.DataFrame({'slope': boot_slopes, 'intercept': boot_intercepts, 'r2': boot_r2}).to_csv(
    'bootstrap_resamples.csv', index=False
)

# ---------------------------------------------------------------
# 3.3.6 Systematic LOOCV comparison across all four features
# ---------------------------------------------------------------
log("\n=== 3.3.6 Systematic per-feature LOOCV comparison (Table 5) ===")
feature_rows = []
for feat in ['H1', 'LZ', 'H0', 'Symmetry']:
    Xf = merged[[feat]].values
    m = LinearRegression().fit(Xf, y)
    insample_r2 = m.score(Xf, y)
    y_loo = cross_val_predict(LinearRegression(), Xf, y, cv=LeaveOneOut())
    loocv_r2 = r2_score(y, y_loo)
    loocv_mae = mean_absolute_error(y, y_loo)
    feature_rows.append({
        'Feature': feat, 'In-sample R2': round(insample_r2, 3),
        'LOOCV R2': round(loocv_r2, 3), 'LOOCV MAE': round(loocv_mae, 3),
    })

table5 = pd.DataFrame(feature_rows).sort_values('LOOCV R2', ascending=False)
log(table5.to_string(index=False))
log("(paper Table 5: H1 .789/.661/.273; LZ .190/-.299/.673; H0 .099/-.633/.735; Symmetry .001/-.813/.788)")
table5.to_csv('table5_feature_comparison.csv', index=False)

# ---------------------------------------------------------------
# 3.3.7 Simulation-based power analysis
# ---------------------------------------------------------------
log("\n=== 3.3.7 Simulation-based power analysis (Table 6) ===")
fit = LinearRegression().fit(h1.reshape(-1, 1), y)
resid_sd = np.std(y - fit.predict(h1.reshape(-1, 1)), ddof=2)
slope, intercept = fit.coef_[0], fit.intercept_

log(f"Data-generating process: slope={slope:.3f}, intercept={intercept:.3f}, residual SD={resid_sd:.3f}")

n_replicates = 500
sizes = [9, 15, 25, 50, 100, 200]
power_rows = []
h1_lo, h1_hi = h1.min(), h1.max()

for N in sizes:
    cv_r2_samples = np.empty(n_replicates)
    for rep in range(n_replicates):
        # Draw synthetic H1 values continuously across the observed range
        # (rather than resampling only the 9 discrete observed values,
        # which produces duplicate x's and degenerate, over-noisy LOOCV
        # folds at small N).
        x_sim = RNG.uniform(h1_lo, h1_hi, size=N)
        y_sim = intercept + slope * x_sim + RNG.normal(0, resid_sd, size=N)
        X_sim = x_sim.reshape(-1, 1)
        if N <= 30:
            y_pred = cross_val_predict(LinearRegression(), X_sim, y_sim, cv=LeaveOneOut())
        else:
            # K-fold for larger N -- equivalent out-of-sample estimate, tractable to compute
            y_pred = cross_val_predict(LinearRegression(), X_sim, y_sim, cv=10)
        cv_r2_samples[rep] = r2_score(y_sim, y_pred)
    power_rows.append({
        'N': N, 'Mean CV R2 (simulated)': round(np.mean(cv_r2_samples), 3),
        'SD of CV R2': round(np.std(cv_r2_samples), 3),
    })

table6 = pd.DataFrame(power_rows)
log(table6.to_string(index=False))
log("(paper Table 6: N=9 mean=.592 sd=.278; N=200 mean=.758 sd=.026)")
table6.to_csv('table6_power_simulation.csv', index=False)

# ---------------------------------------------------------------
# 3.3.8 Bayesian linear model + Gaussian Process
# ---------------------------------------------------------------
log("\n=== 3.3.8 Bayesian linear model + Gaussian Process (Table 7) ===")

# --- Bayesian linear regression (weak Gaussian prior on slope, N(0, 3^2);
#     noise variance plugged in from OLS residual variance) ---
X_design = np.column_stack([np.ones(n), h1])  # [intercept, slope]
sigma2 = resid_sd ** 2
prior_mean = np.zeros(2)
prior_cov = np.diag([100.0, 9.0])  # weak/diffuse prior: intercept sd=10, slope sd=3

prior_precision = np.linalg.inv(prior_cov)
post_precision = prior_precision + (X_design.T @ X_design) / sigma2
post_cov = np.linalg.inv(post_precision)
post_mean = post_cov @ (prior_precision @ prior_mean + (X_design.T @ y) / sigma2)

post_slope_mean = post_mean[1]
post_slope_sd = np.sqrt(post_cov[1, 1])
post_slope_ci = (post_slope_mean - 1.96 * post_slope_sd, post_slope_mean + 1.96 * post_slope_sd)

log(f"Bayesian posterior mean slope = {post_slope_mean:.3f}, "
    f"95% credible interval [{post_slope_ci[0]:.2f}, {post_slope_ci[1]:.2f}]")
log("(paper reports posterior slope 2.223, 95% CI [0.43, 4.02])")


def bayes_predict(x_train, y_train, x_test):
    Xd = np.column_stack([np.ones(len(x_train)), x_train])
    s2 = np.var(y_train - LinearRegression().fit(x_train.reshape(-1, 1), y_train)
                .predict(x_train.reshape(-1, 1)), ddof=2) or 1e-6
    prec = np.linalg.inv(prior_cov) + (Xd.T @ Xd) / s2
    cov = np.linalg.inv(prec)
    mean = cov @ ((np.linalg.inv(prior_cov) @ prior_mean) + (Xd.T @ y_train) / s2)
    return mean[0] + mean[1] * x_test


loo = LeaveOneOut()
bayes_preds = np.empty(n)
for i, (train_idx, test_idx) in enumerate(loo.split(h1)):
    bayes_preds[test_idx[0]] = bayes_predict(h1[train_idx], y[train_idx], h1[test_idx[0]])

bayes_loocv_r2 = r2_score(y, bayes_preds)
bayes_loocv_mae = mean_absolute_error(y, bayes_preds)

# --- Gaussian Process (RBF kernel) ---
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10)) + WhiteKernel(noise_level=0.1)
gp_preds = np.empty(n)
for train_idx, test_idx in loo.split(h1):
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3)
    gp.fit(h1[train_idx].reshape(-1, 1), y[train_idx])
    gp_preds[test_idx[0]] = gp.predict(h1[test_idx].reshape(-1, 1))[0]

gp_loocv_r2 = r2_score(y, gp_preds)
gp_loocv_mae = mean_absolute_error(y, gp_preds)

table7 = pd.DataFrame([
    {'Model': 'OLS (frequentist)', 'LOOCV R2': round(loocv := r2_score(
        y, cross_val_predict(LinearRegression(), h1.reshape(-1, 1), y, cv=LeaveOneOut())), 3),
     'LOOCV MAE': round(mean_absolute_error(
         y, cross_val_predict(LinearRegression(), h1.reshape(-1, 1), y, cv=LeaveOneOut())), 3)},
    {'Model': 'Bayesian linear (posterior mean)', 'LOOCV R2': round(bayes_loocv_r2, 3),
     'LOOCV MAE': round(bayes_loocv_mae, 3)},
    {'Model': 'Gaussian Process (RBF kernel)', 'LOOCV R2': round(gp_loocv_r2, 3),
     'LOOCV MAE': round(gp_loocv_mae, 3)},
])
log(table7.to_string(index=False))
log("(paper Table 7: OLS .661/.273; Bayesian .635/.325; GP .390/.416)")
table7.to_csv('table7_model_comparison.csv', index=False)

with open('deep_validation_results.txt', 'w') as f:
    f.write("\n".join(results_log))

print("\nSaved: permutation_null_distribution.npy, bootstrap_resamples.csv, "
      "table5_feature_comparison.csv, table6_power_simulation.csv, "
      "table7_model_comparison.csv, deep_validation_results.txt")
