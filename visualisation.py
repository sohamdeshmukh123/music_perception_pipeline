# visualisation.py
#
# Reproduces Figures 1-9 from the paper. Run after load_data.py,
# compute_features.py, eda_and_inference.py, modeling.py, and
# deep_validation.py have all been run (it reads their output CSVs).
#
# FIX vs. the original visualisation.py: that script only plotted
# hardcoded numbers copied from the paper's own Table 3 rather than
# reading anything from the pipeline's output files, and only
# produced one chart. This version generates all 9 figures from the
# actual pipeline outputs.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

CLIP_ORDER = [
    'Pattern', 'Pattern Break', 'Symmetry', 'Symmetry Break',
    'Sequence', 'Sequence Break', 'Fifth Ratio', 'Tritone Ratio', 'Random',
]

clip_ratings = pd.read_csv('aggregated_clip_ratings.csv').set_index('Clip').reindex(CLIP_ORDER)
clip_features = pd.read_csv('clip_features.csv').set_index('Clip').reindex(CLIP_ORDER)
merged = clip_ratings.join(clip_features)
long_df = pd.read_csv('participant_ratings_clean.csv')

h1 = merged['H1'].values
y = merged['Pleasantness_mean'].values
n = len(y)

plt.rcParams.update({'figure.dpi': 110})


# ---------------------------------------------------------------
# Figure 1: mean pleasantness by clip, ranked, with 95% CI
# ---------------------------------------------------------------
def ci95(sd, n_obs=31):
    return 1.96 * sd / np.sqrt(n_obs)

ranked = merged.sort_values('Pleasantness_mean', ascending=False)
errs = ci95(ranked['Pleasantness_std'].values)

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(ranked.index, ranked['Pleasantness_mean'], yerr=errs, capsize=4, color='#4C72B0')
ax.set_ylabel('Mean pleasantness (95% CI)')
ax.set_title('Listener pleasantness ratings by clip (N=31), ranked')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('figure1_pleasantness_ranked.png')
plt.close()

# ---------------------------------------------------------------
# Figure 2: feature-target correlation heatmap
# ---------------------------------------------------------------
corr_df = pd.read_csv('correlation_matrix.csv').set_index('Feature')
r_cols = [c for c in corr_df.columns if c.endswith('_r')]
p_cols = [c for c in corr_df.columns if c.endswith('_p')]
r_matrix = corr_df[r_cols].values
p_matrix = corr_df[p_cols].values
targets_disp = ['Pleasantness', 'Stability', 'Restlessness']

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(r_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(targets_disp)))
ax.set_xticklabels(targets_disp)
ax.set_yticks(range(len(corr_df.index)))
ax.set_yticklabels(corr_df.index)
for i in range(r_matrix.shape[0]):
    for j in range(r_matrix.shape[1]):
        stars = '**' if p_matrix[i, j] < .01 else ('*' if p_matrix[i, j] < .05 else '')
        ax.text(j, i, f"{r_matrix[i, j]:.2f}{stars}", ha='center', va='center', fontsize=10)
ax.set_title('Feature-rating correlation matrix (Spearman, n=9)\n* p<.05  ** p<.01')
fig.colorbar(im, ax=ax, label='Spearman r')
plt.tight_layout()
plt.savefig('figure2_correlation_matrix.png')
plt.close()

# ---------------------------------------------------------------
# Figure 3: H1 predicts pleasantness (in-sample fit + LOOCV predictions)
# ---------------------------------------------------------------
model = LinearRegression().fit(h1.reshape(-1, 1), y)
insample_r2 = model.score(h1.reshape(-1, 1), y)
y_loo = cross_val_predict(LinearRegression(), h1.reshape(-1, 1), y, cv=LeaveOneOut())
from sklearn.metrics import r2_score
loocv_r2 = r2_score(y, y_loo)

x_line = np.linspace(0, max(h1.max(), 0.8), 100)
y_line = model.predict(x_line.reshape(-1, 1))

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(x_line, y_line, color='red', linestyle='--', label='Fitted model (in-sample)')
ax.scatter(h1, y, color='#4C72B0', s=60, label='Observed mean rating', zorder=3)
ax.scatter(h1, y_loo, facecolors='none', edgecolors='green', marker='D', s=60,
           label='Leave-one-out prediction', zorder=3)
for clip in merged.index:
    ax.annotate(clip, (merged.loc[clip, 'H1'], merged.loc[clip, 'Pleasantness_mean']),
                fontsize=7, xytext=(4, 4), textcoords='offset points')
ax.set_xlabel('Order-1 transition entropy, H1 (bits)')
ax.set_ylabel('Mean pleasantness rating')
ax.set_title(f'Transition entropy predicts pleasantness\nin-sample R2 = {insample_r2:.2f}, '
             f'leave-one-out R2 = {loocv_r2:.2f}')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('figure3_h1_regression.png')
plt.close()

# ---------------------------------------------------------------
# Figure 4: exact permutation null distribution
# ---------------------------------------------------------------
perm_rs = np.load('permutation_null_distribution.npy')
observed_r = np.corrcoef(pd.Series(h1).rank(), pd.Series(y).rank())[0, 1]
exact_p = np.mean(np.abs(perm_rs) >= np.abs(observed_r))

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(perm_rs, bins=60, color='#888888')
ax.axvline(observed_r, color='red', label=f'Observed r = {observed_r:.2f}')
ax.set_xlabel(f'Spearman r under label permutation (all 9! = {len(perm_rs)} relabelings)')
ax.set_ylabel('Count')
ax.set_title(f'Exact permutation null distribution\np = {exact_p:.5f} (two-sided)')
ax.legend()
plt.tight_layout()
plt.savefig('figure4_permutation_test.png')
plt.close()

# ---------------------------------------------------------------
# Figure 5: bootstrap distribution of the H1 slope
# ---------------------------------------------------------------
boot = pd.read_csv('bootstrap_resamples.csv').dropna()
slope_ci = np.percentile(boot['slope'], [2.5, 97.5])

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(boot['slope'], bins=60, color='#4C72B0', alpha=0.8)
ax.axvline(boot['slope'].mean(), color='red', label=f"Point estimate = {boot['slope'].mean():.3f}")
ax.axvline(slope_ci[0], color='black', linestyle='--',
           label=f'95% bootstrap CI [{slope_ci[0]:.2f}, {slope_ci[1]:.2f}]')
ax.axvline(slope_ci[1], color='black', linestyle='--')
ax.set_xlabel('Bootstrap slope estimate (beta for H1)')
ax.set_ylabel('Frequency (of 20,000 resamples)')
ax.set_title('Bootstrap distribution of the H1 -> pleasantness slope')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('figure5_bootstrap_slope.png')
plt.close()

# ---------------------------------------------------------------
# Figure 6: per-feature LOOCV comparison bar chart
# ---------------------------------------------------------------
table5 = pd.read_csv('table5_feature_comparison.csv').set_index('Feature').reindex(
    ['H1', 'LZ', 'H0', 'Symmetry']
)
colors = ['#2ca02c' if v >= 0 else '#d62728' for v in table5['LOOCV R2']]

fig, ax = plt.subplots(figsize=(8, 5.5))
bars = ax.bar(table5.index, table5['LOOCV R2'], color=colors)
for bar, val in zip(bars, table5['LOOCV R2']):
    ax.text(bar.get_x() + bar.get_width() / 2, val + (0.02 if val >= 0 else -0.05),
            f'{val:.2f}', ha='center', fontsize=10)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Leave-one-out cross-validated R2')
ax.set_title('Only transition entropy (H1) generalizes out-of-sample\n(negative R2 = worse than predicting the mean)')
plt.tight_layout()
plt.savefig('figure6_feature_loocv_comparison.png')
plt.close()

# ---------------------------------------------------------------
# Figure 7: simulated effect of clip-library size on estimate stability
# ---------------------------------------------------------------
table6 = pd.read_csv('table6_power_simulation.csv')

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(table6['N'], table6['Mean CV R2 (simulated)'], marker='o', color='#4C72B0')
ax.fill_between(table6['N'],
                table6['Mean CV R2 (simulated)'] - table6['SD of CV R2'],
                table6['Mean CV R2 (simulated)'] + table6['SD of CV R2'],
                alpha=0.2, color='#4C72B0', label='+/-1 SD across simulation replicates')
ax.axvline(9, color='red', linestyle='--', label='This study (n=9)')
ax.set_xlabel('Number of stimulus clips in design matrix')
ax.set_ylabel('Cross-validated R2 (simulated)')
ax.set_title("This study's actual sample size sits in the highest-variance\nregion of the curve")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('figure7_power_simulation.png')
plt.close()

# ---------------------------------------------------------------
# Figure 8: OLS vs Bayesian vs GP fits
# ---------------------------------------------------------------
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10)) + WhiteKernel(noise_level=0.1)
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3)
gp.fit(h1.reshape(-1, 1), y)

x_grid = np.linspace(0, max(h1.max(), 0.8), 100).reshape(-1, 1)
gp_mean, gp_std = gp.predict(x_grid, return_std=True)
ols_line = model.predict(x_grid)

# Bayesian posterior mean line (same weak-prior setup as deep_validation.py)
X_design = np.column_stack([np.ones(n), h1])
sigma2 = np.var(y - model.predict(h1.reshape(-1, 1)), ddof=2)
prior_cov = np.diag([100.0, 9.0])
post_precision = np.linalg.inv(prior_cov) + (X_design.T @ X_design) / sigma2
post_cov = np.linalg.inv(post_precision)
post_mean = post_cov @ ((X_design.T @ y) / sigma2)
bayes_line = post_mean[0] + post_mean[1] * x_grid.flatten()

fig, ax = plt.subplots(figsize=(8, 6))
ax.fill_between(x_grid.flatten(), gp_mean - 1.96 * gp_std, gp_mean + 1.96 * gp_std,
                alpha=0.15, color='purple', label='GP 95% uncertainty band')
ax.plot(x_grid, gp_mean, color='purple', label='Gaussian Process mean')
ax.plot(x_grid, ols_line, '--', color='red', label='OLS fit')
ax.plot(x_grid, bayes_line, ':', color='green', label='Bayesian posterior mean')
ax.scatter(h1, y, color='black', zorder=5, label='Observed clips')
ax.set_xlabel('Order-1 transition entropy, H1 (bits)')
ax.set_ylabel('Pleasantness')
ax.set_title('Three modeling paradigms on the same n=9 dataset')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('figure8_model_paradigms.png')
plt.close()

# ---------------------------------------------------------------
# Figure 9: model comparison bar chart
# ---------------------------------------------------------------
table7 = pd.read_csv('table7_model_comparison.csv')

fig, ax = plt.subplots(figsize=(8, 5.5))
colors9 = ['#d62728', '#2ca02c', '#9467bd']
bars = ax.bar(table7['Model'], table7['LOOCV R2'], color=colors9)
for bar, val in zip(bars, table7['LOOCV R2']):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f'{val:.2f}', ha='center', fontsize=10)
ax.set_ylabel('LOOCV R2')
ax.set_title('Model comparison: added flexibility does not help at n=9')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('figure9_model_comparison.png')
plt.close()

print("Saved figure1_pleasantness_ranked.png through figure9_model_comparison.png")
