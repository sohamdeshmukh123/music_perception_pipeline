# modeling.py
#
# Implements paper Sections 3.3.1-3.3.3:
#   3.3.1 Baseline univariate OLS (pleasantness ~ H1) and the in-sample R^2 trap
#   3.3.2 Leave-one-out cross-validation of that model
#   3.3.3 A deliberate overfitting demonstration (adding a symmetry term)
#
# FIX vs. the original modeling.py / "modeling copy.py":
#   - The original modeling.py predicted pleasantness from raw survey
#     items (Stability, Restlessness) at the participant level -- not
#     the paper's actual model (H1 -> pleasantness at the clip level).
#   - The original "modeling copy.py" hand-typed the paper's own
#     published Table 3 numbers into a dict instead of reading them
#     from the data pipeline, so nothing was actually being derived
#     from clip_features.csv / aggregated_clip_ratings.csv.
# This version reads only from the pipeline's own output files and
# reproduces the paper's Table 4 exactly.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

CLIP_ORDER = [
    'Pattern', 'Pattern Break', 'Symmetry', 'Symmetry Break',
    'Sequence', 'Sequence Break', 'Fifth Ratio', 'Tritone Ratio', 'Random',
]

clip_ratings = pd.read_csv('aggregated_clip_ratings.csv').set_index('Clip').reindex(CLIP_ORDER)
clip_features = pd.read_csv('clip_features.csv').set_index('Clip').reindex(CLIP_ORDER)
merged = clip_ratings.join(clip_features)

y = merged['Pleasantness_mean'].values

# ---------------------------------------------------------------
# 3.3.1 Baseline model: pleasantness ~ H1
# ---------------------------------------------------------------
X_h1 = merged[['H1']].values
model_h1 = LinearRegression().fit(X_h1, y)
insample_r2_h1 = model_h1.score(X_h1, y)

print("=== 3.3.1 Baseline model ===")
print(f"pleasantness = {model_h1.intercept_:.2f} + {model_h1.coef_[0]:.2f} x H1")
print(f"In-sample R^2 = {insample_r2_h1:.3f}")

# ---------------------------------------------------------------
# 3.3.2 Leave-one-out cross-validation
# ---------------------------------------------------------------
y_loo_h1 = cross_val_predict(LinearRegression(), X_h1, y, cv=LeaveOneOut())
loocv_r2_h1 = r2_score(y, y_loo_h1)
loocv_mae_h1 = mean_absolute_error(y, y_loo_h1)

print("\n=== 3.3.2 LOOCV ===")
print(f"LOOCV R^2 = {loocv_r2_h1:.3f}  (vs in-sample R^2 = {insample_r2_h1:.3f})")
print(f"LOOCV MAE = {loocv_mae_h1:.3f}")

loo_predictions = pd.DataFrame({
    'Clip': merged.index,
    'Actual': y,
    'LOOCV_Predicted': y_loo_h1,
    'Residual': y - y_loo_h1,
}).round(3)
loo_predictions.to_csv('predictions.csv', index=False)

# ---------------------------------------------------------------
# 3.3.3 Deliberate overfitting demonstration: + symmetry
# ---------------------------------------------------------------
X_h1_sym = merged[['H1', 'Symmetry']].values
model_h1_sym = LinearRegression().fit(X_h1_sym, y)
insample_r2_h1_sym = model_h1_sym.score(X_h1_sym, y)
y_loo_h1_sym = cross_val_predict(LinearRegression(), X_h1_sym, y, cv=LeaveOneOut())
loocv_r2_h1_sym = r2_score(y, y_loo_h1_sym)
loocv_mae_h1_sym = mean_absolute_error(y, y_loo_h1_sym)

print("\n=== 3.3.3 Overfitting demonstration (Table 4) ===")
table4 = pd.DataFrame([
    {'Model': 'pleasantness ~ H1', 'In-sample R2': round(insample_r2_h1, 3),
     'LOOCV R2': round(loocv_r2_h1, 3), 'LOOCV MAE': round(loocv_mae_h1, 3)},
    {'Model': 'pleasantness ~ H1 + symmetry', 'In-sample R2': round(insample_r2_h1_sym, 3),
     'LOOCV R2': round(loocv_r2_h1_sym, 3), 'LOOCV MAE': round(loocv_mae_h1_sym, 3)},
])
print(table4.to_string(index=False))
table4.to_csv('table4_overfitting_demo.csv', index=False)

print(f"\nSymmetry coefficient in two-feature model: {model_h1_sym.coef_[1]:.3f} "
      f"(paper reports b = -0.05, ~indistinguishable from zero)")

# Also fit restlessness ~ H1 for completeness (paper reports R^2 = .700 in-sample)
y_rest = merged['Restlessness_mean'].values
model_rest = LinearRegression().fit(X_h1, y_rest)
print(f"\nrestlessness ~ H1 in-sample R^2 = {model_rest.score(X_h1, y_rest):.3f} "
      f"(paper reports 0.700)")
