# interactive_tool.py
#
# Section 4.1's deployed artifact: an interactive calculator that
# takes a user-entered note sequence, computes H0, H1, LZ complexity,
# and symmetry, and returns the fitted model's predicted pleasantness.
#
# FIX vs. the original interactive_tool.py:
#   - It reimplemented compute_transition_entropy with the same joint-
#     entropy bug as the old utils.py (see utils.py's docstring), so
#     a perfectly repeating sequence like "C D C D" would incorrectly
#     score high H1 and a high predicted pleasantness, when it should
#     score H1 = 0.
#   - It hardcoded a coefficient (2.5) and intercept (6.0) that don't
#     match the fitted model in the paper (slope 2.674/2.675,
#     intercept 6.488/6.49) or this pipeline's own regression output.
# This version imports the corrected entropy functions from utils.py
# and reads its coefficients from modeling.py's actual fitted model
# (falling back to the paper's published values if that file isn't
# available), so the "deployed" tool is consistent with the rest of
# the pipeline rather than a separate, drifted reimplementation.

import numpy as np
import pandas as pd
from utils import compute_entropy, compute_transition_entropy, compute_symmetry, compute_lz_complexity

note_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
            'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}


def load_fitted_model():
    """Refit pleasantness ~ H1 from the pipeline's own data files so the
    tool always matches whatever data it was trained on. Falls back to
    the paper's published coefficients if the pipeline outputs aren't
    present (e.g. running this file standalone)."""
    try:
        from sklearn.linear_model import LinearRegression
        clip_ratings = pd.read_csv('aggregated_clip_ratings.csv').set_index('Clip')
        clip_features = pd.read_csv('clip_features.csv').set_index('Clip')
        merged = clip_ratings.join(clip_features)
        X = merged[['H1']].values
        y = merged['Pleasantness_mean'].values
        model = LinearRegression().fit(X, y)
        return model.intercept_, model.coef_[0]
    except Exception:
        # Paper-published fallback coefficients (Section 3.3.1)
        return 6.49, 2.67


INTERCEPT, COEF_H1 = load_fitted_model()


def predict_pleasantness(sequence_notes):
    """Given a list of note names, compute H0/H1/LZ/symmetry and return
    the fitted model's predicted pleasantness (H1-based, per the paper's
    headline model)."""
    seq = [note_map.get(n, 0) for n in sequence_notes]
    H0 = compute_entropy(seq)
    H1 = compute_transition_entropy(seq)
    LZ = compute_lz_complexity(seq)
    symmetry = compute_symmetry(seq)
    predicted = INTERCEPT + COEF_H1 * H1
    return {'H0': H0, 'H1': H1, 'LZ': LZ, 'Symmetry': symmetry, 'Predicted_Pleasantness': predicted}


if __name__ == '__main__':
    print(f"Using fitted model: pleasantness = {INTERCEPT:.2f} + {COEF_H1:.2f} x H1")
    raw = input("Enter note sequence separated by spaces (e.g., C D E F G): ").strip().split()
    if not raw:
        print("No sequence entered.")
    else:
        result = predict_pleasantness(raw)
        print(f"Features: H0={result['H0']:.2f}, H1={result['H1']:.2f}, "
              f"LZ={result['LZ']:.2f}, Symmetry={result['Symmetry']:.2f}")
        print(f"Predicted Pleasantness: {result['Predicted_Pleasantness']:.2f} "
              f"(on the study's 11-point scale)")
