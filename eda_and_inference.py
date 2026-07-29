# eda_and_inference.py
#
# Implements paper Sections 3.1 (Exploratory Data Analysis) and
# 3.2 (Inferential Validation).
#
# Inputs:  participant_ratings_clean.csv, aggregated_clip_ratings.csv, clip_features.csv
# Outputs: correlation_matrix.csv, inferential_tests.txt

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, friedmanchisquare, wilcoxon

CLIP_ORDER = [
    'Pattern', 'Pattern Break', 'Symmetry', 'Symmetry Break',
    'Sequence', 'Sequence Break', 'Fifth Ratio', 'Tritone Ratio', 'Random',
]

# ---------------------------------------------------------------
# 3.1 Feature-target correlation matrix (Figure 2)
# ---------------------------------------------------------------
clip_ratings = pd.read_csv('aggregated_clip_ratings.csv').set_index('Clip').reindex(CLIP_ORDER)
clip_features = pd.read_csv('clip_features.csv').set_index('Clip').reindex(CLIP_ORDER)

targets = ['Pleasantness_mean', 'Stability_mean', 'Restlessness_mean']
features = ['H0', 'H1', 'LZ', 'Symmetry']

rows = []
for feat in features:
    row = {'Feature': feat}
    for tgt in targets:
        r, p = spearmanr(clip_features[feat], clip_ratings[tgt])
        row[f'{tgt}_r'] = round(r, 3)
        row[f'{tgt}_p'] = round(p, 4)
    rows.append(row)

corr_df = pd.DataFrame(rows)
corr_df.to_csv('correlation_matrix.csv', index=False)

print("=== Section 3.1: Feature-target Spearman correlations (n=9 clips) ===")
print(corr_df.to_string(index=False))

# ---------------------------------------------------------------
# 3.2 Inferential validation: Friedman + planned Wilcoxon comparisons
# ---------------------------------------------------------------
long_df = pd.read_csv('participant_ratings_clean.csv')

# Wide pivot: rows = participants, columns = clips, per target
def wide(target_col):
    return long_df.pivot(index='Timestamp', columns='Clip', values=target_col)[CLIP_ORDER]

friedman_results = {}
for target_col, label in [('Pleasantness', 'pleasantness'),
                           ('Stability', 'stability'),
                           ('Restlessness', 'restlessness')]:
    w = wide(target_col)
    stat, p = friedmanchisquare(*[w[c].values for c in CLIP_ORDER])
    friedman_results[label] = (stat, p)

# Planned pairwise comparisons (Holm-corrected), matching the paper's
# three named contrasts: Symmetry vs Random, Pattern vs Pattern Break,
# Fifth vs Tritone -- each tested on all three targets, Holm-corrected
# across the resulting family of tests.
planned_pairs = [
    ('Symmetry', 'Random'),
    ('Pattern', 'Pattern Break'),
    ('Fifth Ratio', 'Tritone Ratio'),
]

def holm_correct(pvals):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values
    in the original order."""
    pvals = np.asarray(pvals)
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running_max = max(running_max, val)
        adj[idx] = min(running_max, 1.0)
    return adj

def cohens_d_paired(x, y):
    diff = x - y
    return diff.mean() / diff.std(ddof=1)

pair_rows = []
raw_pvals = []
for a, b in planned_pairs:
    for target_col, label in [('Pleasantness', 'pleasantness'),
                               ('Stability', 'stability'),
                               ('Restlessness', 'restlessness')]:
        w = wide(target_col)
        stat, p = wilcoxon(w[a], w[b])
        d = cohens_d_paired(w[a].values, w[b].values)
        pair_rows.append({'Pair': f'{a} vs {b}', 'Target': label,
                           'W': stat, 'p_raw': p, 'd': round(d, 2)})
        raw_pvals.append(p)

adj_pvals = holm_correct(raw_pvals)
for row, p_holm in zip(pair_rows, adj_pvals):
    row['p_holm'] = round(p_holm, 4)

pair_df = pd.DataFrame(pair_rows)

with open('inferential_tests.txt', 'w') as f:
    f.write("=== Section 3.2: Friedman tests (clip identity effect) ===\n")
    for label, (stat, p) in friedman_results.items():
        line = f"{label}: chi2(8) = {stat:.2f}, p = {p:.4g}\n"
        f.write(line)
        print(line.strip())
    f.write("\n=== Planned Wilcoxon comparisons (Holm-corrected) ===\n")
    f.write(pair_df.to_string(index=False))
    f.write("\n")

print("\n=== Planned Wilcoxon comparisons (Holm-corrected) ===")
print(pair_df.to_string(index=False))
print("\nSaved correlation_matrix.csv and inferential_tests.txt")
