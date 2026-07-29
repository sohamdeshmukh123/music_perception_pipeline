"""
run_pipeline.py

Runs the full data science pipeline end to end, in the order the
paper describes it (Sections 2-4):

  1. load_data.py         -- reshape raw survey export -> clean long +
                             clip-level aggregated ratings (Table 3)
  2. compute_features.py  -- engineer H0, H1, LZ, symmetry, Tenney
                             height per clip (Table 2)
  3. eda_and_inference.py -- feature-target correlations (Fig 2),
                             Friedman + Wilcoxon tests (Section 3.2)
  4. modeling.py           -- baseline OLS, LOOCV, overfitting demo
                             (Section 3.3.1-3.3.3, Table 4)
  5. deep_validation.py    -- permutation test, bootstrap CIs,
                             per-feature LOOCV comparison, power
                             simulation, Bayesian + GP comparison
                             (Section 3.3.4-3.3.8, Tables 5-7)
  6. visualisation.py      -- Figures 1-9

Run this from the same directory as the raw CSV export, e.g.:
    python3 run_pipeline.py

interactive_tool.py is a standalone deployed artifact (Section 4.1)
and is not part of the batch pipeline -- run it separately:
    python3 interactive_tool.py
"""

import subprocess
import sys

STAGES = [
    'load_data.py',
    'compute_features.py',
    'eda_and_inference.py',
    'modeling.py',
    'deep_validation.py',
    'visualisation.py',
]

for stage in STAGES:
    print(f"\n{'=' * 70}\nRunning {stage}\n{'=' * 70}")
    result = subprocess.run([sys.executable, stage])
    if result.returncode != 0:
        print(f"\n{stage} failed (exit code {result.returncode}) -- stopping pipeline.")
        sys.exit(result.returncode)

print(f"\n{'=' * 70}\nPipeline complete. See README.md for a guide to the output files.\n{'=' * 70}")
