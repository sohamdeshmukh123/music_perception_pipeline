# load_data.py
#
# FIX (replaces the old load_data.py / check.py / preprocess.py):
#
# The raw Google Form export uses inconsistent header text across
# clip blocks ("How Stable did THIS feel?" vs "...did IT feel?", same
# for Restlessness), so regex/text matching on column names silently
# drops columns and only reconstructs some clips. The paper itself
# notes (Sec 2.1.2) that one clip block's headers were actually
# mislabeled in the export, and that the fix was to trust column
# POSITION, not header text. This script does that: it assumes the
# 27 response columns are 9 repeating blocks of exactly 3
# (Pleasantness, Stability, Restlessness) in clip order, regardless
# of what each column happens to be named.
#
# It also fixes preprocess.py's bug of only ever keeping the FIRST
# clip's 3 columns (which discarded 8 of 9 clips), and check.py's/
# load_data.py's mistake of averaging each participant's ratings
# ACROSS clips (a per-participant response-bias score) rather than
# aggregating each clip's ratings ACROSS participants, which is what
# the paper's Table 3 and all clip-level modeling actually needs.

import pandas as pd

RAW_FILE = 'Music_Perception_Study__Responses__-_Form_Responses_1.csv'

# Clip order matches the paper's Table 1 / Table 2 presentation order,
# and matches the column order in the raw export.
CLIP_ORDER = [
    'Pattern', 'Pattern Break', 'Symmetry', 'Symmetry Break',
    'Sequence', 'Sequence Break', 'Fifth Ratio', 'Tritone Ratio', 'Random',
]

df = pd.read_csv(RAW_FILE)

# First 2 columns are Timestamp, Age Group; the next 27 are the 9
# clip blocks of 3 (Pleasantness, Stability, Restlessness), in order.
response_cols = df.columns[2:29].tolist()
assert len(response_cols) == 27, f"expected 27 response columns, got {len(response_cols)}"

records = []
for i, clip in enumerate(CLIP_ORDER):
    pleasant_col, stable_col, restless_col = response_cols[i * 3:(i + 1) * 3]
    for _, row in df.iterrows():
        records.append({
            'Timestamp': row['Timestamp'],
            'Age_Group': row['Age Group'],
            'Clip': clip,
            'Pleasantness': pd.to_numeric(row[pleasant_col], errors='coerce'),
            'Stability': pd.to_numeric(row[stable_col], errors='coerce'),
            'Restlessness': pd.to_numeric(row[restless_col], errors='coerce'),
        })

long_df = pd.DataFrame(records).dropna(
    subset=['Pleasantness', 'Stability', 'Restlessness']
)

# Participant-level clean long-format data (31 participants x 9 clips = 279 rows/target)
long_df.to_csv('participant_ratings_clean.csv', index=False)
print(f"Saved participant_ratings_clean.csv: {len(long_df)} rows "
      f"({long_df['Timestamp'].nunique()} participants x {long_df['Clip'].nunique()} clips)")

# Clip-level aggregated means/SDs -- this is what reproduces Table 3
summary = (
    long_df.groupby('Clip')[['Pleasantness', 'Stability', 'Restlessness']]
    .agg(['mean', 'std'])
    .round(2)
    .reindex(CLIP_ORDER)
)
summary.columns = ['_'.join(c) for c in summary.columns]
summary.to_csv('aggregated_clip_ratings.csv')
print("\nClip-level summary (compare to paper Table 3):")
print(summary)
