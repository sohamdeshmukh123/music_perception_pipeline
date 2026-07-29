# compute_features.py

import pandas as pd
import numpy as np
from utils import compute_entropy, compute_transition_entropy, compute_symmetry, compute_lz_complexity, compute_tenney_height

# Clip sequences per Table 1 of the paper.
#
# FIX: 'Tritone Ratio' was previously an exact copy of 'Random' (same
# 9-note list) — a copy-paste bug. Per Table 1, Fifth Ratio and
# Tritone Ratio are both "C x 4"-style alternating dyads (simple vs.
# complex frequency ratio), so Tritone Ratio should mirror Fifth
# Ratio's alternating structure, not the melodic Random sequence.
# Both are written here as 7-note alternating patterns (4 of the
# tonic + 3 of the interval note), which reproduces the paper's
# Table 2 values exactly: H0 = 0.985, H1 = 0.000, Symmetry = 1.000
# for both clips.
clip_sequences = {
    'Pattern': ['C', 'D', 'C', 'D', 'C', 'D', 'C', 'D'],
    'Pattern Break': ['C', 'D', 'C', 'D', 'C', 'E', 'C', 'D'],
    'Symmetry': ['C', 'E', 'G', 'A', 'G', 'E', 'C'],
    'Symmetry Break': ['C', 'E', 'G', 'A', 'F', 'E', 'C'],
    'Sequence': ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'G', 'F', 'E'],
    'Sequence Break': ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'F', 'G', 'E'],
    'Fifth Ratio': ['C', 'G', 'C', 'G', 'C', 'G', 'C'],
    'Tritone Ratio': ['C', 'F#', 'C', 'F#', 'C', 'F#', 'C'],
    'Random': ['C', 'F#', 'D', 'A#', 'G', 'E', 'C#', 'F', 'A'],
}

# Harmonic ratios for the two simultaneous-dyad clips (Tenney height
# only applies to these; FIX: previously both clips got the same
# hardcoded log2(3/2) value regardless of which dyad they represent).
clip_harmonic_ratios = {
    'Fifth Ratio': (3, 2),     # simple ratio (perfect fifth)
    'Tritone Ratio': (45, 32),  # complex ratio (tritone)
}

# Map note names to integers
note_map = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}

# Function to convert sequence of notes to integers
def sequence_to_int(seq):
    return [note_map.get(note, 0) for note in seq]

# Prepare features for each clip
feature_list = []

for clip_name, notes in clip_sequences.items():
    seq_int = sequence_to_int(notes)
    entropy_H0 = compute_entropy(seq_int)
    entropy_H1 = compute_transition_entropy(seq_int)
    lz = compute_lz_complexity(seq_int)
    symmetry_score = compute_symmetry(seq_int)
    # Tenney height only applies to the two harmonic-ratio (dyad) clips
    tenney = (
        compute_tenney_height(clip_harmonic_ratios[clip_name])
        if clip_name in clip_harmonic_ratios
        else np.nan
    )

    feature_list.append({
        'Clip': clip_name,
        'H0': entropy_H0,
        'H1': entropy_H1,
        'LZ': lz,
        'Symmetry': symmetry_score,
        'Tenney': tenney
    })

# Save features to CSV
features_df = pd.DataFrame(feature_list)
features_df.to_csv('clip_features.csv', index=False)

print("Features computed and saved to 'clip_features.csv'")
print(features_df.round(3))
