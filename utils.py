# utils.py

import numpy as np


def compute_entropy(sequence):
    """Order-0 Shannon entropy of the symbol-frequency distribution."""
    sequence = np.asarray(sequence)
    _, counts = np.unique(sequence, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))


def compute_transition_entropy(sequence):
    """
    Order-1 conditional entropy H(X_t | X_{t-1}).

    FIX: the previous version computed the entropy of the *joint*
    distribution over (X_{t-1}, X_t) pairs, which is a different
    (larger) quantity than the conditional entropy the paper defines
    as H1. True conditional entropy is the context-weighted average
    of the entropy of "what comes next" for each context symbol:

        H1 = sum_c P(c) * H(X_t | X_{t-1} = c)

    For a perfectly predictable sequence like "C D C D C D C D",
    this correctly returns ~0.0 (the paper's reported value),
    whereas the joint-entropy version returned ~0.985.
    """
    sequence = np.asarray(sequence)
    contexts = {}
    for a, b in zip(sequence[:-1], sequence[1:]):
        contexts.setdefault(a, []).append(b)

    total = len(sequence) - 1
    H1 = 0.0
    for _, nexts in contexts.items():
        p_ctx = len(nexts) / total
        _, counts = np.unique(nexts, return_counts=True)
        probs = counts / counts.sum()
        H_cond = -np.sum(probs * np.log2(probs))
        H1 += p_ctx * H_cond
    return H1


def compute_symmetry(sequence):
    # Symmetry score (palindrome measure)
    rev = sequence[::-1]
    hamming_distance = sum(a != b for a, b in zip(sequence, rev))
    return 1 - (hamming_distance / len(sequence))


def _lz76_factor_count(sequence):
    """Raw LZ76 factor count via incremental parsing."""
    i, l = 0, 1
    n = len(sequence)
    substrings = set()
    c = 0
    while True:
        if i + l > n:
            break
        substring = tuple(sequence[i:i + l])
        if substring in substrings:
            l += 1
        else:
            substrings.add(substring)
            i += l
            l = 1
            c += 1
    return c


def compute_lz_complexity(sequence):
    """
    Normalized Lempel-Ziv complexity following Kaspar & Schuster (1987):
    C(n) = c(n) * log2(n) / n, where c(n) is the raw LZ76 factor count.

    FIX: the previous version returned c(n) / n with no log2(n) term,
    which is not the Kaspar-Schuster normalization the paper cites and
    produced values on a much smaller (and differently-scaled) range
    than Table 2. Note: depending on the exact phrase-termination
    convention used, c(n) can differ by +/-1 from other LZ76
    implementations for edge cases (e.g. a fully-repeating tail) —
    if your LZ values still don't match Table 2 to the decimal, check
    that convention against your specific reference implementation.
    """
    n = len(sequence)
    c = _lz76_factor_count(sequence)
    return c * np.log2(n) / n


def compute_tenney_height(ratio):
    """
    Tenney height: log2(p * q) for a harmonic ratio p:q (Tenney, 1988) —
    a measure of harmonic complexity/dissonance for a simultaneous dyad.

    FIX: the previous version (a) used the wrong formula, log2(p/q)
    instead of log2(p*q), and (b) returned the same hardcoded
    log2(3/2) placeholder for every clip, so Fifth Ratio and Tritone
    Ratio (which should differ sharply in harmonic complexity) came
    out identical. This version takes the actual (p, q) ratio and
    reproduces the paper's Table 2 values exactly:
        compute_tenney_height((3, 2))   -> 2.585  (Fifth Ratio)
        compute_tenney_height((45, 32)) -> 10.492 (Tritone Ratio)
    """
    p, q = ratio
    return np.log2(p * q)
