"""Risk-controlled abstention with Learn-then-Test (fixed-sequence testing, exact binomial p-values).

Guarantee (Angelopoulos et al., 2025, Learn then Test): with probability at least 1 - delta over the
calibration draw, the selective band-error risk of the chosen threshold is at most alpha, provided
calibration and deployment data are exchangeable (within each group for the Mondrian variant).
"""
import numpy as np
from scipy.stats import binom

# Protocol v1 amendment A1 (2026-11-21, before any test-set evaluation): fixed threshold grid.
# The first implementation used the observed confidences as the grid, so the sequence started on
# selections of one or two vulnerabilities whose p-value can never fall below delta.
GRID = np.round(np.concatenate([1 - np.logspace(-4, -1, 13), np.arange(0.89, 0.495, -0.01)]), 5)


def ltt_threshold(conf, err, alpha, delta=0.05, grid=None):
    """conf: confidence of accepted prediction, err: 1 if the band is wrong. Returns threshold (or inf)."""
    conf = np.asarray(conf); err = np.asarray(err).astype(int)
    if grid is None:
        grid = GRID  # fixed, data-independent order from strict to permissive
    chosen = np.inf
    for lam in grid:
        acc = conf >= lam
        n = int(acc.sum())
        if n == 0:  # empty selection has zero selective risk: the null is false, keep going
            continue
        k = int(err[acc].sum())
        p = binom.cdf(k, n, alpha)  # H0: risk > alpha; small p rejects H0
        if p <= delta:
            chosen = lam
        else:
            break
    return chosen


def ltt_bonferroni(conf, err, alpha, delta=0.05, grid=None):
    """Learn-then-Test with Bonferroni over the fixed grid: every threshold is tested at level delta/|grid|,
    the most permissive rejected threshold is kept. Valid without any ordering, so an under-powered strict
    threshold cannot stop the search."""
    conf = np.asarray(conf); err = np.asarray(err).astype(int)
    grid = GRID if grid is None else grid
    level = delta / len(grid)
    chosen = np.inf
    for lam in grid:
        acc = conf >= lam
        n = int(acc.sum())
        if n == 0:
            continue
        if binom.cdf(int(err[acc].sum()), n, alpha) <= level:
            chosen = min(chosen, lam)
    return chosen


def mondrian_thresholds(conf, err, groups, alpha, delta=0.05, fn=None):
    fn = fn or ltt_threshold
    return {g: fn(conf[groups == g], err[groups == g], alpha, delta) for g in np.unique(groups)}


def apply(conf, groups, thr):
    if isinstance(thr, dict):
        return np.array([c >= thr.get(g, np.inf) for c, g in zip(conf, groups)])
    return conf >= thr


def risk_coverage(conf, err):
    order = np.argsort(-conf)
    e = np.asarray(err)[order]
    cov = np.arange(1, len(e) + 1) / len(e)
    risk = np.cumsum(e) / np.arange(1, len(e) + 1)
    return cov, risk, float(np.mean(risk))  # AURC
