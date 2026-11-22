"""Shared definitions: metrics, splits, score table, evaluation measures."""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = [("AV", "NALP"), ("AC", "LH"), ("AT", "NP"), ("PR", "NLH"), ("UI", "NPA"),
           ("VC", "HLN"), ("VI", "HLN"), ("VA", "HLN"), ("SC", "HLN"), ("SI", "HLN"), ("SA", "HLN")]
NAMES = [m for m, _ in METRICS]
CARD = [len(v) for _, v in METRICS]
BANDS = ["None", "Low", "Medium", "High", "Critical"]
DEV_START = "2026-03-01"


def split_of(r):
    if r["split"] == "train" and r["published"][:10] >= DEV_START:
        return "dev"
    return r["split"]


def provenance(cna):
    if cna == "cna@vuldb.com":
        return "VulDB"
    if cna == "disclosure@vulncheck.com":
        return "VulnCheck"
    if cna == "security-advisories@github.com":
        return "GitHub"
    return "Other"


def load_rows(path=None):
    path = path or os.path.join(ROOT, "data", "splits.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    for r in rows:
        r["split2"] = split_of(r)
        r["prov"] = provenance(r["cna"])
        r["y"] = vec_to_idx(r["label_vector"])
    return rows


def vec_to_idx(vec):
    p = dict(x.split(":") for x in vec.split("/")[1:])
    return [vals.index(p[m]) for m, vals in METRICS]


def idx_to_vec(ix):
    return "CVSS:4.0/" + "/".join(f"{m}:{vals[i]}" for (m, vals), i in zip(METRICS, ix))


def band_of(score):
    if score == 0:
        return 0
    if score < 4.0:
        return 1
    if score < 7.0:
        return 2
    if score < 9.0:
        return 3
    return 4


class ScoreTable:
    """All 104,976 base vectors in mixed-radix order of METRICS (first metric most significant)."""

    def __init__(self, path=None):
        path = path or os.path.join(ROOT, "data", "cvss4_score_table.json")
        d = json.load(open(path))
        rows = d["rows"]
        self.n = len(rows)
        self.score = np.array([r[1] for r in rows], dtype=np.float32)
        self.band = np.array([band_of(s) for s in self.score], dtype=np.int8)
        mv = sorted({r[3] for r in rows})
        self.mv_names = mv
        self.mv = np.array([mv.index(r[3]) for r in rows], dtype=np.int16)
        self.scores_u = np.unique(self.score)
        self.score_id = np.searchsorted(self.scores_u, self.score).astype(np.int16)
        idx = np.zeros((self.n, len(METRICS)), dtype=np.int8)
        k = np.arange(self.n)
        for j in reversed(range(len(METRICS))):
            idx[:, j] = k % CARD[j]
            k //= CARD[j]
        self.idx = idx
        self.radix = np.array([int(np.prod(CARD[j + 1:])) for j in range(len(METRICS))], dtype=np.int64)
        # sanity: row order must match itertools.product over value strings
        assert rows[0][0] == idx_to_vec(idx[0]) and rows[-1][0] == idx_to_vec(idx[-1])

    def flat(self, y):
        return int(np.dot(np.asarray(y, dtype=np.int64), self.radix))


def macro_f1(y, p, k):
    f = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        if tp + fp + fn == 0:
            continue
        f.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def weighted_kappa(a, b, k=5):
    O = np.zeros((k, k))
    for i, j in zip(a, b):
        O[i, j] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], dtype=float) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return float(1 - (W * O).sum() / (W * E).sum())


def evaluate(Y, P, table):
    """Y, P: (N, 11) int arrays of true / predicted metric indices."""
    Y = np.asarray(Y); P = np.asarray(P)
    fy = Y.astype(np.int64) @ table.radix; fp = P.astype(np.int64) @ table.radix
    sy, sp = table.score[fy], table.score[fp]
    by, bp = table.band[fy], table.band[fp]
    out = {"n": len(Y), "exact_vector": float(np.mean(fy == fp)),
           "macrovector": float(np.mean(table.mv[fy] == table.mv[fp])),
           "score_mae": float(np.mean(np.abs(sy - sp))), "band_acc": float(np.mean(by == bp)),
           "band_kappa": weighted_kappa(by, bp)}
    for j, m in enumerate(NAMES):
        out[f"acc_{m}"] = float(np.mean(Y[:, j] == P[:, j]))
        out[f"f1_{m}"] = macro_f1(Y[:, j], P[:, j], CARD[j])
    out["mean_metric_acc"] = float(np.mean([out[f"acc_{m}"] for m in NAMES]))
    out["mean_metric_f1"] = float(np.mean([out[f"f1_{m}"] for m in NAMES]))
    return out
