"""Knowledge layer and inference engine (the expert-system part of the method).

Knowledge base (estimated on the TRAIN split only, Laplace-smoothed):
  K1  CWE -> metric priors           P(v4 metric | CWE group)
  K2  v3.1 -> v4.0 correspondences   P(v4 metric | aligned v3.1 metric(s)), from the specification's metric
                                     alignment (AV->AV, AC->AC/AT, PR->PR, UI->UI, C/I/A x Scope -> VC/VI/VA and SC/SI/SA)
  K3  inter-metric compatibility     pointwise mutual information of metric pairs
Inference engine:
  per-metric stacking (multinomial logistic regression on dev) of model log-probabilities with K1 and K2,
  joint distribution over all 104,976 base vectors with K3 weighted by lambda (tuned on dev),
  minimum-Bayes-risk decoding under the official CVSS v4.0 scoring function.
"""
import collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from common import METRICS, NAMES, CARD

V31_ALIGN = {"AV": ["AV"], "AC": ["AC"], "AT": ["AC"], "PR": ["PR"], "UI": ["UI"],
             "VC": ["C", "S"], "VI": ["I", "S"], "VA": ["A", "S"], "SC": ["C", "S"], "SI": ["I", "S"], "SA": ["A", "S"]}
N_CWE = 40


def v31_parts(v):
    if not v:
        return None
    return dict(x.split(":") for x in v.split("/")[1:])


class KnowledgeBase:
    def __init__(self, train_rows, alpha=1.0):
        self.alpha = alpha
        cwe_counts = collections.Counter(c for r in train_rows for c in r["cwe_input"])
        self.cwe_groups = [c for c, _ in cwe_counts.most_common(N_CWE)]
        self.k1 = {m: collections.defaultdict(lambda k=k: np.full(k, alpha)) for m, k in zip(NAMES, CARD)}
        self.k2 = {m: collections.defaultdict(lambda k=k: np.full(k, alpha)) for m, k in zip(NAMES, CARD)}
        self.k3 = {}
        Y = np.array([r["y"] for r in train_rows])
        for r, y in zip(train_rows, Y):
            g = self.cwe_key(r)
            p31 = v31_parts(r["v31_input"])
            for j, m in enumerate(NAMES):
                self.k1[m][g][y[j]] += 1
                if p31:
                    self.k2[m][self.v31_key(m, p31)][y[j]] += 1
        # K3: smoothed PMI for every metric pair
        n = len(Y)
        marg = [np.bincount(Y[:, j], minlength=CARD[j]) + alpha for j in range(len(NAMES))]
        marg = [c / c.sum() for c in marg]
        for i in range(len(NAMES)):
            for j in range(i + 1, len(NAMES)):
                joint = np.full((CARD[i], CARD[j]), alpha)
                np.add.at(joint, (Y[:, i], Y[:, j]), 1)
                joint /= joint.sum()
                self.k3[(i, j)] = np.log(joint) - np.log(np.outer(marg[i], marg[j]))

    def cwe_key(self, r):
        cw = [c for c in r["cwe_input"] if c in self.cwe_groups]
        return cw[0] if cw else ("other" if r["cwe_input"] else "none")

    @staticmethod
    def v31_key(m, p31):
        return tuple(p31.get(a, "?") for a in V31_ALIGN[m])

    def log_table(self, m, table, key, k):
        c = table[m][key] if key in table[m] else np.full(k, self.alpha)
        return np.log(c / c.sum())

    def features(self, rows, logp):
        """logp: list of (N, k_m) model log-probabilities. Returns per-metric feature matrices."""
        feats = []
        for j, (m, k) in enumerate(zip(NAMES, CARD)):
            F = np.zeros((len(rows), 3 * k + 1))
            F[:, :k] = logp[j]
            for i, r in enumerate(rows):
                F[i, k:2 * k] = self.log_table(m, self.k1, self.cwe_key(r), k)
                p31 = v31_parts(r["v31_input"])
                if p31:
                    F[i, 2 * k:3 * k] = self.log_table(m, self.k2, self.v31_key(m, p31), k)
                else:
                    F[i, 3 * k] = 1.0
            feats.append(F)
        return feats


class Stacker:
    """Per-metric multinomial logistic regression on [model log-probs, K1, K2]; fitted on dev only."""

    def __init__(self, use_k1=True, use_k2=True, C=1.0):
        self.use_k1, self.use_k2, self.C = use_k1, use_k2, C
        self.models = []

    def _mask(self, F, k):
        F = F.copy()
        if not self.use_k1:
            F[:, k:2 * k] = 0
        if not self.use_k2:
            F[:, 2 * k:] = 0
        return F

    def fit(self, feats, Y):
        self.models = []
        for j, k in enumerate(CARD):
            lr = LogisticRegression(C=self.C, max_iter=2000)
            lr.fit(self._mask(feats[j], k), Y[:, j])
            self.models.append(lr)
        return self

    def predict_logp(self, feats):
        out = []
        for j, k in enumerate(CARD):
            lr = self.models[j]
            P = np.full((feats[j].shape[0], k), 1e-9)
            P[:, lr.classes_] = lr.predict_proba(self._mask(feats[j], k))
            P /= P.sum(1, keepdims=True)
            out.append(np.log(P))
        return out


class JointEngine:
    """Exact joint inference over the 104,976 CVSS v4.0 base vectors."""

    def __init__(self, table, kb=None, lam=0.0):
        self.t, self.lam = table, lam
        self.pair = np.zeros(table.n, dtype=np.float32)
        if kb is not None and lam:
            for (i, j), W in kb.k3.items():
                self.pair += W[table.idx[:, i], table.idx[:, j]].astype(np.float32)
        self.n_scores = len(table.scores_u)
        from scipy.sparse import csr_matrix
        n = table.n
        self.S = csr_matrix((np.ones(n, np.float32), (np.arange(n), table.score_id.astype(np.int64))),
                            shape=(n, self.n_scores))
        self.Bm = csr_matrix((np.ones(n, np.float32), (np.arange(n), table.band.astype(np.int64))), shape=(n, 5))
        self.perm_s = np.argsort(table.score_id, kind="stable")
        self.seg_s = np.searchsorted(table.score_id[self.perm_s], np.arange(self.n_scores + 1))
        self.perm_b = np.argsort(table.band, kind="stable")
        self.seg_b = np.searchsorted(table.band[self.perm_b], np.arange(6))

    @staticmethod
    def _seg_argmax(P, perm, seg):
        """For each segment (score or band value), the global index of the most probable vector in it."""
        Pp = P[:, perm]
        out = np.full((P.shape[0], len(seg) - 1), -1, dtype=np.int64)
        for s in range(len(seg) - 1):
            a, b = seg[s], seg[s + 1]
            if b > a:
                out[:, s] = perm[a + np.argmax(Pp[:, a:b], axis=1)]
        return out

    def posterior(self, logp_batch):
        """logp_batch: list of (B, k_m). Returns (B, 104976) normalised probabilities."""
        L = np.zeros((logp_batch[0].shape[0], self.t.n), dtype=np.float32)
        for j in range(len(NAMES)):
            L += logp_batch[j][:, self.t.idx[:, j]].astype(np.float32)
        if self.lam:
            L += self.lam * self.pair[None, :]
        L -= L.max(1, keepdims=True)
        P = np.exp(L)
        P /= P.sum(1, keepdims=True)
        return P

    def decode(self, logp, mode="score", batch=128):
        """Returns predicted index vectors (N,11), band distribution (N,5), score distribution (N,S), log-lik helper."""
        N = logp[0].shape[0]
        pred = np.zeros((N, len(NAMES)), dtype=np.int64)
        bandp = np.zeros((N, 5), dtype=np.float32)
        scorep = np.zeros((N, self.n_scores), dtype=np.float32)
        top = np.zeros(N, dtype=np.int64)
        for s in range(0, N, batch):
            P = self.posterior([lp[s:s + batch] for lp in logp])
            B = P.shape[0]
            sp = np.asarray(self.S.T.dot(P.T).T)
            bp = np.asarray(self.Bm.T.dot(P.T).T)
            scorep[s:s + B] = sp; bandp[s:s + B] = bp
            if mode == "argmax_joint":
                v = P.argmax(1)
            elif mode == "score":  # median of the predictive score distribution, then most probable vector there
                sid = (np.cumsum(sp, 1) >= 0.5).argmax(1)
                v = self._seg_argmax(P, self.perm_s, self.seg_s)[np.arange(B), sid]
            elif mode == "band":
                bd = bp.argmax(1)
                v = self._seg_argmax(P, self.perm_b, self.seg_b)[np.arange(B), bd]
            else:
                raise ValueError(mode)
            top[s:s + B] = v
        pred[:] = self.t.idx[top]
        return pred, bandp, scorep

    def explain(self, logp, batch=128):
        """Explanation facility: for each metric, total-variation distance between the band distribution and
        the band distribution conditional on that metric taking its most probable value. Returns (N, 11)."""
        from scipy.sparse import csr_matrix
        N = logp[0].shape[0]
        out = np.zeros((N, len(NAMES)), dtype=np.float32)
        masks = {}
        for j, k in enumerate(CARD):
            for a in range(k):
                sel = (self.t.idx[:, j] == a).astype(np.float32)
                masks[(j, a)] = self.Bm.multiply(sel[:, None]).tocsr()
        for s in range(0, N, batch):
            P = self.posterior([lp[s:s + batch] for lp in logp])
            bp = np.asarray(self.Bm.T.dot(P.T).T)
            for j, k in enumerate(CARD):
                cond = np.stack([np.asarray(masks[(j, a)].T.dot(P.T).T) for a in range(k)], 1)  # (B, k, 5)
                marg = cond.sum(2)  # (B, k) marginal of metric j
                mode = marg.argmax(1)
                c = cond[np.arange(P.shape[0]), mode]
                c = c / c.sum(1, keepdims=True)
                out[s:s + P.shape[0], j] = 0.5 * np.abs(c - bp).sum(1)
        return out

    def joint_loglik(self, logp, Y, batch=128):
        fy = Y.astype(np.int64) @ self.t.radix
        ll = 0.0
        for s in range(0, len(fy), batch):
            P = self.posterior([lp[s:s + batch] for lp in logp])
            ll += float(np.sum(np.log(P[np.arange(P.shape[0]), fy[s:s + batch]] + 1e-12)))
        return ll / len(fy)
