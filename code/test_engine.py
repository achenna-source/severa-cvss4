import numpy as np, time
from common import ScoreTable, NAMES, CARD
from knowledge import JointEngine
t = ScoreTable(); eng = JointEngine(t, None, 0.0)
rng = np.random.default_rng(0)
lp = [np.log(rng.dirichlet(np.ones(k) * 0.5, size=64)) for k in CARD]
t0 = time.time(); pred, bandp, scorep = eng.decode(lp, "score"); print("decode 64 in", round(time.time() - t0, 2), "s")
P = eng.posterior(lp)
for b in range(3):  # brute-force check
    sp = np.bincount(t.score_id, weights=P[b], minlength=len(t.scores_u))
    sid = int(np.searchsorted(np.cumsum(sp), 0.5)); cand = np.where(t.score_id == sid)[0]
    v = cand[np.argmax(P[b, cand])]
    assert np.array_equal(t.idx[v], pred[b]), b
    assert np.allclose(sp, scorep[b], atol=1e-5)
pa, _, _ = eng.decode(lp, "argmax_joint")
ind = np.stack([l.argmax(1) for l in lp], 1)
print("independent joint argmax == per-metric argmax:", np.array_equal(pa, ind))
pb, bp, _ = eng.decode(lp, "band")
print("band of decoded vector == argmax band:", np.mean(t.band[pb.astype(np.int64) @ t.radix] == bp.argmax(1)))
print("ok")
