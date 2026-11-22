"""Final analyses from the per-vulnerability arrays written by run_pipeline.py.

  python analyze.py --split test --systems tfidf=results/tfidf_FINAL_arrays.npz deberta_s1=... --out results/analysis_test.json
Bootstrap: 2,000 resamples of CNAs (cluster bootstrap), percentile 95% intervals.
McNemar: exact two-sided binomial test on discordant pairs; Holm correction over the protocol comparisons."""
import argparse, json, os
import numpy as np
from scipy.stats import binomtest
from common import ROOT, ScoreTable, NAMES

DECODERS = ["base_argmax", "stack_argmax", "base_mbr", "nopairs", "ours_score", "ours_band"]


def per_item(Y, P, t):
    fy = Y.astype(np.int64) @ t.radix; fp = P.astype(np.int64) @ t.radix
    return {"exact": (fy == fp).astype(float), "mv": (t.mv[fy] == t.mv[fp]).astype(float),
            "abs_err": np.abs(t.score[fy] - t.score[fp]).astype(float), "band": (t.band[fy] == t.band[fp]).astype(float)}


def cluster_boot(vals, clusters, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=vals); cnt = np.bincount(inv)
    stats = []
    for _ in range(B):
        k = rng.integers(0, len(u), len(u))
        stats.append(sums[k].sum() / cnt[k].sum())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def strat_boot(vals, strata, B=2000, seed=0):
    """Item-level bootstrap stratified by provenance group (levels)."""
    rng = np.random.default_rng(seed)
    idx = [np.where(strata == g)[0] for g in np.unique(strata)]
    stats = []
    for _ in range(B):
        k = np.concatenate([rng.choice(i, len(i)) for i in idx])
        stats.append(vals[k].mean())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def mcnemar(a, b):
    n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum())
    if n01 + n10 == 0:
        return 1.0, n01, n10
    return float(binomtest(n01, n01 + n10, 0.5).pvalue), n01, n10


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    t = ScoreTable()
    S = {}
    for spec in a.systems:
        name, path = spec.split("=")
        S[name] = np.load(os.path.join(ROOT, path) if not os.path.isabs(path) else path, allow_pickle=True)
    ref = next(iter(S.values()))
    s = a.split
    ids, Y, grp, cna, v31 = ref[f"{s}_ids"], ref[f"{s}_y"], ref[f"{s}_group"], ref[f"{s}_cna"], ref[f"{s}_has_v31"]
    for n, d in S.items():
        assert np.array_equal(d[f"{s}_ids"], ids), n
    res = {"split": s, "n": int(len(ids)), "systems": {}}
    items = {}
    for n, d in S.items():
        res["systems"][n] = {}
        for dec in DECODERS:
            if f"{s}_{dec}" not in d.files:
                continue
            it = per_item(Y, d[f"{s}_{dec}"], t)
            items[(n, dec)] = it
            row = {}
            for k, v in it.items():
                row[k] = float(v.mean()); row[k + "_ci"] = strat_boot(v, grp, a.B)
            row["by_group"] = {g: {k: float(v[grp == g].mean()) for k, v in it.items()} | {"n": int((grp == g).sum())}
                               for g in np.unique(grp)}
            row["by_v31"] = {str(h): {k: float(v[v31 == h].mean()) for k, v in it.items()} | {"n": int((v31 == h).sum())}
                             for h in (True, False)}
            P = d[f"{s}_{dec}"]
            row["metric_acc"] = {m: float((Y[:, j] == P[:, j]).mean()) for j, m in enumerate(NAMES)}
            res["systems"][n][dec] = row
    # protocol comparisons on band correctness and exact vector (McNemar, Holm)
    comps = []
    names = list(S)
    for n in names:
        if (n, "ours_score") in items and (n, "base_argmax") in items:
            comps.append(((n, "ours_score"), (n, "base_argmax")))
            comps.append(((n, "ours_band"), (n, "base_argmax")))
    tests = []
    res["mae_diffs"] = []
    for (x, y) in comps:
        for k in ("band", "exact"):
            p, n01, n10 = mcnemar(items[x][k], items[y][k])
            diff = items[x][k] - items[y][k]
            tests.append({"a": "/".join(x), "b": "/".join(y), "measure": k, "p": p, "a_only": n01, "b_only": n10,
                          "diff": float(diff.mean()), "diff_ci_cluster": cluster_boot(diff, cna, a.B)})
        d_err = items[x]["abs_err"] - items[y]["abs_err"]
        tests_mae = {"a": "/".join(x), "b": "/".join(y), "measure": "abs_err_diff", "diff": float(d_err.mean()),
                     "diff_ci_cluster": cluster_boot(d_err, cna, a.B)}
        res.setdefault("mae_diffs", []).append(tests_mae)
    adj = holm(np.array([z["p"] for z in tests])) if tests else []
    for z, q in zip(tests, adj):
        z["p_holm"] = float(q)
    res["mcnemar"] = tests
    json.dump(res, open(os.path.join(ROOT, a.out), "w"), indent=1)
    for n in res["systems"]:
        for dec, r in res["systems"][n].items():
            print(f"{n:12s} {dec:13s} exact={r['exact']:.3f} [{r['exact_ci'][0]:.3f},{r['exact_ci'][1]:.3f}] "
                  f"mae={r['abs_err']:.3f} [{r['abs_err_ci'][0]:.3f},{r['abs_err_ci'][1]:.3f}] band={r['band']:.3f} "
                  f"[{r['band_ci'][0]:.3f},{r['band_ci'][1]:.3f}]")
    for z in tests:
        print(z["a"], "vs", z["b"], z["measure"], "diff=%.3f" % z["diff"], "cluster CI", np.round(z["diff_ci_cluster"], 3),
              "p_holm=%.2g" % z["p_holm"], z["a_only"], z["b_only"])
    for z in res.get("mae_diffs", []):
        print(z["a"], "vs", z["b"], "MAE diff=%.3f" % z["diff"], "cluster CI", np.round(z["diff_ci_cluster"], 3))


if __name__ == "__main__":
    main()
