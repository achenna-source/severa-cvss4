"""End-to-end pipeline: base model probabilities -> knowledge layer -> MBR decoding -> abstention.

  python run_pipeline.py --system tfidf                    # B0/B1/B3 computed locally
  python run_pipeline.py --system npz --npz_dir runs/deberta_s1 --name B2_s1
Add --final to evaluate on the test split (to be run once per system for the final tables).
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, NAMES, CARD, load_rows, ScoreTable, evaluate, idx_to_vec
from knowledge import KnowledgeBase, Stacker, JointEngine
import abstention as AB

DEVB_START = "2026-03-16"


def render(r):
    return (r["description"] + " " + " ".join(r["cwe_input"]) + " " +
            (" ".join("v31_" + x.replace(":", "_") for x in r["v31_input"].split("/")[1:]) if r["v31_input"] else "nov31"))


def tfidf_probs(rows, parts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from scipy.sparse import hstack
    tr = parts["train"]
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=150000, sublinear_tf=True)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=150000, sublinear_tf=True)
    Xtr = hstack([w.fit_transform([render(r) for r in tr]), c.fit_transform([render(r) for r in tr])]).tocsr()
    Ytr = np.array([r["y"] for r in tr])
    out = {s: [] for s in parts if s != "train"}
    Xs = {s: hstack([w.transform([render(r) for r in parts[s]]), c.transform([render(r) for r in parts[s]])]).tocsr()
          for s in out}
    for j, k in enumerate(CARD):
        lr = LogisticRegression(C=8.0, max_iter=3000)
        lr.fit(Xtr, Ytr[:, j])
        for s in out:
            P = np.full((Xs[s].shape[0], k), 1e-6)
            P[:, lr.classes_] = lr.predict_proba(Xs[s])
            out[s].append(np.log(P / P.sum(1, keepdims=True)))
        print("tfidf metric", NAMES[j], "done", flush=True)
    return out


def majority_probs(parts, by_cwe=False, kb=None):
    tr = parts["train"]; Y = np.array([r["y"] for r in tr])
    out = {}
    for s in parts:
        if s == "train":
            continue
        lps = []
        for j, (m, k) in enumerate(zip(NAMES, CARD)):
            if by_cwe:
                L = np.array([kb.log_table(m, kb.k1, kb.cwe_key(r), k) for r in parts[s]])
            else:
                c = np.bincount(Y[:, j], minlength=k) + 1.0
                L = np.tile(np.log(c / c.sum()), (len(parts[s]), 1))
            lps.append(L)
        out[s] = lps
    return out


def v31_rule_pred(parts):
    """B3: most frequent v4.0 base vector for each observed v3.1 vector in train (only where v3.1 exists)."""
    import collections
    tab = collections.defaultdict(collections.Counter)
    for r in parts["train"]:
        if r["v31_input"]:
            tab[r["v31_input"]][tuple(r["y"])] += 1
    fallback = collections.Counter(tuple(r["y"]) for r in parts["train"]).most_common(1)[0][0]
    res = {}
    for s in parts:
        if s == "train":
            continue
        res[s] = np.array([tab[r["v31_input"]].most_common(1)[0][0] if r["v31_input"] in tab else fallback
                           for r in parts[s]])
    return res


def load_npz(npz_dir, parts):
    out = {}
    for s in ("dev", "calib", "test"):
        d = np.load(os.path.join(npz_dir, f"probs_{s}.npz"))
        order = {cid: i for i, cid in enumerate(d["ids"])}
        ix = [order[r["id"]] for r in parts[s]]
        out[s] = [np.log(np.clip(d[m][ix], 1e-9, 1.0)) for m in NAMES]
    return out


def argmax_pred(lps):
    return np.stack([lp.argmax(1) for lp in lps], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["tfidf", "npz", "majority"], required=True)
    ap.add_argument("--npz_dir"); ap.add_argument("--name", default=None)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    rows = load_rows(); table = ScoreTable()
    parts = {s: [r for r in rows if r["split2"] == s] for s in ("train", "dev", "calib", "test")}
    parts["dev_a"] = [r for r in parts["dev"] if r["published"][:10] < DEVB_START]
    parts["dev_b"] = [r for r in parts["dev"] if r["published"][:10] >= DEVB_START]
    kb = KnowledgeBase(parts["train"])
    name = a.name or a.system
    evals = ["calib", "test"] if a.final else ["dev_b", "calib"]
    mevals = ["test"] if a.final else ["dev_b", "calib"]  # splits on which accuracy measures are reported
    Y = {s: np.array([r["y"] for r in parts[s]]) for s in parts}
    res = {"system": name, "n": {s: len(parts[s]) for s in parts}, "final": a.final}

    if a.system == "tfidf":
        raw = tfidf_probs(rows, {s: parts[s] for s in ("train", "dev", "calib", "test")})
    elif a.system == "majority":
        raw = majority_probs({s: parts[s] for s in ("train", "dev", "calib", "test")})
        res["B0_cwe_majority"] = {s: evaluate(Y[s], argmax_pred(majority_probs(
            {"train": parts["train"], s: parts[s]}, by_cwe=True, kb=kb)[s]), table) for s in evals}
        b3 = v31_rule_pred({s: parts[s] for s in ["train"] + evals})
        res["B3_v31_rule"] = {}
        for s in evals:
            has = np.array([r["v31_input"] is not None for r in parts[s]])
            res["B3_v31_rule"][s] = evaluate(Y[s][has], b3[s][has], table)
    else:
        raw = load_npz(a.npz_dir, parts)
    # split dev log-probs into dev_a / dev_b (parts are ordered subsets of dev)
    ida = {r["id"] for r in parts["dev_a"]}
    mask_a = np.array([r["id"] in ida for r in parts["dev"]])
    raw["dev_a"] = [lp[mask_a] for lp in raw["dev"]]; raw["dev_b"] = [lp[~mask_a] for lp in raw["dev"]]

    # 1) base model, per-metric argmax
    res["base_argmax"] = {s: evaluate(Y[s], argmax_pred(raw[s]), table) for s in mevals}
    # 2) knowledge layer: stacking fitted on dev_a
    variants = {"full": Stacker(), "no_cwe": Stacker(use_k1=False), "no_v31": Stacker(use_k2=False),
                "temp_only": Stacker(use_k1=False, use_k2=False)}
    feats = {s: kb.features(parts[s], raw[s]) for s in ["dev_a", "dev_b"] + evals}
    stacked = {}
    for vn, st in variants.items():
        st.fit(feats["dev_a"], Y["dev_a"])
        stacked[vn] = {s: st.predict_logp(feats[s]) for s in set(evals) | {"dev_b"}}
        res[f"stack_{vn}_argmax"] = {s: evaluate(Y[s], argmax_pred(stacked[vn][s]), table) for s in mevals}
    # 3) pairwise strength lambda tuned on dev_b joint log-likelihood (full stacker)
    lls = {}
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        lls[lam] = JointEngine(table, kb, lam).joint_loglik(stacked["full"]["dev_b"], Y["dev_b"])
    lam = max(lls, key=lls.get)
    res["lambda_search"] = {str(k): v for k, v in lls.items()}; res["lambda"] = lam
    eng = JointEngine(table, kb, lam)
    dec = {}
    dec = {"score": {s: eng.decode(stacked["full"][s], "score") for s in mevals},
           "band": {s: eng.decode(stacked["full"][s], "band") for s in evals}}
    for mode in ("score", "band"):
        res[f"ours_{mode}"] = {s: evaluate(Y[s], dec[mode][s][0], table) for s in mevals}
    eng0 = JointEngine(table, None, 0.0)
    d0 = {s: eng0.decode(stacked["full"][s], "score") for s in mevals}
    res["ours_score_nopairs"] = {s: evaluate(Y[s], d0[s][0], table) for s in mevals}
    d_raw = {s: eng0.decode(raw[s], "score") for s in mevals}
    res["base_score_mbr"] = {s: evaluate(Y[s], d_raw[s][0], table) for s in mevals}
    # 4) abstention (band decoder): thresholds from calib, applied to later splits
    pred, bandp, _ = {}, {}, None
    tsplits = [s for s in evals if s != "calib"]
    res["abstention"] = {}
    fy = {s: Y[s].astype(np.int64) @ table.radix for s in evals}
    for s in evals:
        pv = dec["band"][s][0].astype(np.int64) @ table.radix
        pred[s] = (table.band[pv], dec["band"][s][1].max(1))
    groups = {s: np.array([r["prov"] for r in parts[s]]) for s in evals}
    err_c = (pred["calib"][0] != table.band[fy["calib"]]).astype(int)
    for alpha in (0.05, 0.10, 0.15):
        thrs = {"global": AB.ltt_threshold(pred["calib"][1], err_c, alpha),
                "mondrian": AB.mondrian_thresholds(pred["calib"][1], err_c, groups["calib"], alpha),
                "global_bonf": AB.ltt_bonferroni(pred["calib"][1], err_c, alpha),
                "mondrian_bonf": AB.mondrian_thresholds(pred["calib"][1], err_c, groups["calib"], alpha,
                                                        fn=AB.ltt_bonferroni)}
        entry = {f"thr_{k}": ({g: float(t) for g, t in v.items()} if isinstance(v, dict) else float(v))
                 for k, v in thrs.items()}
        for s in tsplits:
            err = (pred[s][0] != table.band[fy[s]]).astype(int)
            for tag, thr in thrs.items():
                acc = AB.apply(pred[s][1], groups[s], thr)
                e = {"coverage": float(acc.mean()), "risk": float(err[acc].mean()) if acc.any() else None}
                for g in np.unique(groups[s]):
                    gm = groups[s] == g
                    e[g] = {"n": int(gm.sum()), "coverage": float(acc[gm].mean()),
                            "risk": float(err[gm & acc].mean()) if (gm & acc).any() else None}
                entry[f"{s}_{tag}"] = e
        res["abstention"][str(alpha)] = entry
    for s in tsplits:
        err = (pred[s][0] != table.band[fy[s]]).astype(int)
        res.setdefault("aurc", {})[s] = AB.risk_coverage(pred[s][1], err)[2]
        # explanation facility: metric whose uncertainty moves most band probability
        X = eng.explain(stacked["full"][s])
        top = X.argmax(1)
        thr = res["abstention"]["0.1"]["thr_global_bonf"]
        abst = pred[s][1] < thr
        res.setdefault("explain", {})[s] = {
            "all": {NAMES[j]: int((top == j).sum()) for j in range(len(NAMES))},
            "abstained_a010": {NAMES[j]: int((top[abst] == j).sum()) for j in range(len(NAMES))},
            "abstained_wrong_band_a010": {NAMES[j]: int((top[abst & (err == 1)] == j).sum()) for j in range(len(NAMES))},
            "mean_tv": {NAMES[j]: float(X[:, j].mean()) for j in range(len(NAMES))}}
    res["minutes"] = (time.time() - t0) / 60
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    # per-vulnerability outputs for offline analyses (abstention variants, bootstrap, per-group tables)
    arr = {}
    for s in mevals:
        arr[f"{s}_ids"] = np.array([r["id"] for r in parts[s]]); arr[f"{s}_y"] = Y[s]
        arr[f"{s}_group"] = groups[s]; arr[f"{s}_has_v31"] = np.array([r["v31_input"] is not None for r in parts[s]])
        arr[f"{s}_cna"] = np.array([r["cna"] for r in parts[s]])
        arr[f"{s}_base_argmax"] = argmax_pred(raw[s]); arr[f"{s}_stack_argmax"] = argmax_pred(stacked["full"][s])
        arr[f"{s}_ours_score"] = dec["score"][s][0]; arr[f"{s}_ours_band"] = dec["band"][s][0]
        arr[f"{s}_bandp"] = dec["band"][s][1]; arr[f"{s}_base_mbr"] = d_raw[s][0]; arr[f"{s}_nopairs"] = d0[s][0]
    np.savez_compressed(os.path.join(ROOT, "results", f"{name}{'_FINAL' if a.final else '_dev'}_arrays.npz"), **arr)
    fn = os.path.join(ROOT, "results", f"{name}{'_FINAL' if a.final else '_dev'}.json")
    json.dump(res, open(fn, "w"), indent=1)
    # compact summary
    keys = [k for k in res if isinstance(res[k], dict) and isinstance(res[k].get(mevals[0]), dict) and "exact_vector" in res[k][mevals[0]]]
    for k in keys:
        for s in evals:
            if s in res[k]:
                e = res[k][s]
                print(f"{k:24s} {s:6s} vec={e['exact_vector']:.3f} mv={e['macrovector']:.3f} mae={e['score_mae']:.3f} "
                      f"band={e['band_acc']:.3f} k={e['band_kappa']:.3f} macc={e['mean_metric_acc']:.3f} mf1={e['mean_metric_f1']:.3f}")
    print("lambda", lam, "minutes", round(res["minutes"], 1), "->", fn)


if __name__ == "__main__":
    main()
