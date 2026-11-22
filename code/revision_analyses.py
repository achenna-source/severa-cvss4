"""Analyses requested by the simulated review (round 1), all on the frozen test split of the primary system.
Writes results/revision_<name>.json. Nothing here changes a pre-registered choice; every item is reported as
an additional or post-hoc analysis.
  python revision_analyses.py --npz_dir ../kaggle_out/qwen7_s1/qwen7_s1 --name qwen7_s1"""
import argparse, collections, json, os, re
import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score
from common import ROOT, NAMES, CARD, load_rows, ScoreTable, evaluate, band_of, idx_to_vec
from knowledge import KnowledgeBase, Stacker, JointEngine
from run_pipeline import load_npz, argmax_pred, v31_rule_pred, DEVB_START
import abstention as AB


def cp_interval(k, n, a=0.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def signflip(diff, clusters, B=20000, seed=0):
    """CNA-level paired permutation test: flip the sign of each CNA's summed difference."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(clusters, return_inverse=True)
    s = np.bincount(inv, weights=diff); n = len(diff)
    obs = s.sum() / n
    flips = rng.choice([-1.0, 1.0], size=(B, len(u)))
    null = (flips * s).sum(1) / n
    return float(obs), float((np.abs(null) >= abs(obs)).mean()), int(len(u))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--npz_dir", required=True); ap.add_argument("--name", required=True)
    a = ap.parse_args()
    t = ScoreTable(); rows = load_rows()
    parts = {s: [r for r in rows if r["split2"] == s] for s in ("train", "dev", "calib", "test")}
    parts["dev_a"] = [r for r in parts["dev"] if r["published"][:10] < DEVB_START]
    raw = load_npz(a.npz_dir, parts)
    ida = {r["id"] for r in parts["dev_a"]}
    mask_a = np.array([r["id"] in ida for r in parts["dev"]])
    raw["dev_a"] = [lp[mask_a] for lp in raw["dev"]]
    kb = KnowledgeBase(parts["train"])
    Y = {s: np.array([r["y"] for r in parts[s]]) for s in parts}
    feats = {s: kb.features(parts[s], raw[s]) for s in ("dev_a", "calib", "test")}
    R = json.load(open(os.path.join(ROOT, "results", f"{a.name}_FINAL.json")))
    lam = R["lambda"]
    res = {}
    T = parts["test"]; Yt = Y["test"]; fy = Yt.astype(np.int64) @ t.radix
    grp = np.array([r["prov"] for r in T]); cna = np.array([r["cna"] for r in T]); has31 = np.array([r["v31_input"] is not None for r in T])
    eng = JointEngine(t, kb, lam); eng0 = JointEngine(t, None, 0.0)
    st = {"full": Stacker(), "k2": Stacker(use_k1=False), "temp": Stacker(use_k1=False, use_k2=False)}
    lp = {}
    for k, s in st.items():
        s.fit(feats["dev_a"], Y["dev_a"]); lp[k] = {sp: s.predict_logp(feats[sp]) for sp in ("calib", "test")}
    # ---- extra ablation rows
    abl = {}
    abl["K2 + Bayes (score)"] = eng0.decode(lp["k2"]["test"], "score")[0]
    abl["recalibration + Bayes (score)"] = eng0.decode(lp["temp"]["test"], "score")[0]
    abl["K1+K2+K3, joint MAP"] = eng.decode(lp["full"]["test"], "argmax_joint")[0]
    res["ablation_extra"] = {k: evaluate(Yt, v, t) for k, v in abl.items()}
    # ---- hybrid and v3.1-band baselines
    b3 = v31_rule_pred({"train": parts["train"], "test": T})["test"]
    base = argmax_pred(raw["test"])
    dsev = np.load(os.path.join(ROOT, "results", f"{a.name}_FINAL_arrays.npz"), allow_pickle=True)
    sev = dsev["test_ours_score"]; sevb = dsev["test_ours_band"]
    hyb = np.where(has31[:, None], b3, base); hyb_s = np.where(has31[:, None], b3, sev)
    res["hybrid_B3_else_base"] = evaluate(Yt, hyb, t); res["hybrid_B3_else_SEVERA"] = evaluate(Yt, hyb_s, t)
    from cvss import CVSS3
    by = t.band[fy]
    v31band = np.array([band_of(float(CVSS3(r["v31_input"]).base_score)) if r["v31_input"] else -1 for r in T])
    m = has31
    res["v31_band_as_v4_band"] = {"n": int(m.sum()), "band_acc": float((v31band[m] == by[m]).mean())}
    res["SEVERA_band_on_v31_subset"] = float((t.band[sev[m].astype(np.int64) @ t.radix] == by[m]).mean())
    # ---- band calibration and confidence ranking
    bandp = dsev["test_bandp"]; pb = t.band[sevb.astype(np.int64) @ t.radix]
    conf = bandp.max(1); err = (pb != by).astype(int)
    bins = np.linspace(0, 1, 11); ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mm = (conf > lo) & (conf <= hi)
        if mm.any():
            ece += mm.mean() * abs((1 - err[mm]).mean() - conf[mm].mean())
    res["band_ece"] = float(ece)
    res["band_nll"] = float(-np.mean(np.log(np.clip(bandp[np.arange(len(by)), by], 1e-12, 1))))
    res["confidence_auroc_by_group"] = {g: float(roc_auc_score(1 - err[grp == g], conf[grp == g])) for g in np.unique(grp)}
    # ---- under / over estimation and Critical recall among answered (alpha 0.10, per group)
    ab = R["abstention"]["0.1"]["thr_mondrian_bonf"]
    acc = AB.apply(conf, grp, ab)
    res["answered_under_over"] = {"answered": int(acc.sum()), "under": int(((pb < by) & acc).sum()), "over": int(((pb > by) & acc).sum())}
    res["critical_recall"] = {}
    for g in np.unique(grp):
        mm = (grp == g) & (by == 4)
        res["critical_recall"][g] = {"n_critical": int(mm.sum()), "recall_all": float((pb[mm] == 4).mean()) if mm.any() else None,
                                     "answered": int((mm & acc).sum()), "recall_answered": float((pb[mm & acc] == 4).mean()) if (mm & acc).any() else None}
    # ---- Clopper-Pearson for realised selective risk, per alpha and group
    cpres = {}
    for al in ("0.05", "0.1", "0.15"):
        thr = R["abstention"][al]["thr_mondrian_bonf"]; ac = AB.apply(conf, grp, thr)
        cpres[al] = {}
        for g in list(np.unique(grp)) + ["All"]:
            mm = ac & ((grp == g) if g != "All" else True)
            n, k = int(mm.sum()), int(err[mm].sum())
            cpres[al][g] = {"n": n, "k": k, "risk": k / n if n else None, "cp95": cp_interval(k, n) if n else None}
    res["clopper_pearson"] = cpres
    # ---- post-hoc variant: Mondrian over provenance x availability of a CNA v3.1 vector
    Tc = parts["calib"]
    grp_c = np.array([r["prov"] for r in Tc]); h31c = np.array([r["v31_input"] is not None for r in Tc])
    pc, bpc, _ = eng.decode(lp["full"]["calib"], "band")
    errc = (t.band[pc.astype(np.int64) @ t.radix] != t.band[Y["calib"].astype(np.int64) @ t.radix]).astype(int)
    g2c = np.array([f"{g}|{'v31' if h else 'nov31'}" for g, h in zip(grp_c, h31c)])
    g2t = np.array([f"{g}|{'v31' if h else 'nov31'}" for g, h in zip(grp, has31)])
    res["mondrian_prov_x_v31"] = {}
    for al in (0.05, 0.10, 0.15):
        thr = AB.mondrian_thresholds(bpc.max(1), errc, g2c, al, fn=AB.ltt_bonferroni)
        ac = AB.apply(conf, g2t, thr)
        e = {"coverage": float(ac.mean()), "risk": float(err[ac].mean()) if ac.any() else None,
             "thresholds": {k: float(v) for k, v in thr.items()}}
        for g in np.unique(g2t):
            mm = g2t == g
            e[g] = {"n": int(mm.sum()), "coverage": float(ac[mm].mean()), "risk": float(err[mm & ac].mean()) if (mm & ac).any() else None}
        res["mondrian_prov_x_v31"][str(al)] = e
    # ---- monthly x group realised risk with the fixed thresholds
    month = np.array([r["published"][:7] for r in T]); cells = []
    for al in ("0.05", "0.1", "0.15"):
        thr = R["abstention"][al]["thr_mondrian_bonf"]; ac = AB.apply(conf, grp, thr)
        above = tot = 0
        for mo in np.unique(month):
            for g in np.unique(grp):
                mm = ac & (month == mo) & (grp == g)
                if mm.sum() >= 30:
                    tot += 1; r_ = err[mm].mean(); above += r_ > float(al)
                    cells.append({"alpha": al, "month": mo, "group": g, "n": int(mm.sum()), "risk": float(r_)})
        res.setdefault("monthly_cells", {})[al] = {"cells_n>=30": tot, "above_alpha": int(above)}
    res["monthly_cells_detail"] = cells
    # ---- paired CNA-level sign-flip tests (SEVERA score decoder vs per-metric argmax)
    it_b = (t.band[base.astype(np.int64) @ t.radix] == by).astype(float); it_s = (t.band[sev.astype(np.int64) @ t.radix] == by).astype(float)
    e_b = np.abs(t.score[base.astype(np.int64) @ t.radix] - t.score[fy]); e_s = np.abs(t.score[sev.astype(np.int64) @ t.radix] - t.score[fy])
    res["signflip_band"] = signflip(it_s - it_b, cna); res["signflip_mae"] = signflip(e_s - e_b, cna)
    # ---- leakage audit: severity words and CVSS strings in descriptions
    pat = re.compile(r"(classified as|rated as|declared as) (critical|problematic|high|medium|low)|cvss|AV:[NALP]/|critical severity|high severity|base score", re.I)
    cue = np.array([bool(pat.search(r["description"])) for r in T])
    res["leakage_audit"] = {"share_with_cue": float(cue.mean()),
                            "by_group": {g: float(cue[grp == g].mean()) for g in np.unique(grp)},
                            "SEVERA_band_with_cue": float(it_s[cue].mean()) if cue.any() else None,
                            "SEVERA_band_without_cue": float(it_s[~cue].mean()),
                            "base_band_without_cue": float(it_b[~cue].mean()),
                            "SEVERA_mae_without_cue": float(e_s[~cue].mean()), "base_mae_without_cue": float(e_b[~cue].mean())}
    # ---- explanation facility validation: is the flagged metric actually wrong?
    X = eng.explain(lp["full"]["test"]); top = X.argmax(1)
    wrong_metric = (sevb != Yt)
    abst = ~acc
    flagged_wrong = wrong_metric[np.arange(len(top)), top]
    res["explain_validation"] = {
        "abstained": int(abst.sum()),
        "abstained_with_any_wrong_metric": int((abst & wrong_metric.any(1)).sum()),
        "flagged_metric_wrong_given_any_wrong": float(flagged_wrong[abst & wrong_metric.any(1)].mean()),
        "chance_rate": float((wrong_metric[abst & wrong_metric.any(1)].sum(1) / 11).mean()),
        "flagged_metric_wrong_given_band_wrong": float(flagged_wrong[abst & (err == 1)].mean()) if (abst & (err == 1)).any() else None}
    # ---- worked example: per-metric argmax in a lower band than the MBR decision, MBR band correct
    tb = t.band[base.astype(np.int64) @ t.radix]; ts = t.band[sev.astype(np.int64) @ t.radix]
    cand = np.where((tb < ts) & (ts == by) & (grp == "VulnCheck"))[0]
    ex = []
    for i in cand[:5]:
        ex.append({"id": T[i]["id"], "cwe": T[i]["cwe_input"], "has_v31": bool(has31[i]),
                   "label": T[i]["label_vector"], "label_score": float(t.score[fy[i]]),
                   "argmax_vector": idx_to_vec(base[i]), "severa_vector": idx_to_vec(sev[i]),
                   "argmax_idx": base[i].tolist(), "argmax_score": float(t.score[int(base[i].astype(np.int64) @ t.radix)]),
                   "severa_idx": sev[i].tolist(), "severa_score": float(t.score[int(sev[i].astype(np.int64) @ t.radix)]),
                   "band_probs": bandp[i].round(3).tolist(), "description": T[i]["description"][:400]})
    res["worked_examples"] = ex
    res["n_worked_candidates"] = int(len(cand))
    json.dump(res, open(os.path.join(ROOT, "results", f"revision_{a.name}.json"), "w"), indent=1, default=str)
    show = {k: v for k, v in res.items() if k not in ("monthly_cells_detail", "worked_examples")}
    print(json.dumps(show, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
