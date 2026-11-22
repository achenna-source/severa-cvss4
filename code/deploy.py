"""Deployment analysis: apply SEVERA to the vulnerabilities published Jul-Nov 2026 without any CVSS v4.0 vector.
Labels are unknown, so only coverage and the distribution of answered bands are reported.
  python deploy.py --npz_dir ../kaggle_out/deberta/deberta_s1 --name deberta_s1"""
import argparse, collections, json, os
import numpy as np
from common import ROOT, NAMES, CARD, load_rows, ScoreTable, provenance, vec_to_idx
from knowledge import KnowledgeBase, Stacker, JointEngine
import abstention as AB
from run_pipeline import load_npz, DEVB_START


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True); ap.add_argument("--name", required=True)
    a = ap.parse_args()
    t = ScoreTable(); rows = load_rows()
    parts = {s: [r for r in rows if r["split2"] == s] for s in ("train", "dev", "calib", "test")}
    parts["dev_a"] = [r for r in parts["dev"] if r["published"][:10] < DEVB_START]
    dep = [json.loads(l) for l in open(os.path.join(ROOT, "data", "deploy.jsonl"), encoding="utf-8")]
    for r in dep:
        r["prov"] = provenance(r["cna"])
    raw = load_npz(a.npz_dir, parts)
    d = np.load(os.path.join(a.npz_dir, "probs_deploy.npz"))
    order = {cid: i for i, cid in enumerate(d["ids"])}
    ix = [order[r["id"]] for r in dep]
    raw["deploy"] = [np.log(np.clip(d[m][ix], 1e-9, 1.0)) for m in NAMES]
    ida = {r["id"] for r in parts["dev_a"]}
    mask_a = np.array([r["id"] in ida for r in parts["dev"]])
    raw["dev_a"] = [lp[mask_a] for lp in raw["dev"]]
    kb = KnowledgeBase(parts["train"])
    Ya = np.array([r["y"] for r in parts["dev_a"]])
    st = Stacker().fit(kb.features(parts["dev_a"], raw["dev_a"]), Ya)
    lam = json.load(open(os.path.join(ROOT, "results", f"{a.name}_FINAL.json")))["lambda"]
    eng = JointEngine(t, kb, lam)
    out = {}
    for s, rr in (("calib", parts["calib"]), ("deploy", dep)):
        lp = st.predict_logp(kb.features(rr, raw[s]))
        pred, bandp, _ = eng.decode(lp, "band")
        out[s] = (t.band[pred.astype(np.int64) @ t.radix], bandp.max(1), np.array([r["prov"] for r in rr]))
    Yc = np.array([r["y"] for r in parts["calib"]])
    err_c = (out["calib"][0] != t.band[Yc.astype(np.int64) @ t.radix]).astype(int)
    res = {"n": len(dep), "lambda": lam}
    bands = ["None", "Low", "Medium", "High", "Critical"]
    for alpha in (0.05, 0.10, 0.15):
        thr = AB.mondrian_thresholds(out["calib"][1], err_c, out["calib"][2], alpha, fn=AB.ltt_bonferroni)
        acc = AB.apply(out["deploy"][1], out["deploy"][2], thr)
        e = {"coverage": float(acc.mean()), "answered": int(acc.sum()),
             "answered_bands": {bands[k]: int(v) for k, v in enumerate(np.bincount(out["deploy"][0][acc], minlength=5))}}
        for g in np.unique(out["deploy"][2]):
            m = out["deploy"][2] == g
            e[g] = {"n": int(m.sum()), "coverage": float(acc[m].mean())}
        has31 = np.array([r["v31_input"] is not None for r in dep])
        e["coverage_with_v31"] = float(acc[has31].mean()); e["coverage_without_v31"] = float(acc[~has31].mean())
        top = collections.Counter(r["cna"] for r, k in zip(dep, acc) if k).most_common(8)
        e["top_answered_cnas"] = top
        res[str(alpha)] = e
    res["all_bands_predicted"] = {bands[k]: int(v) for k, v in enumerate(np.bincount(out["deploy"][0], minlength=5))}
    # post-hoc variant: groups = provenance x availability of a CNA v3.1 vector
    h_c = np.array([r["v31_input"] is not None for r in parts["calib"]]); h_d = np.array([r["v31_input"] is not None for r in dep])
    g2c = np.array([f"{g}|{'v31' if h else 'nov31'}" for g, h in zip(out["calib"][2], h_c)])
    g2d = np.array([f"{g}|{'v31' if h else 'nov31'}" for g, h in zip(out["deploy"][2], h_d)])
    res["prov_x_v31"] = {}
    for alpha in (0.05, 0.10, 0.15):
        thr = AB.mondrian_thresholds(out["calib"][1], err_c, g2c, alpha, fn=AB.ltt_bonferroni)
        acc = AB.apply(out["deploy"][1], g2d, thr)
        e = {"coverage": float(acc.mean()), "answered": int(acc.sum()),
             "calibrated_groups": sorted(thr), "deploy_groups_without_calibration": sorted(set(g2d) - set(thr))}
        for g in np.unique(g2d):
            m = g2d == g
            e[g] = {"n": int(m.sum()), "coverage": float(acc[m].mean())}
        res["prov_x_v31"][str(alpha)] = e
    json.dump(res, open(os.path.join(ROOT, "results", f"deploy_{a.name}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    main()
