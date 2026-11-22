"""Evaluate B4 (retrieval-augmented prompting) on its stratified test sample, and every other system on the
same sample, so that the comparison is paired. Invalid generations fall back to the per-metric majority
class of the training split (counted and reported)."""
import argparse, json, os
import numpy as np
from common import ROOT, load_rows, ScoreTable, evaluate, vec_to_idx, CARD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b4", required=True)
    ap.add_argument("--systems", nargs="+", required=True)  # name=path_to_FINAL_arrays.npz
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = ScoreTable()
    rows = load_rows()
    Ytr = np.array([r["y"] for r in rows if r["split2"] == "train"])
    maj = [int(np.bincount(Ytr[:, j], minlength=k).argmax()) for j, k in enumerate(CARD)]
    b4 = [json.loads(l) for l in open(a.b4, encoding="utf-8")]
    ids = [x["id"] for x in b4]
    lab = {r["id"]: r for r in rows}
    Y = np.array([lab[i]["y"] for i in ids])
    P = np.array([vec_to_idx(x["pred"]) if x["pred"] else maj for x in b4])
    res = {"n": len(ids), "invalid": int(sum(not x["valid"] for x in b4)), "B4": evaluate(Y, P, t)}
    for spec in a.systems:
        name, path = spec.split("=")
        d = np.load(os.path.join(ROOT, path), allow_pickle=True)
        pos = {cid: i for i, cid in enumerate(d["test_ids"])}
        ix = [pos[i] for i in ids]
        for dec in ("base_argmax", "ours_score", "ours_band"):
            res[f"{name}/{dec}"] = evaluate(Y, d[f"test_{dec}"][ix], t)
    json.dump(res, open(os.path.join(ROOT, a.out), "w"), indent=1)
    for k, e in res.items():
        if isinstance(e, dict):
            print(f"{k:28s} vec={e['exact_vector']:.3f} mae={e['score_mae']:.3f} band={e['band_acc']:.3f} macc={e['mean_metric_acc']:.3f}")
    print("invalid generations:", res["invalid"], "of", res["n"])


if __name__ == "__main__":
    main()
