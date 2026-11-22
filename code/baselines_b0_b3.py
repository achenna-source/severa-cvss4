"""B0 (majority, overall and per CWE group) and B3 (version 3.1 lookup) on the test split."""
import json, os
import numpy as np
from common import ROOT, load_rows, ScoreTable, evaluate
from knowledge import KnowledgeBase
from run_pipeline import majority_probs, v31_rule_pred, argmax_pred

t = ScoreTable(); rows = load_rows()
parts = {s: [r for r in rows if r["split2"] == s] for s in ("train", "test")}
kb = KnowledgeBase(parts["train"])
Y = np.array([r["y"] for r in parts["test"]])
res = {"B0_majority": evaluate(Y, argmax_pred(majority_probs(parts)["test"]), t),
       "B0_cwe_majority": evaluate(Y, argmax_pred(majority_probs(parts, by_cwe=True, kb=kb)["test"]), t)}
has = np.array([r["v31_input"] is not None for r in parts["test"]])
b3 = v31_rule_pred(parts)["test"]
res["B3_v31_lookup"] = evaluate(Y[has], b3[has], t)
res["B3_n"] = int(has.sum())
json.dump(res, open(os.path.join(ROOT, "results", "baselines_b0_b3_FINAL.json"), "w"), indent=1)
for k, e in res.items():
    if isinstance(e, dict):
        print(f"{k:16s} n={e['n']} vec={e['exact_vector']:.3f} mae={e['score_mae']:.3f} band={e['band_acc']:.3f} k={e['band_kappa']:.3f} mf1={e['mean_metric_f1']:.3f}")
