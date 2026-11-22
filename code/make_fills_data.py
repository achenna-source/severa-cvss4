"""Data-description values for the manuscript (fills_data.json), computed from data/splits.jsonl only."""
import collections, json, os, time
import numpy as np
from common import ROOT, load_rows, ScoreTable, CARD
from knowledge import JointEngine

rows = load_rows()
F = {}
order = [("train", "2023-01-01 to 2026-02-28"), ("dev", "2026-03-01 to 2026-03-31"),
         ("calib", "2026-04-01 to 2026-06-30"), ("test", "2026-07-01 to 2026-11-20")]
names = {"train": "Training", "dev": "Development", "calib": "Calibration", "test": "Test"}
lines = []
share = {}
for s, dates in order:
    S = [r for r in rows if r["split2"] == s]
    c = collections.Counter(r["prov"] for r in S)
    n = len(S)
    share[s] = {g: c[g] / n for g in ("VulDB", "VulnCheck", "GitHub", "Other")}
    cells = " | ".join(f"{c[g]:,} ({100 * c[g] / n:.0f}%)" for g in ("VulDB", "VulnCheck", "GitHub", "Other"))
    lines.append(f"| {names[s]} | {dates} | {n:,} | {cells} |")
F["TABLE_SPLITS"] = "\n".join(lines)
F["PROV_VULDB_TRAIN"] = f"{100 * share['train']['VulDB']:.0f}%"
F["PROV_VULDB_TEST"] = f"{100 * share['test']['VulDB']:.0f}%"
F["PROV_VULNCHECK_TRAIN"] = f"{100 * share['train']['VulnCheck']:.0f}%"
F["PROV_VULNCHECK_TEST"] = f"{100 * share['test']['VulnCheck']:.0f}%"
train_cna = {r["cna"] for r in rows if r["split2"] == "train"}
F["UNSEEN_CNA_TEST"] = f"{sum(1 for r in rows if r['split2'] == 'test' and r['cna'] not in train_cna):,}"
train_h = {r["desc_hash"] for r in rows if r["split2"] == "train"}
F["DUP_EVAL_TRAIN"] = str(sum(1 for r in rows if r["split2"] != "train" and r["desc_hash"] in train_h))
# best deterministic v3.1 -> v4.0 map, per CNA, over all labelled records (in-sample upper bound)
for cna, key in (("cna@vuldb.com", "V31_DET_VULDB"), ("disclosure@vulncheck.com", "V31_DET_VULNCHECK")):
    g = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["cna"] == cna and r["v31_input"]:
            g[r["v31_input"]][r["label_vector"]] += 1
    n = sum(sum(c.values()) for c in g.values())
    F[key] = f"{100 * sum(max(c.values()) for c in g.values()) / n:.0f}%"
# decoding time per vulnerability
t = ScoreTable(); eng = JointEngine(t, None, 0.0)
rng = np.random.default_rng(0)
lp = [np.log(rng.dirichlet(np.ones(k), size=512)) for k in CARD]
t0 = time.time(); eng.decode(lp, "score"); F["DECODE_MS"] = f"{1000 * (time.time() - t0) / 512:.0f}"
json.dump(F, open(os.path.join(ROOT, "manuscript", "fills_data.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for k, v in F.items():
    print(k, "=", v)
