"""Tables and result values for the manuscript, from results/*_FINAL.json and *_FINAL_arrays.npz.
Writes manuscript/fills_results.json ({{KEY}} -> text). Every printed number comes from these files."""
import json, os, sys
import numpy as np
from common import ROOT, ScoreTable, NAMES
from analyze import per_item, strat_boot, cluster_boot, mcnemar, holm

R = os.path.join(ROOT, "results")
t = ScoreTable()
F = {}


def load(name):
    return json.load(open(os.path.join(R, f"{name}_FINAL.json"))), np.load(os.path.join(R, f"{name}_FINAL_arrays.npz"), allow_pickle=True)


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


systems = [s for s in ("tfidf", "deberta_s1", "deberta_s2", "deberta_s3", "qwen7_s1", "qwen7_s2", "qwen7_s3") if os.path.exists(os.path.join(R, f"{s}_FINAL.json"))]
J, A = {}, {}
for s in systems:
    J[s], A[s] = load(s)
ids = A[systems[0]]["test_ids"]; Y = A[systems[0]]["test_y"]; grp = A[systems[0]]["test_group"]
cna = A[systems[0]]["test_cna"]; v31 = A[systems[0]]["test_has_v31"]


def row_metrics(s, dec):
    e = J[s][dec]["test"]
    return e


# ---- Table: main results (test)
COLS = [("exact_vector", "Exact vector (%)", True), ("macrovector", "MacroVector (%)", True),
        ("mean_metric_f1", "Mean metric F1", False), ("score_mae", "Score MAE", False),
        ("band_acc", "Band acc. (%)", True), ("band_kappa", "Band κ", False)]


def fmt(v, is_pct):
    return pct(v) if is_pct else f"{v:.3f}"


def agg(names, dec):
    vals = {c: [J[s][dec]["test"][c] for s in names] for c, _, _ in COLS}
    return vals


lines = ["| System | Decoding | " + " | ".join(h for _, h, _ in COLS) + " |", "|" + "---|" * (len(COLS) + 2)]
groups = [("B1 TF-IDF", ["tfidf"])]
deb = [s for s in systems if s.startswith("deberta")]
if deb:
    groups.append(("B2 DeBERTa-v3", deb))
qw = [s for s in systems if s.startswith("qwen7")]
if qw:
    groups.append(("B5 Qwen2.5-7B", qw))
for label, names in groups:
    for dec, dname in (("base_argmax", "per-metric argmax"), ("ours_score", "SEVERA (score)"), ("ours_band", "SEVERA (band)")):
        v = agg(names, dec)
        cells = []
        for c, _, isp in COLS:
            m = np.mean(v[c])
            if len(names) > 1:
                sd = np.std(v[c], ddof=1)
                cells.append(f"{fmt(m, isp)} ± {fmt(sd, isp) if not isp else pct(sd)}")
            else:
                cells.append(fmt(m, isp))
        sys_label = label if dec == "base_argmax" else ("SEVERA-enc" if label.startswith("B2") else ("SEVERA" if label.startswith("B5") else "SEVERA on TF-IDF"))
        lines.append(f"| {sys_label} | {dname} | " + " | ".join(cells) + " |")
F["TABLE_MAIN"] = "\n".join(lines[2:])
F["TABLE_MAIN_HEAD"] = "\n".join(lines[:2])

# ---- Table: ablation on the primary system
prim = "qwen7_s1" if "qwen7_s1" in systems else deb[0]
F["PRIMARY"] = {"qwen7_s1": "Qwen2.5-7B", "deberta_s1": "DeBERTa-v3"}.get(prim, prim)
abl = [("base_argmax", "Perception module, per-metric argmax"), ("base_score_mbr", "+ Bayes decision only (no knowledge)"),
       ("stack_temp_only_argmax", "+ recalibration only, argmax"), ("stack_no_cwe_argmax", "+ K2 (v3.1), argmax"),
       ("stack_no_v31_argmax", "+ K1 (CWE), argmax"), ("stack_full_argmax", "+ K1 + K2, argmax"),
       ("ours_score_nopairs", "+ K1 + K2, Bayes decision (score)"), ("ours_score", "SEVERA: + K3, Bayes decision (score)"),
       ("ours_band", "SEVERA: + K3, Bayes decision (band)")]
L = []
for k, lab in abl:
    e = J[prim][k]["test"]
    L.append(f"| {lab} | " + " | ".join(fmt(e[c], isp) for c, _, isp in COLS) + " |")
F["TABLE_ABLATION"] = "\n".join(L)

# ---- paired comparisons with clustered CIs (primary)
d = A[prim]
it_b = per_item(Y, d["test_base_argmax"], t); it_s = per_item(Y, d["test_ours_score"], t); it_bd = per_item(Y, d["test_ours_band"], t)
diff_mae = it_s["abs_err"] - it_b["abs_err"]; diff_band = it_s["band"] - it_b["band"]
F["GAIN_MAE"] = f"{-diff_mae.mean():.2f}"
ci = cluster_boot(diff_mae, cna); F["GAIN_MAE_CI"] = f"[{-ci[1]:.2f}, {-ci[0]:.2f}]"
F["GAIN_MAE_REL"] = f"{100 * (-diff_mae.mean()) / it_b['abs_err'].mean():.0f}%"
F["GAIN_BAND"] = f"{100 * diff_band.mean():.1f}"
ci = cluster_boot(diff_band, cna); F["GAIN_BAND_CI"] = f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"
p, a1, b1 = mcnemar(it_s["band"], it_b["band"]); F["MCNEMAR_BAND"] = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
F["BASE_MAE"] = f"{it_b['abs_err'].mean():.2f}"; F["SEVERA_MAE"] = f"{it_s['abs_err'].mean():.2f}"
F["BASE_BAND"] = pct(it_b["band"].mean()); F["SEVERA_BAND"] = pct(it_s["band"].mean())
F["SEVERA_BAND_BANDDEC"] = pct(it_bd["band"].mean())
F["SEVERA_EXACT"] = pct(it_s["exact"].mean()); F["SEVERA_EXACT_BANDDEC"] = pct(it_bd["exact"].mean()); F["BASE_EXACT"] = pct(it_b["exact"].mean())

# ---- per group and v3.1 availability (primary, base vs SEVERA-score)
G = []
for g in ("VulDB", "VulnCheck", "GitHub", "Other"):
    m = grp == g
    G.append(f"| {g} | {m.sum():,} | {pct(v31[m].mean(), 0)} | {pct(it_b['band'][m].mean())} | {pct(it_s['band'][m].mean())} | "
             f"{it_b['abs_err'][m].mean():.2f} | {it_s['abs_err'][m].mean():.2f} | {pct(it_b['exact'][m].mean())} | {pct(it_s['exact'][m].mean())} |")
for h, lab in ((True, "with CNA v3.1"), (False, "without v3.1 (cold)")):
    m = v31 == h
    G.append(f"| All, {lab} | {m.sum():,} | {pct(float(h), 0)} | {pct(it_b['band'][m].mean())} | {pct(it_s['band'][m].mean())} | "
             f"{it_b['abs_err'][m].mean():.2f} | {it_s['abs_err'][m].mean():.2f} | {pct(it_b['exact'][m].mean())} | {pct(it_s['exact'][m].mean())} |")
F["TABLE_GROUPS"] = "\n".join(G)
cold = v31 == False
F["COLD_N"] = f"{cold.sum():,}"; F["COLD_BAND_BASE"] = pct(it_b["band"][cold].mean()); F["COLD_BAND_SEVERA"] = pct(it_s["band"][cold].mean())

# ---- abstention (primary)
ab = J[prim]["abstention"]
AL = []
for a in ("0.05", "0.1", "0.15"):
    for tag, lab in (("global_bonf", "global"), ("mondrian_bonf", "per group"), ("mondrian", "per group, fixed sequence")):
        e = ab[a][f"test_{tag}"]
        cells = [f"{pct(e['coverage'])} / {pct(e['risk']) if e['risk'] is not None else '–'}"]
        for g in ("VulDB", "VulnCheck", "Other", "GitHub"):
            x = e[g]
            cells.append(f"{pct(x['coverage'], 0)} / {pct(x['risk']) if x['risk'] is not None else '–'}")
        AL.append(f"| {float(a):.2f} | {lab} | " + " | ".join(cells) + " |")
F["TABLE_ABSTENTION"] = "\n".join(AL)
e = ab["0.1"]["test_mondrian_bonf"]
F["ABST_COV_010"] = pct(e["coverage"]); F["ABST_RISK_010"] = pct(e["risk"])
eg = ab["0.1"]["test_global_bonf"]
F["ABST_GLOBAL_GITHUB_RISK"] = pct(eg["GitHub"]["risk"]); F["ABST_GLOBAL_OTHER_RISK"] = pct(eg["Other"]["risk"])
F["ABST_GLOBAL_RISK"] = pct(eg["risk"]); F["ABST_GLOBAL_COV"] = pct(eg["coverage"])
F["AURC"] = f"{J[prim]['aurc']['test']:.3f}"
ex = J[prim]["explain"]["test"]["abstained_a010"]
tot = sum(ex.values())
top = sorted(ex.items(), key=lambda kv: -kv[1])[:3]
F["EXPLAIN_TOP"] = ", ".join(f"{k} ({pct(v / tot, 0)}%)" for k, v in top)
json.dump(F, open(os.path.join(ROOT, "manuscript", "fills_results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for k, v in F.items():
    print(k, "=", v if "\n" not in str(v) else "\n" + v)

# ---- B0 / B3 / SEVERA on the v3.1 subset
bb = json.load(open(os.path.join(R, "baselines_b0_b3_FINAL.json")))
F["B0_EXACT"] = f"{pct(bb['B0_majority']['exact_vector'])}% (overall) and {pct(bb['B0_cwe_majority']['exact_vector'])}% (per CWE group)"
F["B3_N"] = f"{bb['B3_n']:,}"; F["B3_EXACT"] = pct(bb["B3_v31_lookup"]["exact_vector"]) + "%"
F["B3_BAND"] = pct(bb["B3_v31_lookup"]["band_acc"]) + "%"
m = v31 == True
F["SEVERA_V31_EXACT"] = pct(it_s["exact"][m].mean()) + "%"; F["SEVERA_V31_BAND"] = pct(it_s["band"][m].mean()) + "%"
F["SEVERA_V31_EXACT_BANDDEC"] = pct(it_bd["exact"][m].mean()) + "%"
# ---- B4 table (paired sample)
if os.path.exists(os.path.join(R, "b4_comparison.json")):
    b4 = json.load(open(os.path.join(R, "b4_comparison.json")))
    F["B4_N"] = f"{b4['n']:,}"; F["B4_INVALID"] = f"{pct(b4['invalid'] / b4['n'])}%"
    F["B4_EXACT"] = pct(b4["B4"]["exact_vector"]) + "%"; F["B4_BAND"] = pct(b4["B4"]["band_acc"]) + "%"
    key = f"{prim}/ours_score" if f"{prim}/ours_score" in b4 else "deberta_s1/ours_score"
    F["B4_CMP_EXACT"] = pct(b4[key]["exact_vector"]) + "%"; F["B4_CMP_BAND"] = pct(b4[key]["band_acc"]) + "%"
    rows_b4 = [("B4 retrieval-augmented prompting (Qwen2.5-7B, 8 shots)", b4["B4"])]
    for k, lab in (("tfidf/base_argmax", "B1 TF-IDF, per-metric argmax"), ("deberta_s1/base_argmax", "B2 DeBERTa-v3, per-metric argmax"),
                   ("qwen7_s1/base_argmax", "B5 Qwen2.5-7B fine-tuned, per-metric argmax"), (key, "SEVERA (score)")):
        if k in b4:
            rows_b4.append((lab, b4[k]))
    F["TABLE_B4"] = "| System | Exact vector (%) | Mean metric acc. (%) | Score MAE | Band acc. (%) |\n|---|---|---|---|---|\n" + "\n".join(
        f"| {lab} | {pct(e['exact_vector'])} | {pct(e['mean_metric_acc'])} | {e['score_mae']:.3f} | {pct(e['band_acc'])} |" for lab, e in rows_b4)
# ---- deployment
dep_name = prim if os.path.exists(os.path.join(R, f"deploy_{prim}.json")) else "deberta_s1"
if os.path.exists(os.path.join(R, f"deploy_{dep_name}.json")):
    dp = json.load(open(os.path.join(R, f"deploy_{dep_name}.json")))
    e = dp["0.1"]
    F["DEPLOY_N"] = f"{dp['n']:,}"; F["DEPLOY_COV"] = pct(e["coverage"]) + "%"; F["DEPLOY_ANSWERED"] = f"{e['answered']:,}"
    F["DEPLOY_COV_V31"] = pct(e["coverage_with_v31"]) + "%"; F["DEPLOY_COV_NOV31"] = pct(e["coverage_without_v31"]) + "%"
    F["DEPLOY_GITHUB_N"] = f"{e['GitHub']['n']:,}"
    hb = e["answered_bands"]; F["DEPLOY_HIGHCRIT"] = f"{hb['High'] + hb['Critical']:,}"
    F["DEPLOY_COV_015"] = pct(dp["0.15"]["coverage"]) + "%"; F["DEPLOY_COV_005"] = pct(dp["0.05"]["coverage"]) + "%"
    F["DEPLOY_SYSTEM"] = dep_name
json.dump(F, open(os.path.join(ROOT, "manuscript", "fills_results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("extra keys written")

# ---- per-backbone means (over seeds) for the narrative
for label, names in groups:
    tag = {"B1": "TFIDF", "B2": "DEB", "B5": "QWEN"}[label[:2]]
    for dec, dt in (("base_argmax", "BASE"), ("ours_score", "SEV"), ("ours_band", "SEVB")):
        F[f"{tag}_{dt}_MAE"] = f"{np.mean([J[s][dec]['test']['score_mae'] for s in names]):.2f}"
        F[f"{tag}_{dt}_BAND"] = pct(np.mean([J[s][dec]["test"]["band_acc"] for s in names]))
        F[f"{tag}_{dt}_EXACT"] = pct(np.mean([J[s][dec]["test"]["exact_vector"] for s in names]))
        F[f"{tag}_{dt}_KAPPA"] = f"{np.mean([J[s][dec]['test']['band_kappa'] for s in names]):.2f}"
    b = np.mean([J[s]["base_argmax"]["test"]["score_mae"] for s in names]); v = np.mean([J[s]["ours_score"]["test"]["score_mae"] for s in names])
    F[f"{tag}_MAE_REL"] = f"{100 * (b - v) / b:.0f}%"
    F[f"{tag}_BAND_GAIN"] = f"{round(100 * np.mean([J[s]['ours_score']['test']['band_acc'] for s in names]), 1) - round(100 * np.mean([J[s]['base_argmax']['test']['band_acc'] for s in names]), 1):.1f}"
    F[f"{tag}_NSEEDS"] = str(len(names))
e = J[prim]
F["MBR_ONLY_MAE"] = f"{e['base_score_mbr']['test']['score_mae']:.2f}"; F["MBR_ONLY_BAND"] = pct(e["base_score_mbr"]["test"]["band_acc"])
F["NOPAIRS_MAE"] = f"{e['ours_score_nopairs']['test']['score_mae']:.2f}"; F["NOPAIRS_BAND"] = pct(e["ours_score_nopairs"]["test"]["band_acc"])
F["K12_ARGMAX_EXACT"] = pct(e["stack_full_argmax"]["test"]["exact_vector"]); F["K12_ARGMAX_MV"] = pct(e["stack_full_argmax"]["test"]["macrovector"])
F["K2_ARGMAX_EXACT"] = pct(e["stack_no_cwe_argmax"]["test"]["exact_vector"]); F["K2_ARGMAX_MV"] = pct(e["stack_no_cwe_argmax"]["test"]["macrovector"]); F["K1_ARGMAX_EXACT"] = pct(e["stack_no_v31_argmax"]["test"]["exact_vector"])
F["BASE_MV"] = pct(e["base_argmax"]["test"]["macrovector"])
ab1 = J[prim]["abstention"]["0.1"]["test_mondrian_bonf"]
for g in ("VulDB", "VulnCheck", "Other", "GitHub"):
    F[f"ABST_{g.upper()}_COV"] = pct(ab1[g]["coverage"]); F[f"ABST_{g.upper()}_RISK"] = pct(ab1[g]["risk"]) if ab1[g]["risk"] is not None else "–"
ab5 = J[prim]["abstention"]["0.05"]["test_mondrian_bonf"]; ab15 = J[prim]["abstention"]["0.15"]["test_mondrian_bonf"]
F["ABST_COV_005"] = pct(ab5["coverage"]); F["ABST_RISK_005"] = pct(ab5["risk"]); F["ABST_COV_015"] = pct(ab15["coverage"]); F["ABST_RISK_015"] = pct(ab15["risk"])
fs = J[prim]["abstention"]["0.1"]["test_mondrian"]; F["ABST_FS_COV_010"] = pct(fs["coverage"])
for g in ("GitHub", "Other", "VulnCheck", "VulDB"):
    m = grp == g
    F[f"GRP_{g.upper()}_BASE_BAND"] = pct(it_b["band"][m].mean()); F[f"GRP_{g.upper()}_SEV_BAND"] = pct(it_s["band"][m].mean())
    F[f"GRP_{g.upper()}_SEV_MAE"] = f"{it_s['abs_err'][m].mean():.2f}"; F[f"GRP_{g.upper()}_BASE_MAE"] = f"{it_b['abs_err'][m].mean():.2f}"
F["COLD_MAE_BASE"] = f"{it_b['abs_err'][cold].mean():.2f}"; F["COLD_MAE_SEVERA"] = f"{it_s['abs_err'][cold].mean():.2f}"
json.dump(F, open(os.path.join(ROOT, "manuscript", "fills_results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("narrative keys written")

# ---- revision round 1 analyses (post hoc or additional; see revision_analyses.py)
rv_path = os.path.join(R, f"revision_{prim}.json")
if os.path.exists(rv_path):
    rv = json.load(open(rv_path))
    ex = rv["ablation_extra"]
    extra_rows = [("K2 + Bayes decision (score), no K1, no K3", ex["K2 + Bayes (score)"]),
                  ("Recalibration + Bayes decision (score)", ex["recalibration + Bayes (score)"]),
                  ("K1 + K2 + K3, joint MAP (most probable vector)", ex["K1+K2+K3, joint MAP"])]
    F["TABLE_ABLATION"] = F["TABLE_ABLATION"] + "\n" + "\n".join(
        f"| {lab} | " + " | ".join(fmt(e[c], isp) for c, _, isp in COLS) + " |" for lab, e in extra_rows)
    F["K2_BAYES_MAE"] = f"{ex['K2 + Bayes (score)']['score_mae']:.2f}"; F["K2_BAYES_BAND"] = pct(ex["K2 + Bayes (score)"]["band_acc"])
    F["RECAL_BAYES_MAE"] = f"{ex['recalibration + Bayes (score)']['score_mae']:.2f}"
    F["JOINTMAP_MAE"] = f"{ex['K1+K2+K3, joint MAP']['score_mae']:.2f}"; F["JOINTMAP_BAND"] = pct(ex["K1+K2+K3, joint MAP"]["band_acc"])
    F["JOINTMAP_EXACT"] = pct(ex["K1+K2+K3, joint MAP"]["exact_vector"])
    F["HYB_BASE_EXACT"] = pct(rv["hybrid_B3_else_base"]["exact_vector"]); F["HYB_BASE_BAND"] = pct(rv["hybrid_B3_else_base"]["band_acc"])
    F["HYB_BASE_MAE"] = f"{rv['hybrid_B3_else_base']['score_mae']:.2f}"
    F["HYB_SEV_EXACT"] = pct(rv["hybrid_B3_else_SEVERA"]["exact_vector"]); F["HYB_SEV_BAND"] = pct(rv["hybrid_B3_else_SEVERA"]["band_acc"])
    F["V31BAND_ACC"] = pct(rv["v31_band_as_v4_band"]["band_acc"]); F["V31BAND_N"] = f"{rv['v31_band_as_v4_band']['n']:,}"
    F["SEV_BAND_V31SUBSET"] = pct(rv["SEVERA_band_on_v31_subset"])
    F["BAND_ECE"] = f"{rv['band_ece']:.3f}"; F["BAND_NLL"] = f"{rv['band_nll']:.3f}"
    au = rv["confidence_auroc_by_group"]
    F["AUROC_GITHUB"] = f"{au['GitHub']:.2f}"; F["AUROC_VULNCHECK"] = f"{au['VulnCheck']:.2f}"; F["AUROC_OTHER"] = f"{au['Other']:.2f}"
    uo = rv["answered_under_over"]
    F["ANS_UNDER"] = pct(uo["under"] / uo["answered"]); F["ANS_OVER"] = pct(uo["over"] / uo["answered"])
    cr = rv["critical_recall"]
    F["CRIT_RECALL_VULNCHECK"] = pct(cr["VulnCheck"]["recall_answered"]) if cr["VulnCheck"]["recall_answered"] is not None else "–"
    cp = rv["clopper_pearson"]
    for al in ("0.05", "0.1", "0.15"):
        for g in ("All", "VulDB", "VulnCheck", "Other"):
            c = cp[al][g]
            if c["n"]:
                F[f"CP_{al.replace('.', '')}_{g.upper()}"] = f"{pct(c['risk'])}% [{pct(c['cp95'][0])}, {pct(c['cp95'][1])}]"
    mc = rv["monthly_cells"]
    F["MONTHLY_ABOVE_010"] = f"{mc['0.1']['above_alpha']} of {mc['0.1']['cells_n>=30']}"
    F["MONTHLY_ABOVE_005"] = f"{mc['0.05']['above_alpha']} of {mc['0.05']['cells_n>=30']}"
    F["MONTHLY_ABOVE_015"] = f"{mc['0.15']['above_alpha']} of {mc['0.15']['cells_n>=30']}"
    sb, sm = rv["signflip_band"], rv["signflip_mae"]
    F["SIGNFLIP_BAND_P"] = f"= {sb[1]:.3f}" if sb[1] >= 0.001 else "< 0.001"; F["SIGNFLIP_MAE_P"] = f"= {sm[1]:.3f}" if sm[1] >= 0.001 else "< 0.001"
    F["N_CNA_CLUSTERS"] = str(sb[2])
    la = rv["leakage_audit"]
    F["CUE_SHARE"] = pct(la["share_with_cue"]); F["CUE_SHARE_VULDB"] = pct(la["by_group"]["VulDB"])
    F["NOCUE_SEV_BAND"] = pct(la["SEVERA_band_without_cue"]); F["NOCUE_BASE_BAND"] = pct(la["base_band_without_cue"])
    F["NOCUE_SEV_MAE"] = f"{la['SEVERA_mae_without_cue']:.2f}"; F["NOCUE_BASE_MAE"] = f"{la['base_mae_without_cue']:.2f}"
    ev = rv["explain_validation"]
    F["EXPL_HIT"] = pct(ev["flagged_metric_wrong_given_any_wrong"]); F["EXPL_CHANCE"] = pct(ev["chance_rate"])
    mx = rv["mondrian_prov_x_v31"]["0.1"]
    F["PXV_COV_010"] = pct(mx["coverage"]); F["PXV_RISK_010"] = pct(mx["risk"])
    F["PXV_OTHER_V31_COV"] = pct(mx["Other|v31"]["coverage"]) if "Other|v31" in mx else "–"
    F["PXV_OTHER_V31_RISK"] = pct(mx["Other|v31"]["risk"]) if "Other|v31" in mx and mx["Other|v31"]["risk"] is not None else "–"
    F["PXV_OTHER_NOV31_COV"] = pct(mx["Other|nov31"]["coverage"]) if "Other|nov31" in mx else "–"
bk = os.path.join(R, "deploy_breakdown.json")
if os.path.exists(bk):
    b = json.load(open(bk))["breakdown"]
    F["DEP_OWN31"] = f"{b['own CNA v3.1']:,}"; F["DEP_OTHER31"] = f"{b['v3.1 from another source (NVD or ADP)']:,}"
    F["DEP_V30"] = f"{b['v3.0 only']:,}"; F["DEP_NONE"] = f"{b['no v3.x']:,}"
    F["DEP_GH_V31"] = f"{json.load(open(bk))['github']['own_v31']:,}"
dp_path = os.path.join(R, f"deploy_{dep_name}.json") if 'dep_name' in dir() else None
if dp_path and os.path.exists(dp_path):
    dp = json.load(open(dp_path))
    if "prov_x_v31" in dp:
        e = dp["prov_x_v31"]["0.1"]
        F["DEPLOY_PXV_COV"] = pct(e["coverage"]) + "%"
        F["DEPLOY_PXV_GH"] = pct(e["GitHub|v31"]["coverage"]) + "%" if "GitHub|v31" in e else "–"
json.dump(F, open(os.path.join(ROOT, "manuscript", "fills_results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("revision keys written")

# ---- DeBERTa gate on test (drives the deployment analysis) and worked example
if "deberta_s1" in J:
    eb = J["deberta_s1"]["abstention"]["0.1"]["test_mondrian_bonf"]
    F["DEB_ABST_COV_010"] = pct(eb["coverage"]); F["DEB_ABST_RISK_010"] = pct(eb["risk"])
if os.path.exists(rv_path):
    wx = json.load(open(rv_path)).get("worked_examples", [])
    if wx:
        w = wx[0]
        from common import idx_to_vec, band_of
        BN = ["None", "Low", "Medium", "High", "Critical"]
        av, sv = idx_to_vec(w["argmax_idx"]), idx_to_vec(w["severa_idx"])
        bp = w["band_probs"]
        sb = band_of(w["severa_score"]); ab_ = band_of(w["argmax_score"])
        F["RES_EXAMPLE_TEXT"] = (
            f"A test example shows the mechanism. For {w['id']} ({', '.join(w['cwe']) or 'no CWE'}"
            f"{', with a CNA version 3.1 vector' if w['has_v31'] else ''}), per-metric decoding returns "
            f"{av.replace('CVSS:4.0/', '')}, scored {w['argmax_score']:.1f} ({BN[ab_]}). "
            f"SEVERA's predictive distribution puts {100 * bp[sb]:.0f}% of its mass on the {BN[sb]} band, "
            f"and the score-risk decision returns {sv.replace('CVSS:4.0/', '')}, scored {w['severa_score']:.1f} "
            f"({BN[sb]}), which is the CNA's own vector. "
            f"In the test split, {json.load(open(rv_path))['n_worked_candidates']:,} VulnCheck vulnerabilities are moved in this way from a lower band to the correct one.")
json.dump(F, open(os.path.join(ROOT, "manuscript", "fills_results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("deberta gate and example keys written")
