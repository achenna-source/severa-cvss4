"""Frozen split definition (protocol v1, 2026-11-21). Produces data/splits.jsonl.
label_vector: base metrics of the CVE's own CNA v4.0 vector; inputs only what the CNA published itself."""
import json, re, collections, hashlib
BASE = ["AV", "AC", "AT", "PR", "UI", "VC", "VI", "VA", "SC", "SI", "SA"]
TRAIN_END, CALIB_END, TEST_END = "2026-04-01", "2026-07-01", "2026-11-21"
def base_vec(v):
    p = dict(x.split(":") for x in v.split("/")[1:])
    return "CVSS:4.0/" + "/".join(f"{m}:{p[m]}" for m in BASE)
def norm(t):
    t = t.lower(); t = re.sub(r"\d+(\.\d+)*", "0", t); return " ".join(t.split())
out, stats = [], collections.Counter()
for l in open("data/cves.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if not r["v40"] or not r["description"]:
        continue
    own = [e for e in r["v40"] if e["source"] == r["sourceIdentifier"]]
    if not own:
        stats["no_own_cna_v4"] += 1; continue
    d = r["published"][:10]
    split = "train" if d < TRAIN_END else "calib" if d < CALIB_END else "test" if d < TEST_END else None
    if split is None: continue
    v31 = [e["vector"] for e in r["v31"] if e["source"] == r["sourceIdentifier"]]
    cwe_in = sorted({c for s, c in r["cwe_src"] if s == r["sourceIdentifier"] and c.startswith("CWE-")})
    out.append({"id": r["id"], "published": r["published"], "split": split, "cna": r["sourceIdentifier"],
                "description": r["description"], "cwe_input": cwe_in, "v31_input": v31[0] if v31 else None,
                "label_vector": base_vec(own[0]["vector"]),
                "desc_hash": hashlib.sha1(norm(r["description"]).encode()).hexdigest()[:16]})
    stats[split] += 1
train_hash = {o["desc_hash"] for o in out if o["split"] == "train"}
train_cna = {o["cna"] for o in out if o["split"] == "train"}
for o in out:
    o["dup_in_train"] = o["split"] != "train" and o["desc_hash"] in train_hash
    o["cna_seen"] = o["cna"] in train_cna
with open("data/splits.jsonl", "w", encoding="utf-8") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(dict(stats))
for s in ("calib", "test"):
    S = [o for o in out if o["split"] == s]
    print(s, "n", len(S), "| v3.1 input", sum(o["v31_input"] is not None for o in S),
          "| CWE input", sum(bool(o["cwe_input"]) for o in S), "| dup desc in train", sum(o["dup_in_train"] for o in S),
          "| unseen CNA", sum(not o["cna_seen"] for o in S))
T = [o for o in out if o["split"] == "train"]
print("train v3.1 input", sum(o["v31_input"] is not None for o in T), "CWE input", sum(bool(o["cwe_input"]) for o in T))
