"""Deployment population: CVEs published 2026-07-01..2026-11-20 with no CVSS v4.0 vector from any source."""
import json, collections
out = []; c = collections.Counter()
for l in open("data/cves.jsonl", encoding="utf-8"):
    r = json.loads(l)
    d = r["published"][:10]
    if not ("2026-07-01" <= d < "2026-11-21") or r["v40"] or not r["description"]:
        continue
    if r["status"] in ("Rejected",) or r["description"].startswith("** REJECT"):
        continue
    v31 = [e["vector"] for e in r["v31"] if e["source"] == r["sourceIdentifier"]]
    cwe_in = sorted({cw for s, cw in r["cwe_src"] if s == r["sourceIdentifier"] and cw.startswith("CWE-")})
    out.append({"id": r["id"], "published": r["published"], "split": "deploy", "cna": r["sourceIdentifier"],
                "description": r["description"], "cwe_input": cwe_in, "v31_input": v31[0] if v31 else None,
                "label_vector": None, "any_v31": bool(r["v31"])})
    c["n"] += 1; c["own_v31"] += bool(v31); c["any_v31"] += bool(r["v31"]); c["no_cvss_at_all"] += (not r["v31"] and not r["v30"])
with open("data/deploy.jsonl", "w", encoding="utf-8") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(dict(c))
print(collections.Counter(o["cna"] for o in out).most_common(8))
