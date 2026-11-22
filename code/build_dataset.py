"""Flatten raw NVD pages into one JSONL record per CVE (data/cves.jsonl) and print an inventory.
Keeps: id, published, lastModified, vulnStatus, English description, CWE ids, every CVSS v4.0 / v3.1 / v3.0
metric entry (source, type, vector, score), reference count, source identifier (CNA)."""
import json, glob, os, sys, collections
RAW, OUT = sys.argv[1], sys.argv[2]
seen = {}
for fn in sorted(glob.glob(os.path.join(RAW, "*.json"))):
    for v in json.load(open(fn, encoding="utf-8")).get("vulnerabilities", []):
        c = v["cve"]
        if c.get("vulnStatus") == "Rejected":
            continue
        desc = next((d["value"] for d in c.get("descriptions", []) if d["lang"] == "en"), "")
        cwes = sorted({d["value"] for w in c.get("weaknesses", []) for d in w.get("description", []) if d["lang"] == "en"})
        cwe_src = sorted({(w.get("source"), d["value"]) for w in c.get("weaknesses", []) for d in w.get("description", []) if d["lang"] == "en"})
        m = c.get("metrics", {})
        def grab(key):
            out = []
            for e in m.get(key, []):
                cd = e.get("cvssData", {})
                out.append({"source": e.get("source"), "type": e.get("type"),
                            "vector": cd.get("vectorString"), "score": cd.get("baseScore")})
            return out
        seen[c["id"]] = {"id": c["id"], "published": c.get("published"), "lastModified": c.get("lastModified"),
                         "status": c.get("vulnStatus"), "sourceIdentifier": c.get("sourceIdentifier"),
                         "description": desc, "cwe": cwes, "cwe_src": cwe_src, "n_refs": len(c.get("references", [])),
                         "v40": grab("cvssMetricV40"), "v31": grab("cvssMetricV31"), "v30": grab("cvssMetricV30")}
with open(OUT, "w", encoding="utf-8") as f:
    for r in seen.values():
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
recs = list(seen.values())
by_year = collections.Counter(r["published"][:4] for r in recs)
v4 = [r for r in recs if r["v40"]]
both = [r for r in v4 if r["v31"]]
multi4 = [r for r in v4 if len({e["vector"] for e in r["v40"]}) > 1 or len(r["v40"]) > 1]
src = collections.Counter(e["source"] for r in v4 for e in r["v40"])
print("CVEs (non-rejected):", len(recs), dict(sorted(by_year.items())))
print("with v4.0:", len(v4), "| with v4.0 and v3.1:", len(both), "| v4.0 entries from >1 source:", len(multi4))
print("v4.0 by publication year:", dict(sorted(collections.Counter(r["published"][:4] for r in v4).items())))
print("top v4.0 sources:", src.most_common(15))
print("with CWE (non-NVD-noinfo):", sum(1 for r in v4 if any(c.startswith("CWE-") for c in r["cwe"])))
