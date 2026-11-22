import json, collections, math, re, sys
recs=[json.loads(l) for l in open("data/cves.jsonl",encoding="utf-8")]
v4=[r for r in recs if r["v40"]]
def lab(r):
    cna=[e for e in r["v40"] if e["source"]==r["sourceIdentifier"]]
    return (cna or r["v40"])[0]
mon=collections.Counter(r["published"][:7] for r in v4)
allmon=collections.Counter(r["published"][:7] for r in recs if r["published"]>="2025-07")
print("month  v4  all  share")
for m in sorted(allmon): print(m, mon[m], allmon[m], f"{mon[m]/allmon[m]:.2f}")
# label from CNA itself?
own=sum(1 for r in v4 if any(e["source"]==r["sourceIdentifier"] for e in r["v40"]))
print("v4 label from the CVE's own CNA:", own, "of", len(v4))
# per-source share by period
def per(r): return "train<=2025" if r["published"]<"2026" else ("2026H1" if r["published"]<"2026-07" else "2026H2")
src=collections.defaultdict(collections.Counter)
for r in v4: src[per(r)][lab(r)["source"]]+=1
for p in src: tot=sum(src[p].values()); print(p, tot, [(s.split('@')[-1],round(c/tot,2)) for s,c in src[p].most_common(6)])
# duplicate / templated descriptions
def norm(t): 
    t=t.lower(); t=re.sub(r"[0-9]+(\.[0-9]+)*","0",t); t=re.sub(r"\"[^\"]*\"|'[^']*'","q",t); return t
first12=collections.Counter(" ".join(norm(r["description"]).split()[:12]) for r in v4)
print("top description openings:", first12.most_common(5))
exact=collections.Counter(r["description"] for r in v4); print("exact duplicate descriptions:", sum(c for c in exact.values() if c>1))
# v3.1 -> v4 determinism per source (conditional entropy)
def H(c): n=sum(c.values()); return -sum(v/n*math.log2(v/n) for v in c.values())
for s in ["cna@vuldb.com","disclosure@vulncheck.com","security-advisories@github.com"]:
    grp=collections.defaultdict(collections.Counter)
    for r in v4:
        L=lab(r)
        if L["source"]!=s: continue
        v31=[e for e in r["v31"] if e["source"]==s]
        if v31: grp[v31[0]["vector"]][L["vector"]]+=1
    n=sum(sum(c.values()) for c in grp.values())
    ce=sum(sum(c.values())/n*H(c) for c in grp.values()) if n else float('nan')
    top=sum(max(c.values()) for c in grp.values())/n if n else float('nan')
    print(s, "pairs",n,"H(v4|v3.1)=",round(ce,3),"bits; acc of best deterministic map=",round(top,3))
# metric value distribution
M=["AV","AC","AT","PR","UI","VC","VI","VA","SC","SI","SA"]
dist={m:collections.Counter() for m in M}
for r in v4:
    p=dict(x.split(":") for x in lab(r)["vector"].split("/")[1:])
    for m in M: dist[m][p.get(m,"?")]+=1
for m in M: print(m, dict(dist[m]))
print("distinct label vectors:", len({lab(r)['vector'].split('/E:')[0] for r in v4}))
print("vectors with extra (threat/env/suppl) metrics:", sum(1 for r in v4 if len(lab(r)['vector'].split('/'))>12))
