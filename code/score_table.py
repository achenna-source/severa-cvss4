"""Enumerate every CVSS v4.0 base vector (104,976) and store score, severity, MacroVector with RedHat cvss 3.6."""
import itertools, json, sys, time
import math
import cvss.cvss4 as _cvss4
from cvss import CVSS4

# Amendment A4: FIRST's reference calculator rounds Math.round(x * 10) / 10 on the raw double; the Red Hat
# library (3.6) adds an epsilon first and disagrees on 46 of the 10,000 base-only vectors of FIRST's reference
# set. With this rounding the library agrees with all 41,270 scored vectors of the reference set.
_cvss4.final_rounding = lambda x: math.floor(x * 10 + 0.5) / 10
METRICS = [("AV", "NALP"), ("AC", "LH"), ("AT", "NP"), ("PR", "NLH"), ("UI", "NPA"),
           ("VC", "HLN"), ("VI", "HLN"), ("VA", "HLN"), ("SC", "HLN"), ("SI", "HLN"), ("SA", "HLN")]
t = time.time(); rows = []
for vals in itertools.product(*[list(v) for _, v in METRICS]):
    vec = "CVSS:4.0/" + "/".join(f"{m}:{x}" for (m, _), x in zip(METRICS, vals))
    c = CVSS4(vec)
    rows.append((vec, float(c.base_score), c.severity, c.macroVector()))
json.dump({"metrics": METRICS, "rows": rows}, open(sys.argv[1], "w"))
print(len(rows), "vectors in", round(time.time() - t, 1), "s")
import collections
print(collections.Counter(r[2] for r in rows))
print("distinct scores", len({r[1] for r in rows}), "distinct macrovectors", len({r[3] for r in rows}))
