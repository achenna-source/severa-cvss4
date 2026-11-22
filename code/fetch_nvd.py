"""Fetch NVD CVE records (API 2.0) published in [START, END) by 110-day windows.
Saves one JSON per page under data/raw_nvd/. Public data, no API key (6 s between calls)."""
import json, os, sys, time, datetime as dt, urllib.request, urllib.parse
OUT = sys.argv[1]
START = dt.datetime(2023, 1, 1)
END = dt.datetime(2026, 11, 21)
BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
def get(params, tries=6):
    url = BASE + "?" + urllib.parse.urlencode(params)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research-script"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:
            print("retry", k, e, flush=True); time.sleep(15 * (k + 1))
    raise RuntimeError(url)
w0 = START
while w0 < END:
    w1 = min(w0 + dt.timedelta(days=110), END)
    idx = 0
    while True:
        fn = os.path.join(OUT, f"nvd_{w0:%Y%m%d}_{idx:06d}.json")
        if os.path.exists(fn):
            d = json.load(open(fn, encoding="utf-8"))
        else:
            d = get({"pubStartDate": w0.strftime("%Y-%m-%dT00:00:00.000"),
                     "pubEndDate": (w1 - dt.timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S.999"),
                     "resultsPerPage": 2000, "startIndex": idx})
            json.dump(d, open(fn, "w", encoding="utf-8"))
            time.sleep(6.5)
        n = len(d.get("vulnerabilities", []))
        print(f"{w0:%Y-%m-%d} idx={idx} got={n} total={d.get('totalResults')}", flush=True)
        idx += n
        if n == 0 or idx >= d.get("totalResults", 0):
            break
    w0 = w1
print("DONE", flush=True)
