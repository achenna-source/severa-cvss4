"""Baseline B4: Qwen2.5-7B-Instruct without fine-tuning, CVSS v4.0 specification summary in the prompt
and 8 training examples retrieved by BM25 on the description (retrieval-augmented few-shot).
Runs on a fixed stratified sample of 2,000 test CVEs (seed 20261121). Outputs predicted vectors (JSONL)."""
import argparse, json, math, os, random, re, collections, time
import torch

SPEC = """You assign CVSS v4.0 base metrics to a vulnerability. Metrics and values:
AV Attack Vector: N network, A adjacent, L local, P physical.
AC Attack Complexity: L low, H high (attacker must evade or circumvent security-enhancing conditions).
AT Attack Requirements: N none, P present (deployment or execution conditions such as a race condition or a specific configuration).
PR Privileges Required: N none, L low, H high.
UI User Interaction: N none, P passive (involuntary interaction), A active (the user must perform specific actions).
VC VI VA confidentiality, integrity, availability impact on the vulnerable system: H high, L low, N none.
SC SI SA confidentiality, integrity, availability impact on subsequent systems (beyond the vulnerable system): H high, L low, N none.
Answer with the vector only, in the form CVSS:4.0/AV:_/AC:_/AT:_/PR:_/UI:_/VC:_/VI:_/VA:_/SC:_/SI:_/SA:_"""
BASE = ["AV", "AC", "AT", "PR", "UI", "VC", "VI", "VA", "SC", "SI", "SA"]
VALS = dict(AV="NALP", AC="LH", AT="NP", PR="NLH", UI="NPA", VC="HLN", VI="HLN", VA="HLN", SC="HLN", SI="HLN", SA="HLN")


def tok(t):
    return re.findall(r"[a-z0-9_\-]+", t.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [collections.Counter(tok(d)) for d in docs]
        self.len = [sum(d.values()) for d in self.docs]; self.avg = sum(self.len) / len(self.len)
        df = collections.Counter(w for d in self.docs for w in d)
        N = len(docs); self.idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.inv = collections.defaultdict(list)
        for i, d in enumerate(self.docs):
            for w in d:
                self.inv[w].append(i)
        self.k1, self.b = k1, b

    def top(self, q, k):
        sc = collections.Counter()
        for w in set(tok(q)):
            if w not in self.idf:
                continue
            for i in self.inv[w]:
                f = self.docs[i][w]
                sc[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return [i for i, _ in sc.most_common(k)]


def render(r):
    s = f"CWE: {', '.join(r['cwe_input']) or 'unknown'}\n"
    if r.get("v31_input"):
        s += f"CVSS v3.1 (same CNA): {r['v31_input']}\n"
    return s + f"Description: {r['description'][:1500]}"


def parse(text):
    found = dict(re.findall(r"\b(AV|AC|AT|PR|UI|VC|VI|VA|SC|SI|SA):([A-Z])", text))
    ok = all(m in found and found[m] in VALS[m] for m in BASE)
    return ("CVSS:4.0/" + "/".join(f"{m}:{found[m]}" for m in BASE)) if ok else None, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct"); ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--shots", type=int, default=8); ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    for r in rows:
        r["split2"] = "dev" if (r["split"] == "train" and r["published"][:10] >= "2026-03-01") else r["split"]
    train = [r for r in rows if r["split2"] == "train"]
    test = [r for r in rows if r["split2"] == "test"]
    prov = lambda c: {"cna@vuldb.com": "VulDB", "disclosure@vulncheck.com": "VulnCheck",
                      "security-advisories@github.com": "GitHub"}.get(c, "Other")
    rng = random.Random(20261121)
    by = collections.defaultdict(list)
    for r in test:
        by[prov(r["cna"])].append(r)
    sample = []
    for g, lst in sorted(by.items()):  # proportional stratified sample
        k = round(a.n * len(lst) / len(test)); sample += rng.sample(lst, min(k, len(lst)))
    bm = BM25([r["description"] for r in train])
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tk = AutoTokenizer.from_pretrained(a.model); tk.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map={"": 0}, torch_dtype=torch.float16,
                                                 quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                                 bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
    prompts = []
    for r in sample:
        msgs = [{"role": "system", "content": SPEC}]
        for i in bm.top(r["description"], a.shots)[::-1]:
            msgs += [{"role": "user", "content": render(train[i])}, {"role": "assistant", "content": train[i]["label_vector"]}]
        msgs.append({"role": "user", "content": render(r)})
        prompts.append(tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    os.makedirs(a.out, exist_ok=True); fo = open(os.path.join(a.out, "b4_predictions.jsonl"), "w")
    t0 = time.time(); nbad = 0
    for s in range(0, len(prompts), a.bs):
        enc = tk(prompts[s:s + a.bs], return_tensors="pt", padding=True, truncation=True, max_length=6000).to(0)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=64, do_sample=False, pad_token_id=tk.eos_token_id)
        for r, o in zip(sample[s:s + a.bs], out):
            txt = tk.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            vec, ok = parse(txt); nbad += (not ok)
            fo.write(json.dumps({"id": r["id"], "raw": txt, "pred": vec, "valid": ok}) + "\n")
        fo.flush()
        print(f"{s + a.bs}/{len(prompts)} invalid={nbad} {time.time() - t0:.0f}s", flush=True)
    json.dump({"n": len(sample), "invalid": nbad, "minutes": (time.time() - t0) / 60, "shots": a.shots},
              open(os.path.join(a.out, "b4_run.json"), "w"))


if __name__ == "__main__":
    main()
