"""Multi-head CVSS v4.0 base-metric classifier on a transformer backbone (Kaggle GPU).

Two backbones share this script:
  * encoder (e.g. microsoft/deberta-v3-base), full fine-tuning           -> baseline B2
  * decoder LLM (e.g. Qwen/Qwen2.5-7B-Instruct), 4-bit QLoRA, last-token  -> B5 / Ours backbone
The script only produces calibrated per-metric probability tables for the calibration and
test splits; decoding, the knowledge layer and abstention run afterwards (decode.py, local).

Usage (Kaggle):
  python kaggle_train_heads.py --data /kaggle/input/<ds>/splits.jsonl --backbone deberta \
      --model microsoft/deberta-v3-base --seed 1 --out /kaggle/working/deberta_s1
"""
import argparse, json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

METRICS = [("AV", "NALP"), ("AC", "LH"), ("AT", "NP"), ("PR", "NLH"), ("UI", "NPA"),
           ("VC", "HLN"), ("VI", "HLN"), ("VA", "HLN"), ("SC", "HLN"), ("SI", "HLN"), ("SA", "HLN")]


def parse_vector(vec):
    parts = dict(p.split(":") for p in vec.split("/")[1:])
    return [vals.index(parts[m]) for m, vals in METRICS]


def render_input(r, use_v31):
    cwe = ", ".join(r.get("cwe_input", [])) or "unknown"
    txt = f"CWE: {cwe}\n"
    if use_v31 and r.get("v31_input"):
        txt += f"CVSS v3.1: {r['v31_input']}\n"
    txt += f"Description: {r['description']}"
    return txt


class CVEData(Dataset):
    def __init__(self, rows, tok, max_len, use_v31, llm):
        self.rows, self.tok, self.max_len, self.use_v31, self.llm = rows, tok, max_len, use_v31, llm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        text = render_input(r, self.use_v31)
        if self.llm:
            text = ("Assess the CVSS v4.0 base metrics of this vulnerability.\n" + text +
                    "\nCVSS v4.0 base vector:")
        enc = self.tok(text, truncation=True, max_length=self.max_len)
        y = parse_vector(r["label_vector"]) if r.get("label_vector") else [-100] * len(METRICS)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": y}


def collate(batch, pad_id, left_pad):
    L = max(len(b["input_ids"]) for b in batch)
    ids, att = [], []
    for b in batch:
        n = L - len(b["input_ids"])
        if left_pad:
            ids.append([pad_id] * n + b["input_ids"]); att.append([0] * n + b["attention_mask"])
        else:
            ids.append(b["input_ids"] + [pad_id] * n); att.append(b["attention_mask"] + [0] * n)
    return (torch.tensor(ids), torch.tensor(att), torch.tensor([b["labels"] for b in batch]))


class Heads(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.drop = nn.Dropout(0.1)
        self.heads = nn.ModuleList([nn.Linear(hidden, len(v)) for _, v in METRICS])

    def forward(self, h):
        h = self.drop(h)
        return [head(h) for head in self.heads]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--backbone", choices=["deberta", "llm"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--use_v31", type=int, default=0)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_deploy", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    from transformers import AutoModel, AutoTokenizer
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    for r in rows:  # protocol v1: March 2026 is held out from model training as the dev split
        r["split2"] = "dev" if (r["split"] == "train" and r["published"][:10] >= "2026-03-01") else r["split"]
    split = {s: [r for r in rows if r["split2"] == s] for s in ("train", "dev", "calib", "test")}
    dep = os.path.join(os.path.dirname(a.data), "deploy.jsonl")
    split["deploy"] = [json.loads(l) for l in open(dep, encoding="utf-8")] if (os.path.exists(dep) and not a.no_deploy) else []
    if a.train_limit:
        random.shuffle(split["train"]); split["train"] = split["train"][: a.train_limit]
    llm = a.backbone == "llm"
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda"
    if llm:
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        base = AutoModel.from_pretrained(a.model, quantization_config=bnb, device_map={"": 0},
                                         torch_dtype=torch.float16)
        base.config.use_cache = False
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        base = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                               "gate_proj", "up_proj", "down_proj"]))
        hidden = base.config.hidden_size
        lr = a.lr or 1e-4
    else:
        base = AutoModel.from_pretrained(a.model, torch_dtype=torch.float32).float().to(dev)
        hidden = base.config.hidden_size
        lr = a.lr or 3e-5
    heads = Heads(hidden).to(dev).float()
    params = [p for p in base.parameters() if p.requires_grad] + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    mk = lambda rs, sh: DataLoader(CVEData(rs, tok, a.max_len, a.use_v31, llm), batch_size=a.bs, shuffle=sh,
                                   collate_fn=lambda b: collate(b, tok.pad_token_id, llm))
    tr = mk(split["train"], True)
    total = math.ceil(len(tr) * a.epochs / a.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * total))) * max(0.0, 1 - s / total))
    scaler = torch.amp.GradScaler("cuda", enabled=not llm)
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    def pooled(ids, att):
        out = base(input_ids=ids, attention_mask=att).last_hidden_state
        if llm:  # left padding: last position is the final prompt token
            return out[:, -1, :].float()
        return out[:, 0, :].float()

    step, t0 = 0, time.time()
    base.train(); heads.train()
    n_batches = int(len(tr) * a.epochs)
    it = iter(tr)
    for b in range(n_batches):
        try:
            ids, att, y = next(it)
        except StopIteration:
            it = iter(tr); ids, att, y = next(it)
        ids, att, y = ids.to(dev), att.to(dev), y.to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=not llm):
            logits = heads(pooled(ids, att))
            loss = sum(ce(lg.float(), y[:, k]) for k, lg in enumerate(logits)) / len(METRICS)
        scaler.scale(loss / a.accum).backward()
        if (b + 1) % a.accum == 0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
        if b % 200 == 0:
            print(f"batch {b}/{n_batches} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)

    base.eval(); heads.eval()
    for s in ("dev", "calib", "test", "deploy"):
        if not split[s]:
            continue
        probs = [[] for _ in METRICS]
        with torch.no_grad():
            for ids, att, _ in mk(split[s], False):
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = heads(pooled(ids.to(dev), att.to(dev)))
                for k, lg in enumerate(logits):
                    probs[k].append(torch.softmax(lg.float(), -1).cpu().numpy())
        np.savez_compressed(os.path.join(a.out, f"probs_{s}.npz"),
                            ids=np.array([r["id"] for r in split[s]]),
                            **{m: np.concatenate(probs[k]) for k, (m, _) in enumerate(METRICS)})
    json.dump(vars(a) | {"n_train": len(split["train"]), "minutes": (time.time() - t0) / 60},
              open(os.path.join(a.out, "run.json"), "w"), indent=1)
    print("done", flush=True)



if __name__ == "__main__":
    import glob, subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes", "peft"], check=False)
    D = os.path.dirname(sorted(glob.glob("/kaggle/input/**/splits.jsonl", recursive=True))[0])
    sys.argv = ["x", "--data", os.path.join(D, "splits.jsonl"), "--backbone", "llm", "--model", "Qwen/Qwen2.5-7B-Instruct",
                "--seed", "3", "--epochs", "1", "--bs", "4", "--accum", "4", "--max_len", "384", "--use_v31", "1",
                "--no_deploy", "--out", "/kaggle/working/qwen7_s3"]
    main()
