import glob, os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes", "peft"], check=False)
D = os.path.dirname(sorted(glob.glob("/kaggle/input/**/kaggle_train_heads.py", recursive=True))[0])
cmd = [sys.executable, os.path.join(D, "kaggle_train_heads.py"), "--data", os.path.join(D, "splits.jsonl"),
       "--backbone", "llm", "--model", "Qwen/Qwen2.5-7B-Instruct", "--seed", "1", "--epochs", "1",
       "--bs", "4", "--accum", "4", "--max_len", "384", "--use_v31", "1", "--out", "/kaggle/working/qwen7_s1"]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=False)
