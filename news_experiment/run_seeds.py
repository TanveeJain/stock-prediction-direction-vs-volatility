import json
import subprocess
import sys
import numpy as np

seeds = [1, 2, 3, 4, 5]
summary = {}

for mode in ["news", "price"]:
    flag = [] if mode == "news" else ["--price-only"]
    rows = []
    for s in seeds:
        print("running", mode, "seed", s, flush=True)
        subprocess.run([sys.executable, "train.py", f"--seed={s}"] + flag, check=True, stdout=subprocess.DEVNULL)
        rows.append(json.load(open(f"results/metrics_{mode}_{s}.json")))
    acc = np.array([r["direction_acc"] for r in rows])
    summary[mode] = {
        "acc_mean": float(acc.mean()),
        "acc_std": float(acc.std()),
        "up_rate": float(np.mean([r["up_rate"] for r in rows])),
        "model_rmse": float(np.mean([r["model_rmse"] for r in rows])),
        "naive_rmse": float(np.mean([r["naive_rmse"] for r in rows])),
    }

print(json.dumps(summary, indent=2))
json.dump(summary, open("results/summary.json", "w"), indent=2)
