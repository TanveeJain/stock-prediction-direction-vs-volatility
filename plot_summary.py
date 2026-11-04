import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(t, target):
    parts = []
    for tag in [target, target + "_deep"]:
        f = f"results/benchmark_{t}_{tag}.csv"
        if os.path.exists(f):
            d = pd.read_csv(f)
            d["kind"] = "deep" if tag.endswith("deep") else "tabular"
            parts.append(d)
    d = pd.concat(parts)
    d = d[~d["model"].isin(["baseline", "ensemble"])]
    return d.sort_values("auc")

for target in ["vol", "dir"]:
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for a, t in zip(ax, ["AAPL", "MSFT"]):
        d = load(t, target)
        col = ["tab:orange" if k == "deep" else "tab:blue" for k in d["kind"]]
        a.barh(d["model"], d["auc"], color=col)
        a.axvline(0.5, color="black", linewidth=0.8)
        a.set_xlim(0.45, 0.62)
        a.set_title(f"{t} {target}: AUC (blue = tabular, orange = deep)")
        a.set_xlabel("AUC (0.5 = no skill)")
    plt.tight_layout()
    plt.savefig(f"results/auc_{target}_compare.png", dpi=130)
    plt.close()
print("saved results/auc_vol_compare.png and auc_dir_compare.png")
