import os, json, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results"
NAME = {"dir": "next-day direction", "vol": "next-day volatility"}


def load(t, target):
    parts = []
    for tag, kind in [(target, "tabular"), (target + "_deep", "deep")]:
        f = f"{OUT}/benchmark_{t}_{tag}.csv"
        if os.path.exists(f):
            d = pd.read_csv(f)
            d["kind"] = kind
            parts.append(d)
    if not parts:
        return None, None
    d = pd.concat(parts)
    base = float(d.loc[d.model == "baseline", "acc"].iloc[0])
    d = d[~d.model.isin(["baseline", "ensemble"])].copy()
    d["label"] = [m + (" (deep)" if k == "deep" else "") for m, k in zip(d.model, d.kind)]
    return d, base


def acc_chart(t, target):
    d, base = load(t, target)
    if d is None:
        return
    ci = json.load(open(f"{OUT}/benchmark_{t}_{target}.json"))["ci95"]
    d = d.sort_values("acc")
    n_win = int(d["beats_baseline"].astype(bool).sum())
    colors = ["tab:green" if b else "tab:gray" for b in d["beats_baseline"].astype(bool)]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(d) + 2.2))
    bars = ax.barh(d["label"], d["acc"], color=colors)
    for b, v in zip(bars, d["acc"]):
        ax.text(v + 0.001, b.get_y() + b.get_height() / 2, f"{v*100:.1f}%", va="center", fontsize=9)
    ax.axvline(base, color="tab:red", ls="--", label=f"Majority baseline {base*100:.1f}%")
    ax.axvspan(base - ci, base + ci, color="tab:red", alpha=0.12, label=f"95% margin of error (±{ci*100:.1f} pts)")
    ax.axvline(0.5, color="black", ls=":", lw=1, label="Coin flip 50%")
    lo = min(d["acc"].min(), base - ci, 0.5) - 0.01
    hi = max(d["acc"].max(), base + ci) + 0.025
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Walk-forward accuracy")
    ax.set_title(f"{t}: {NAME[target]} accuracy\n{n_win} of {len(d)} models beat the baseline "
                 f"(green = significant after Bonferroni correction)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/labeled_acc_{t}_{target}.png", dpi=140)
    plt.close()


def auc_chart(target):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True)
    for ax, t in zip(axes, ["AAPL", "MSFT"]):
        d, _ = load(t, target)
        if d is None:
            continue
        d = d.sort_values("auc")
        cols = ["tab:orange" if k == "deep" else "tab:blue" for k in d["kind"]]
        bars = ax.barh(d["model"], d["auc"], color=cols)
        for b, v in zip(bars, d["auc"]):
            ax.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
        ax.axvline(0.5, color="black", ls="--", label="AUC 0.5 = no skill")
        ax.set_xlim(0.45, 0.63)
        ax.set_xlabel("AUC")
        ax.set_title(f"{t}: {NAME[target]} AUC\n(blue = tabular, orange = deep)", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/labeled_auc_{target}.png", dpi=140)
    plt.close()


def leak_chart(t="AAPL"):
    f = f"{OUT}/leakage_{t}.csv"
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    scen = list(dict.fromkeys(d.scenario))
    models = list(dict.fromkeys(d.model))
    y = np.arange(len(scen))
    h = 0.38
    cols = ["tab:blue", "tab:orange"]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for i, m in enumerate(models):
        g = d[d.model == m].set_index("scenario").reindex(scen)
        pos = y + (i - 0.5) * h
        ax.barh(pos, g["accuracy"], h, color=cols[i], label=m)
        for p, v in zip(pos, g["accuracy"]):
            ax.text(v + 0.005, p, f"{v*100:.1f}%", va="center", fontsize=8)
    base = d.groupby("scenario")["baseline"].mean().reindex(scen)
    ax.plot(base.values, y, "k|", markersize=28, mew=2, linestyle="none", label="Majority baseline")
    ax.axvline(0.5, color="black", ls=":", lw=1, label="Coin flip 50%")
    ax.set_yticks(y)
    ax.set_yticklabels([textwrap.fill(s, 40) for s in scen], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.4, 1.06)
    ax.set_xlabel("Direction accuracy")
    ax.set_title(f"{t}: how leakage inflates accuracy\nHonest setups stay near the baseline; "
                 f"70-99% means the answer leaked into the features", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/labeled_leakage_{t}.png", dpi=140)
    plt.close()


for t in ["AAPL", "MSFT"]:
    for tg in ["dir", "vol"]:
        acc_chart(t, tg)
for tg in ["dir", "vol"]:
    auc_chart(tg)
leak_chart("AAPL")
print("saved labeled_* charts in results/")
