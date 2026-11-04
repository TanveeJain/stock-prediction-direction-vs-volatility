import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results"
MODELS = ["logreg", "extra_trees", "random_forest"]
LABELS = {"dir": ("Down", "Up"), "vol": ("Small move", "Large move")}
TITLE = {"dir": "next-day direction", "vol": "next-day volatility"}


def load(t, target):
    f = f"{OUT}/preds_{t}_{target}_preds.csv"
    return pd.read_csv(f, index_col=0, parse_dates=True) if os.path.exists(f) else None


def confusion(t, target):
    d = load(t, target)
    if d is None:
        return
    models = [m for m in MODELS if m in d.columns]
    y = d["y"].values.astype(int)
    base_acc = float((d["base_pred"].values == y).mean())
    neg, pos = LABELS[target]
    fig, axes = plt.subplots(1, len(models), figsize=(5.3 * len(models), 5.6), squeeze=False)
    for ax, m in zip(axes[0], models):
        pred = (d[m].values > 0.5).astype(int)
        cm = np.array([[int(((y == a) & (pred == b)).sum()) for b in (0, 1)] for a in (0, 1)])
        rows = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(rows, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({rows[i, j] * 100:.0f}% of row)", ha="center", va="center",
                        fontsize=11, color="white" if rows[i, j] > 0.6 else "black")
        acc = np.trace(cm) / cm.sum()
        ax.set_xticks([0, 1]); ax.set_xticklabels([neg, pos])
        ax.set_yticks([0, 1]); ax.set_yticklabels([neg, pos])
        ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
        ax.set_title(f"{m}\naccuracy {acc * 100:.1f}%  vs  baseline {base_acc * 100:.1f}%", fontsize=10)
    fig.suptitle(f"{t}: {TITLE[target]}, confusion matrices (walk-forward, {len(d)} test days)\n"
                 f"Colour = share of each true class. A model with skill has a dark diagonal in BOTH rows.",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/confusion_{t}_{target}.png", dpi=140)
    plt.close()


def backtest(t):
    d = load(t, "dir")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bh = np.exp(d["ret"].cumsum())
    ax.plot(bh.index, bh.values, color="black", lw=2.2, label=f"Buy & hold ({bh.iloc[-1]:.2f}x)")
    for m in MODELS:
        if m in d.columns:
            g = np.exp((d["ret"] * (d[m] > 0.5)).cumsum())
            ax.plot(g.index, g.values, lw=1.4, label=f"{m}: long when model says up ({g.iloc[-1]:.2f}x)")
    ax.axhline(1, color="gray", ls=":", lw=1)
    ax.set_ylabel("Growth of 1")
    ax.set_title(f"{t}: walk-forward backtest, NO costs, honest (no leakage)\n"
                 f"Differences from buy & hold are noise: direction AUC is about 0.50", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT}/backtest_labeled_{t}.png", dpi=140)
    plt.close()


for t in ["AAPL", "MSFT"]:
    for tg in ["dir", "vol"]:
        confusion(t, tg)
    backtest(t)
print("saved confusion_* and backtest_labeled_* in results/")
