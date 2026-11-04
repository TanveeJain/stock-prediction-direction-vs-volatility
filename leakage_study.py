import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import FEATS, build_features, get

warnings.filterwarnings("ignore")


def arg(name, default):
    for a in sys.argv:
        if a.startswith(f"--{name}="):
            return a.split("=")[1]
    return default


TICKER = arg("ticker", "AAPL")
START = arg("start", "2005-01-01")
FIRST = int(arg("first", 1260))
STEP = int(arg("step", 126))
OUT = "results"
os.makedirs(OUT, exist_ok=True)

df = get(TICKER, START)
f = build_features(df, get("^GSPC", START), get("^VIX", START))
c = np.log(df["Close"])
f["y_same"] = (c.diff() > 0).astype(float)
f["y_5d"] = ((c.shift(-5) - c) > 0).astype(float)
f.loc[c.diff().isna(), "y_same"] = np.nan
f.loc[(c.shift(-5) - c).isna(), "y_5d"] = np.nan
f["close"] = df["Close"]
f["close_next"] = df["Close"].shift(-1)

models = {
    "logreg": lambda: LogisticRegression(C=0.1, max_iter=500),
    "extra_trees": lambda: ExtraTreesClassifier(n_estimators=200, max_depth=10, min_samples_leaf=3, n_jobs=-1, random_state=0),
}


def prep(cols, ycol):
    z = f[cols + [ycol]].dropna()
    return z[cols].values, z[ycol].values.astype(int)


def random_split(cols, ycol, make):
    X, y = prep(cols, ycol)
    xa, xb, ya, yb = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=0)
    sc = StandardScaler().fit(xa)
    m = make().fit(np.clip(sc.transform(xa), -5, 5), ya)
    return float((m.predict(np.clip(sc.transform(xb), -5, 5)) == yb).mean()), float(max(yb.mean(), 1 - yb.mean()))


def walk_forward(cols, ycol, make, embargo):
    X, y = prep(cols, ycol)
    hit, tot, maj = 0, 0, 0
    i = FIRST
    while i < len(y):
        j = min(i + STEP, len(y))
        e = max(0, i - 1 - embargo)
        sc = StandardScaler().fit(X[:e])
        m = make().fit(np.clip(sc.transform(X[:e]), -5, 5), y[:e])
        p = m.predict(np.clip(sc.transform(X[i:j]), -5, 5))
        hit += int((p == y[i:j]).sum())
        maj += int((y[i:j] == int(y[:e].mean() >= 0.5)).sum())
        tot += j - i
        i = j
    return hit / tot, maj / tot


scen = [
    ("1. Honest: next-day direction, walk-forward", lambda mk: walk_forward(FEATS, "y_dir", mk, 0)),
    ("2. Random shuffle split, next-day direction", lambda mk: random_split(FEATS, "y_dir", mk)),
    ("3. 5-day direction, random shuffle split (overlapping labels)", lambda mk: random_split(FEATS, "y_5d", mk)),
    ("4. 5-day direction, walk-forward with 5-day embargo (honest)", lambda mk: walk_forward(FEATS, "y_5d", mk, 5)),
    ("5. Same-day market return as feature (information not yet known)", lambda mk: walk_forward(["mk", "vx"], "y_same", mk, 0)),
    ("6. Target not shifted, own return left in features (off-by-one bug)", lambda mk: walk_forward(FEATS, "y_same", mk, 0)),
]
rows = []
for name, fn in scen:
    for mname, make in models.items():
        acc, base = fn(make)
        rows.append({"scenario": name, "model": mname, "accuracy": acc, "baseline": base})
        print(f"{acc:.3f}  (baseline {base:.3f})  {mname:12s} {name}", flush=True)
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/leakage_{TICKER}.csv", index=False)

z = f[["close", "close_next"]].dropna()
xa, xb, ya, yb = train_test_split(z[["close"]].values, z["close_next"].values, test_size=0.25, shuffle=True, random_state=0)
r2 = float(LinearRegression().fit(xa, ya).score(xb, yb))
naive = float(1 - ((yb - xb[:, 0]) ** 2).sum() / ((yb - yb.mean()) ** 2).sum())
print(f"\nPredicting next-day PRICE with a shuffled split: R-squared {r2:.4f}; copying today's price gives {naive:.4f}")
print("A high R-squared on price levels says nothing about skill, the price barely changes in one day.")

pv = res.pivot(index="scenario", columns="model", values="accuracy").reindex([s for s, _ in scen])
fig, ax = plt.subplots(figsize=(11, 4.8))
pv.plot.barh(ax=ax)
ax.axvline(0.5, color="black", linewidth=0.8)
ax.set_xlabel("accuracy")
ax.invert_yaxis()
ax.set_title(f"{TICKER}: how leakage inflates accuracy")
plt.tight_layout()
plt.savefig(f"{OUT}/leakage_{TICKER}.png", dpi=130)
plt.close()
print(f"saved results/leakage_{TICKER}.csv and .png")
