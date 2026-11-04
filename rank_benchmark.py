import json
import os
import sys
import time
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None

DEFAULT = (
    "AAPL,MSFT,GOOGL,AMZN,JPM,BAC,WFC,C,JNJ,PFE,MRK,UNH,XOM,CVX,WMT,HD,MCD,DIS,PG,KO,PEP,"
    "INTC,CSCO,ORCL,IBM,T,VZ,BA,CAT,GE,MMM,NKE"
)


def arg(name, default):
    for a in sys.argv:
        if a.startswith(f"--{name}="):
            return a.split("=")[1]
    return default


TICKERS = arg("tickers", DEFAULT).split(",")
START = arg("start", "2005-01-01")
H = int(arg("horizon", 5))
FIRST = int(arg("first", 1260))
STEP = int(arg("step", 126))
STRIDE = int(arg("stride", 2))
COST = float(arg("cost", 5))
SEED = int(arg("seed", 0))
MINN = int(arg("minstocks", 15))
SKIP_DL = "--skip-dl" in sys.argv
OUT = "results"
os.makedirs(OUT, exist_ok=True)
np.random.seed(SEED)


def load_panel(tickers, start):
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, group_by="column", threads=True)
    out = {}
    for t in tickers:
        try:
            d = pd.DataFrame({f: raw[f][t] for f in ["Open", "High", "Low", "Close", "Volume"]}).dropna(subset=["Close"])
        except KeyError:
            print("no data for", t)
            continue
        d.index = pd.to_datetime(d.index).normalize()
        if len(d) > 800:
            out[t] = d
    return out


def feats(d):
    c = d["Close"]
    lc = np.log(c)
    r = lc.diff()
    v = d["Volume"].replace(0, np.nan)
    f = pd.DataFrame(index=d.index)
    f["r1"] = r
    f["r5"] = lc.diff(5)
    f["r10"] = lc.diff(10)
    f["r21"] = lc.diff(21)
    f["r63"] = lc.diff(63)
    f["r126"] = lc.diff(126)
    f["mom"] = lc.shift(21) - lc.shift(252)
    f["v5"] = r.rolling(5).std()
    f["v21"] = r.rolling(21).std()
    f["v63"] = r.rolling(63).std()
    f["hl"] = ((d["High"] - d["Low"]) / c).rolling(5).mean()
    f["vr"] = np.log(v / v.rolling(20).mean())
    f["m20"] = c / c.rolling(20).mean() - 1
    f["hi"] = c / c.rolling(252).max() - 1
    f["a1"] = r.abs()
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return f


data = load_panel(TICKERS, START)
print(len(data), "stocks loaded")
F = pd.concat({t: feats(d) for t, d in data.items()}, names=["ticker", "date"]).swaplevel().sort_index()
FW = pd.concat({t: np.log(d["Close"]).shift(-H) - np.log(d["Close"]) for t, d in data.items()}, names=["ticker", "date"]).swaplevel().sort_index()
F = F.replace([np.inf, -np.inf], np.nan)
cols = list(F.columns)

Fr = F.groupby(level=0).rank(pct=True) - 0.5
yr = FW.groupby(level=0).rank(pct=True) - 0.5
P = Fr.join(yr.rename("y")).join(FW.rename("fwd")).dropna()
cnt = P.groupby(level=0).size()
good = cnt[cnt >= MINN].index
P = P[P.index.get_level_values(0).isin(good)]
dates = P.index.get_level_values(0)
D = pd.DatetimeIndex(sorted(dates.unique()))
n = len(D)
print(len(P), "rows,", n, "dates,", len(cols), "features")

reg = {
    "ridge": lambda: Ridge(alpha=100.0),
    "lasso": lambda: Lasso(alpha=5e-5),
    "random_forest": lambda: RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_leaf=200, max_features=0.5, n_jobs=-1, random_state=SEED),
    "extra_trees": lambda: ExtraTreesRegressor(n_estimators=150, max_depth=6, min_samples_leaf=200, max_features=0.5, n_jobs=-1, random_state=SEED),
    "hist_gbm": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=200, random_state=SEED),
    "mlp": lambda: MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=100, random_state=SEED),
}
if XGBRegressor is not None:
    reg["xgboost"] = lambda: XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=100, n_jobs=4, random_state=SEED)
else:
    print("xgboost not available, skipping it")
if LGBMRegressor is not None:
    reg["lightgbm"] = lambda: LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=200, verbose=-1, random_state=SEED)
else:
    print("lightgbm not available, skipping it")
use_torch = torch is not None and not SKIP_DL
if torch is None and not SKIP_DL:
    print("torch not available, skipping mlp_torch")

factors = {"f_momentum": ("mom", 1), "f_reversal5": ("r5", -1), "f_reversal1": ("r1", -1), "f_lowvol": ("v63", -1)}


def fit_predict_torch(xa, ya, xb):
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    net = nn.Sequential(nn.Linear(xa.shape[1], 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    xt = torch.tensor(xa, dtype=torch.float32)
    yt = torch.tensor(ya, dtype=torch.float32)
    for ep in range(8):
        net.train()
        perm = torch.randperm(len(xt))
        for s in range(0, len(perm), 1024):
            idx = perm[s:s + 1024]
            opt.zero_grad()
            ((net(xt[idx]).squeeze(-1) - yt[idx]) ** 2).mean().backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(xb, dtype=torch.float32)).squeeze(-1).numpy()


names = list(reg) + (["mlp_torch"] if use_torch else [])
parts = {k: [] for k in names + list(factors) + ["random"]}
failed = set()
secs = {k: 0.0 for k in names}
rs = np.random.RandomState(SEED)

i = max(FIRST, 300)
while i < n:
    j = min(i + STEP, n)
    tr_dates = D[:max(0, i - H)][::STRIDE]
    trm = dates.isin(tr_dates)
    tem = (dates >= D[i]) & (dates <= D[j - 1])
    xa, ya = P.loc[trm, cols].values, P.loc[trm, "y"].values
    Pt = P.loc[tem]
    xb = Pt[cols].values
    for k in names:
        if k in failed:
            continue
        t0 = time.time()
        print(f"  {k} ...", end=" ", flush=True)
        try:
            if k == "mlp_torch":
                p = fit_predict_torch(xa, ya, xb)
            else:
                p = reg[k]().fit(xa, ya).predict(xb)
            parts[k].append(pd.Series(p, index=Pt.index))
            print(f"{time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            print(f"failed: {e}", flush=True)
            failed.add(k)
        secs[k] += time.time() - t0
    for k, (c, sg) in factors.items():
        parts[k].append(pd.Series(sg * Pt[c].values, index=Pt.index))
    parts["random"].append(pd.Series(rs.randn(len(Pt)), index=Pt.index))
    print(f"fold done: {i} / {n}", flush=True)
    i = j

names = [k for k in names if k not in failed]
scores = {k: pd.concat(parts[k]) for k in names + list(factors) + ["random"]}
scores["ensemble"] = pd.concat([scores[k].groupby(level=0).rank(pct=True) for k in names], axis=1).mean(axis=1)

R = P["fwd"].unstack()
PPY = 252 / H
rows = []
curves = {}
for k, s in scores.items():
    S = s.unstack().reindex(R.index).dropna(how="all")
    S = S.iloc[::H]
    Rk = R.reindex(S.index).where(S.notna())
    Sr = S.rank(axis=1, method="first")
    m = S.notna().sum(axis=1)
    q = np.maximum(1, (m * 0.2).round().astype(int))
    top = Sr.gt(m - q, axis=0) & S.notna()
    bot = Sr.le(q, axis=0) & S.notna()
    wl = top.astype(float).div(top.sum(axis=1), axis=0)
    ws = bot.astype(float).div(bot.sum(axis=1), axis=0)
    r0 = Rk.fillna(0)
    ic = S.rank(axis=1).corrwith(Rk.rank(axis=1), axis=1).dropna()
    gross = (wl * r0).sum(axis=1) - (ws * r0).sum(axis=1)
    w = wl - ws
    turn = w.diff().abs().sum(axis=1)
    turn.iloc[0] = w.iloc[0].abs().sum()
    net = gross - COST * 1e-4 * turn
    ex = (wl * r0).sum(axis=1) - Rk.mean(axis=1)
    sd = lambda x: float(x.std()) if x.std() > 0 else np.nan
    rows.append({
        "model": k,
        "ic_mean": float(ic.mean()),
        "ic_t": float(ic.mean() / (ic.std() / np.sqrt(len(ic)))),
        "ls_gross_ann": float(gross.mean() * PPY),
        "ls_sharpe_gross": float(gross.mean() / sd(gross) * np.sqrt(PPY)),
        "ls_sharpe_net": float(net.mean() / sd(net) * np.sqrt(PPY)),
        "turnover": float(turn.mean()),
        "top_excess_ann": float(ex.mean() * PPY),
        "periods": int(len(ic)),
    })
    curves[k] = net.cumsum()

res = pd.DataFrame(rows).sort_values("ic_t", ascending=False)
res["significant"] = res["ic_t"] > 3.0
tag = f"{H}d"
res.to_csv(f"{OUT}/rank_{tag}.csv", index=False)
info = {
    "stocks": len(data),
    "horizon_days": H,
    "cost_bps_per_side": COST,
    "test_from": str(D[max(FIRST, 300)].date()),
    "test_to": str(D[-1].date()),
    "non_overlapping_periods": int(res["periods"].max()),
    "significance_rule": "|IC t-stat| > 3 (stricter than 1.96 because many strategies are compared)",
}
json.dump(info, open(f"{OUT}/rank_{tag}.json", "w"), indent=2)
print(json.dumps(info, indent=2))
print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

order = res.sort_values("ic_mean")
plt.figure(figsize=(9, 0.4 * len(order) + 1.5))
colors = ["tab:green" if b else "tab:gray" for b in order["significant"]]
se = order["ic_mean"] / order["ic_t"].replace(0, np.nan)
plt.barh(order["model"], order["ic_mean"], xerr=1.96 * se.abs(), color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel(f"mean rank IC, {H}-day horizon (green = |t| > 3)")
plt.tight_layout()
plt.savefig(f"{OUT}/rank_ic_{tag}.png", dpi=130)
plt.close()

show = list(res.sort_values("ic_t", ascending=False)["model"].head(3))
for k in ["f_momentum", "f_reversal5", "random"]:
    if k not in show:
        show.append(k)
plt.figure(figsize=(10, 5))
for k in show:
    plt.plot(curves[k].index, curves[k].values, label=k)
plt.axhline(0, color="black", linewidth=0.8)
plt.title(f"Cumulative long-short return (top vs bottom 20%), {H}-day, net of {COST:g} bps per side")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/rank_curve_{tag}.png", dpi=130)
plt.close()
