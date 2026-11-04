import os
import sys
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

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
STRIDE = int(arg("stride", 2))
SEED = int(arg("seed", 0))
MINN = int(arg("minstocks", 15))
OUT = "results"
os.makedirs(OUT, exist_ok=True)


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
F = pd.concat({t: feats(d) for t, d in data.items()}, names=["ticker", "date"]).swaplevel().sort_index()
FW = pd.concat({t: np.log(d["Close"]).shift(-H) - np.log(d["Close"]) for t, d in data.items()}, names=["ticker", "date"]).swaplevel().sort_index()
F = F.replace([np.inf, -np.inf], np.nan)
cols = list(F.columns)

Fr = F.groupby(level=0).rank(pct=True) - 0.5
yr = FW.groupby(level=0).rank(pct=True) - 0.5
ok = Fr.notna().all(axis=1)
cnt = ok.groupby(level=0).sum()
last = cnt[cnt >= MINN].index.max()

train = Fr[ok].join(yr.rename("y")).dropna(subset=["y"])
tdates = train.index.get_level_values(0)
keep = pd.DatetimeIndex(sorted(tdates.unique()))[::STRIDE]
train = train[tdates.isin(keep)]
test = Fr[ok].xs(last, level=0)
print(f"as of {last.date()}, {len(test)} stocks, trained on {len(train)} rows, horizon {H} days")

xa, ya = train[cols].values, train["y"].values
xb = test[cols].values

models = {
    "ridge": lambda: Ridge(alpha=100.0),
    "lasso": lambda: Lasso(alpha=5e-5),
    "random_forest": lambda: RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_leaf=200, max_features=0.5, n_jobs=-1, random_state=SEED),
    "extra_trees": lambda: ExtraTreesRegressor(n_estimators=150, max_depth=6, min_samples_leaf=200, max_features=0.5, n_jobs=-1, random_state=SEED),
    "hist_gbm": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=200, random_state=SEED),
}
if XGBRegressor is not None:
    models["xgboost"] = lambda: XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=100, n_jobs=4, random_state=SEED)
if LGBMRegressor is not None:
    models["lightgbm"] = lambda: LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=200, verbose=-1, random_state=SEED)

ranks = {}
for k, make in models.items():
    p = make().fit(xa, ya).predict(xb)
    ranks[k] = pd.Series(p, index=test.index).rank(pct=True)
    print(f"  {k} done", flush=True)

R = pd.DataFrame(ranks)
res = pd.DataFrame(index=test.index)
res["score"] = R.mean(axis=1)
res["models_top20"] = (R >= 0.8).sum(axis=1)
res["models_bottom20"] = (R <= 0.2).sum(axis=1)
res["momentum_rank"] = test["mom"].rank(ascending=False).astype(int)
res["close"] = [float(data[t]["Close"].loc[:last].iloc[-1]) for t in res.index]
res["ret_5d_pct"] = [float(100 * (data[t]["Close"].loc[:last].iloc[-1] / data[t]["Close"].loc[:last].iloc[-6] - 1)) for t in res.index]
res = res.sort_values("score", ascending=False)
res.insert(0, "rank", range(1, len(res) + 1))
res.to_csv(f"{OUT}/rank_today.csv")

pd.set_option("display.width", 140)
fmt = lambda v: f"{v:.2f}"
print()
print("Top 5 by ensemble score:")
print(res.head(5).to_string(float_format=fmt))
print()
print("Bottom 5:")
print(res.tail(5).to_string(float_format=fmt))
print()
print("Read this first: in the backtest no model had a statistically significant edge")
print("(mean rank IC about 0.015 to 0.018, t-stat about 2 against a bar of 3, and a random")
print("control scored almost as well). Treat this ranking as a demo of the method, not advice.")
print("Saved to results/rank_today.csv")
