import numpy as np
import pandas as pd
import yfinance as yf

FEATS = [
    "r1", "r2", "r3", "r4", "r5", "r10", "r20", "v5", "v20", "v60", "a1", "a5",
    "hl", "vol", "m5", "m20", "m60", "rsi", "mk", "vx", "vxl",
]


def get(sym, start):
    d = yf.download(sym, start=start, auto_adjust=True, progress=False)
    d.columns = d.columns.get_level_values(0)
    d.index = pd.to_datetime(d.index).normalize()
    return d


def build_features(df, sp, vx):
    c = df["Close"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for k in range(1, 6):
        f[f"r{k}"] = r.shift(k - 1)
    f["r10"] = np.log(c).diff(10)
    f["r20"] = np.log(c).diff(20)
    f["v5"] = r.rolling(5).std()
    f["v20"] = r.rolling(20).std()
    f["v60"] = r.rolling(60).std()
    f["a1"] = r.abs()
    f["a5"] = r.abs().rolling(5).mean()
    f["hl"] = (df["High"] - df["Low"]) / c
    f["vol"] = np.log(df["Volume"].replace(0, np.nan)).diff()
    f["m5"] = c / c.rolling(5).mean() - 1
    f["m20"] = c / c.rolling(20).mean() - 1
    f["m60"] = c / c.rolling(60).mean() - 1
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    f["mk"] = np.log(sp["Close"]).diff().reindex(df.index)
    f["vx"] = np.log(vx["Close"]).diff().reindex(df.index)
    f["vxl"] = np.log(vx["Close"]).reindex(df.index)

    nxt = r.shift(-1)
    med = r.abs().rolling(60).median()
    yd = (nxt > 0).astype(float)
    yd[nxt.isna()] = np.nan
    yv = (nxt.abs() > med).astype(float)
    yv[nxt.isna() | med.isna()] = np.nan
    f["y_dir"] = yd
    f["y_vol"] = yv
    f["nxt"] = nxt
    return f.replace([np.inf, -np.inf], np.nan)


def make_dataset(ticker, start, target):
    f = build_features(get(ticker, start), get("^GSPC", start), get("^VIX", start))
    y = "y_dir" if target == "dir" else "y_vol"
    d = f[FEATS + [y, "nxt"]].rename(columns={y: "y"})
    return d.dropna()
