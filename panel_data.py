import numpy as np
import pandas as pd
import yfinance as yf

from countries import COUNTRIES
from features import get


def load_close(tickers, start):
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, group_by="column", threads=True)
    out = {}
    for t in tickers:
        try:
            s = raw["Close"][t].dropna()
        except KeyError:
            continue
        if len(s) > 300:
            out[t] = s
    return pd.DataFrame(out), [t for t in tickers if t not in out]


def load_weekly(use, start, minn):
    crs = {}
    for name in use:
        px, missing = load_close(COUNTRIES[name], start)
        px.index = pd.to_datetime(px.index).normalize()
        print(f"[{name}] loaded {px.shape[1]}/{len(COUNTRIES[name])}" + (f", missing: {' '.join(missing)}" if missing else ""), flush=True)
        if px.shape[1] < minn:
            print(f"[{name}] skipped (too few stocks)")
            continue
        r = np.log(px).diff()
        r = r.where(r.abs() < 0.4)
        cr = r.mean(axis=1)
        cr[r.notna().sum(axis=1) < minn] = np.nan
        crs[name] = cr
    D = pd.DataFrame(crs)
    D = D[D.index >= pd.Timestamp(start)]
    last_day = D.dropna(how="all").index.max()
    nobs = D.notna().resample("W-FRI").sum()
    ret_w = D.resample("W-FRI").sum(min_count=3).where(nobs >= 3)
    ss = (D ** 2).resample("W-FRI").sum(min_count=3)
    vol = np.sqrt(ss / nobs * 252).where(nobs >= 3)
    if vol.index.max() > last_day:
        vol, ret_w = vol.iloc[:-1], ret_w.iloc[:-1]
    vx = get("^VIX", start)["Close"].resample("W-FRI").last().reindex(vol.index)
    return vol, ret_w, np.log(vx)
