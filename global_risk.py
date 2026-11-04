import json
import os
import sys
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from countries import COUNTRIES
from features import get

warnings.filterwarnings("ignore")


def arg(name, default):
    for a in sys.argv:
        if a.startswith(f"--{name}="):
            return a.split("=")[1]
    return default


START = arg("start", "2006-01-01")
COST = float(arg("cost", 10)) * 1e-4
MINN = int(arg("minstocks", 5))
FIRSTW = int(arg("firstw", 312))
REFIT = int(arg("refit", 26))
SEED = int(arg("seed", 0))
USE = arg("countries", ",".join(COUNTRIES)).split(",")
OUT = "results"
os.makedirs(OUT, exist_ok=True)


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


crs = {}
for name in USE:
    px, missing = load_close(COUNTRIES[name], START)
    px.index = pd.to_datetime(px.index).normalize()
    print(f"[{name}] loaded {px.shape[1]}/{len(COUNTRIES[name])}" + (f", missing: {' '.join(missing)}" if missing else ""), flush=True)
    if px.shape[1] < MINN:
        print(f"[{name}] skipped (too few stocks)")
        continue
    r = np.log(px).diff()
    r = r.where(r.abs() < 0.4)
    cr = r.mean(axis=1)
    cr[r.notna().sum(axis=1) < MINN] = np.nan
    crs[name] = cr

D = pd.DataFrame(crs)
D = D[D.index >= pd.Timestamp(START)]
last_day = D.dropna(how="all").index.max()
nobs = D.notna().resample("W-FRI").sum()
ret_w = D.resample("W-FRI").sum(min_count=3).where(nobs >= 3)
ss = (D ** 2).resample("W-FRI").sum(min_count=3)
vol = np.sqrt(ss / nobs * 252).where(nobs >= 3)
if vol.index.max() > last_day:
    vol, ret_w = vol.iloc[:-1], ret_w.iloc[:-1]
lrv = np.log(vol)

vx = get("^VIX", START)["Close"].resample("W-FRI").last().reindex(vol.index)
vix = pd.DataFrame({c: np.log(vx) for c in vol.columns})

feats = {
    "l1": lrv,
    "l4": lrv.rolling(4, min_periods=3).mean(),
    "l13": lrv.rolling(13, min_periods=10).mean(),
    "l52": lrv.rolling(52, min_periods=40).mean(),
    "ew": np.log(np.sqrt((vol ** 2).ewm(alpha=0.1, min_periods=20).mean())),
    "lev": lrv.expanding(min_periods=52).mean(),
    "r1": ret_w,
    "r4": ret_w.rolling(4, min_periods=3).sum(),
    "mom": ret_w.shift(4).rolling(48, min_periods=40).sum(),
    "rel": lrv.sub(lrv.mean(axis=1), axis=0),
    "vix": vix,
    "dvix": vix.diff(),
}
cum = ret_w.fillna(0).cumsum()
feats["dd"] = cum - cum.rolling(52, min_periods=40).max()
FE = list(feats)


def long(wide):
    s = wide.unstack()
    s.index = s.index.set_names(["country", "week"])
    return s.swaplevel().sort_index()


X = pd.concat({k: long(v) for k, v in feats.items()}, axis=1)
X["y"] = long(lrv.shift(-1))
ok = X[FE].notna().all(axis=1)
lab = X[ok & X["y"].notna()]
latest_week = X.index.get_level_values("week").max()
today = X[ok & (X.index.get_level_values("week") == latest_week)]
weeks = lab.index.get_level_values("week")
W = np.array(sorted(weeks.unique()))
print(f"{len(lab)} labelled country-weeks, {len(W)} weeks, {lab.index.get_level_values('country').nunique()} countries", flush=True)

HAR = ["l1", "l4", "l13", "l52"]
ML = ["ridge", "extra_trees", "hist_gbm"]


def fit_predict(trn, tst):
    out = pd.DataFrame(index=tst.index)
    out["persistence"] = tst["l1"]
    out["mean13"] = tst["l13"]
    out["ewma"] = tst["ew"]
    out["har"] = LinearRegression().fit(trn[HAR], trn["y"]).predict(tst[HAR])
    sc = StandardScaler().fit(trn[FE])
    out["ridge"] = Ridge(alpha=10.0).fit(sc.transform(trn[FE]), trn["y"]).predict(sc.transform(tst[FE]))
    out["extra_trees"] = ExtraTreesRegressor(n_estimators=200, max_depth=6, min_samples_leaf=20, n_jobs=-1, random_state=SEED).fit(trn[FE], trn["y"]).predict(tst[FE])
    out["hist_gbm"] = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40, random_state=SEED).fit(trn[FE], trn["y"]).predict(tst[FE])
    out["ensemble"] = out[["har"] + ML].mean(axis=1)
    return out


parts = []
i = FIRSTW
while i < len(W):
    j = min(i + REFIT, len(W))
    trn = lab[weeks < W[i - 1]]
    tst = lab[(weeks >= W[i]) & (weeks <= W[j - 1])]
    parts.append(fit_predict(trn, tst))
    print(f"  block {i}/{len(W)} done", flush=True)
    i = j
P = pd.concat(parts)
act = lab.loc[P.index, "y"]
names = list(P.columns)

A = act.unstack("country")
Lw = lab.loc[P.index, "lev"].unstack("country")
fc, rk = [], []
for k in names:
    err = P[k] - act
    base = ((P["persistence"] - act) ** 2).mean()
    fc.append({"model": k, "rmse_log_vol": float(np.sqrt((err ** 2).mean())), "mae_log_vol": float(err.abs().mean()), "skill_vs_persistence": float(1 - (err ** 2).mean() / base)})
    Pm = P[k].unstack("country")
    ic = Pm.rank(axis=1).corrwith(A.rank(axis=1), axis=1).dropna()
    icd = (Pm - Lw).rank(axis=1).corrwith((A - Lw).rank(axis=1), axis=1).dropna()
    hits, rnd = [], []
    for w in A.index:
        a = A.loc[w].dropna()
        p = Pm.loc[w].reindex(a.index)
        if len(a) >= 6:
            hits.append(len(set(a.nlargest(3).index) & set(p.nlargest(3).index)) / 3)
            rnd.append(3 / len(a))
    rk.append({
        "model": k,
        "rank_ic": float(ic.mean()), "rank_ic_t": float(ic.mean() / (ic.std() / np.sqrt(len(ic)))),
        "rank_ic_vs_own_norm": float(icd.mean()), "rank_ic_vs_own_norm_t": float(icd.mean() / (icd.std() / np.sqrt(len(icd)))),
        "top3_hit": float(np.mean(hits)), "top3_hit_random": float(np.mean(rnd)),
    })
fc = pd.DataFrame(fc).sort_values("rmse_log_vol")
rk = pd.DataFrame(rk).sort_values("rank_ic_vs_own_norm", ascending=False)
fc.to_csv(f"{OUT}/gr_forecast_metrics.csv", index=False)
rk.to_csv(f"{OUT}/gr_rank_metrics.csv", index=False)

simple = np.expm1(ret_w.shift(-1))
tw = sorted(set(P.index.get_level_values("week")))
mom_w = feats["mom"]
strat = {}
for nm, wfun in {
    "equal_weight": lambda w, c: pd.Series(1.0, index=c),
    "invvol_trailing": lambda w, c: 1 / np.exp(feats["l13"].loc[w, c]),
    "invvol_har": lambda w, c: 1 / np.exp(P["har"].loc[w][c]),
    "invvol_ensemble": lambda w, c: 1 / np.exp(P["ensemble"].loc[w][c]),
    "momentum_top3": lambda w, c: (mom_w.loc[w, c].rank(ascending=False) <= 3).astype(float),
}.items():
    prev, rets, turn = None, [], []
    for w in tw:
        c = P.loc[w].index
        c = c[simple.loc[w, c].notna() & mom_w.loc[w, c].notna()]
        if len(c) < 6:
            continue
        wt = wfun(w, c)
        wt = wt / wt.sum()
        rets.append((w, float((wt * simple.loc[w, c]).sum())))
        turn.append(float((wt.sub(prev, fill_value=0).abs().sum()) if prev is not None else wt.abs().sum()))
        prev = wt
    r = pd.Series(dict(rets))
    t_ = pd.Series(turn, index=r.index)
    strat[nm] = (r - COST * t_, t_)

al = []
curves = {}
for nm, (r, t_) in strat.items():
    eq = (1 + r).cumprod()
    curves[nm] = eq
    n = len(r)
    al.append({
        "strategy": nm,
        "ann_return": float(eq.iloc[-1] ** (52 / n) - 1),
        "ann_vol": float(r.std() * np.sqrt(52)),
        "sharpe": float(r.mean() / r.std() * np.sqrt(52)),
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "worst_week": float(r.min()),
        "avg_weekly_turnover": float(t_.mean()),
    })
al = pd.DataFrame(al)
al.to_csv(f"{OUT}/gr_allocation.csv", index=False)

full = fit_predict(lab, today)
pred_vol = np.exp(full["ensemble"])
tab = pd.DataFrame({
    "pred_vol_pct_next_week": 100 * pred_vol,
    "trailing_4wk_vol_pct": 100 * np.exp(today["l4"]),
    "vs_own_norm": pred_vol / np.exp(today["lev"]),
    "risk_parity_weight_pct": 100 * (1 / pred_vol) / (1 / pred_vol).sum(),
    "momentum_12_1_pct": 100 * today["mom"],
})
tab.index = tab.index.get_level_values("country")
tab = tab.sort_values("pred_vol_pct_next_week", ascending=False)
tab.to_csv(f"{OUT}/gr_today.csv")

pd.set_option("display.width", 170)
fmt = lambda v: f"{v:.3f}"
print(f"\nTest period: {pd.Timestamp(tw[0]).date()} to {pd.Timestamp(tw[-1]).date()}, {len(tw)} weeks, {A.shape[1]} countries")
print("\n== Volatility forecast error (log annualized vol; lower is better) ==")
print(fc.to_string(index=False, float_format=fmt))
print("\n== Which countries will be most volatile next week? (cross-country ranking) ==")
print(rk.to_string(index=False, float_format=fmt))
print("\n== Risk-aware allocation backtest (weekly, cost applied, local currencies) ==")
print(al.to_string(index=False, float_format=fmt))
print(f"\n== Outlook for the week after {pd.Timestamp(latest_week).date()} ==")
print(tab.to_string(float_format=lambda v: f"{v:.1f}"))
print("\nNote: the volatility ranking is a forecast with measured skill (see tables above).")
print("Which market will earn the most is not forecast: return prediction showed no edge in earlier tests.")
print("The weights shown are a risk-based allocation, not a return prediction or advice.")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
t2 = tab.sort_values("pred_vol_pct_next_week")
ax[0].barh(t2.index, t2["trailing_4wk_vol_pct"], color="lightgray", label="trailing 4 weeks")
ax[0].barh(t2.index, t2["pred_vol_pct_next_week"], height=0.4, color="tab:red", label="forecast next week")
ax[0].set_xlabel("annualized volatility (%)")
ax[0].legend()
ax[0].set_title("Volatility outlook by country")
for nm, eq in curves.items():
    ax[1].plot(eq.index, eq.values, label=nm)
ax[1].set_title("Allocation backtest, net of costs")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/gr_overview.png", dpi=130)
plt.close()
print("\nsaved gr_* tables and gr_overview.png in results/")
