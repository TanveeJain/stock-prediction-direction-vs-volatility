import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import yfinance as yf
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from features import FEATS, build_features, get

warnings.filterwarnings("ignore")

US = (
    "AAPL,MSFT,GOOGL,AMZN,JPM,BAC,WFC,C,JNJ,PFE,MRK,UNH,XOM,CVX,WMT,HD,MCD,DIS,PG,KO,PEP,"
    "INTC,CSCO,ORCL,IBM,T,VZ,BA,CAT,GE,MMM,NKE"
).split(",")
INDIA = (
    "RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS,ICICIBANK.NS,HINDUNILVR.NS,ITC.NS,SBIN.NS,"
    "BHARTIARTL.NS,KOTAKBANK.NS,LT.NS,AXISBANK.NS,ASIANPAINT.NS,MARUTI.NS,SUNPHARMA.NS,"
    "TITAN.NS,ULTRACEMCO.NS,NESTLEIND.NS,WIPRO.NS,HCLTECH.NS,BAJFINANCE.NS,ONGC.NS,NTPC.NS,"
    "POWERGRID.NS,TATASTEEL.NS,COALINDIA.NS,TECHM.NS,DRREDDY.NS,CIPLA.NS,HEROMOTOCO.NS"
).split(",")
MARKETS = {
    "us": (US, "^GSPC", "^VIX"),
    "india": (INDIA, "^NSEI", "^INDIAVIX"),
}


def arg(name, default):
    for a in sys.argv:
        if a.startswith(f"--{name}="):
            return a.split("=")[1]
    return default


START = arg("start", "2008-01-01")
MINTRAIN = int(arg("mintrain", 756))
K = int(arg("shifts", 300))
SEED = int(arg("seed", 0))
USE = arg("markets", "us,india").split(",")
OUT = "results"
os.makedirs(OUT, exist_ok=True)
rs = np.random.RandomState(SEED)


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
        if len(d) > MINTRAIN + 500:
            out[t] = d
    return out


def auc_perm(s, y, minshift):
    n = len(y)
    idx1 = np.flatnonzero(y == 1)
    n1 = len(idx1)
    n0 = n - n1
    if n1 == 0 or n0 == 0 or n < 2 * minshift + 20:
        return np.nan, np.nan
    r = rankdata(s)
    base = n1 * (n1 + 1) / 2
    obs = (r[idx1].sum() - base) / (n1 * n0)
    ks = rs.randint(minshift, n - minshift, size=K)
    null = (r[(idx1[None, :] - ks[:, None]) % n].sum(axis=1) - base) / (n1 * n0)
    sd = null.std()
    if sd <= 0:
        return float(obs), np.nan
    return float(obs), float(norm.sf((obs - null.mean()) / sd))


def auc_only(s, y):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bh(p):
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qq = np.empty(len(pv))
    qq[order] = np.minimum(ranked, 1.0)
    q[ok] = qq
    return q


def walk_forward(z, ycol):
    z = z[FEATS + [ycol, "hi"]].dropna()
    X = z[FEATS].values
    y = z[ycol].values.astype(int)
    years = z.index.year.values
    rows = []
    for Y in np.unique(years):
        pos = np.flatnonzero(years == Y)
        i, j = pos[0], pos[-1] + 1
        if i < MINTRAIN or j - i < 100:
            continue
        sc = StandardScaler().fit(X[:i - 1])
        xa = np.clip(sc.transform(X[:i - 1]), -5, 5)
        xb = np.clip(sc.transform(X[i:j]), -5, 5)
        m = LogisticRegression(C=0.1, max_iter=500).fit(xa, y[:i - 1])
        p = m.predict_proba(xb)[:, 1]
        major = int(y[:i - 1].mean() >= 0.5)
        rows.append(pd.DataFrame({
            "date": z.index[i:j], "year": Y, "s": p, "y": y[i:j],
            "hi": z["hi"].values[i:j], "major": major,
        }))
    return pd.concat(rows) if rows else None


cells, stocks, regime = [], [], []
oos = {}
for mk in USE:
    tickers, idx_sym, vix_sym = MARKETS[mk]
    print(f"[{mk}] downloading", flush=True)
    sp = get(idx_sym, START)
    vx = get(vix_sym, START)
    data = load_panel(tickers, START)
    print(f"[{mk}] {len(data)} stocks", flush=True)
    for t, d in data.items():
        f = build_features(d, sp, vx)
        f["hi"] = (f["vxl"] > f["vxl"].rolling(252, min_periods=126).median()).astype(float)
        f.loc[f["vxl"].rolling(252, min_periods=126).median().isna(), "hi"] = np.nan
        for target, ycol in [("dir", "y_dir"), ("vol", "y_vol")]:
            w = walk_forward(f, ycol)
            if w is None:
                continue
            for Y, g in w.groupby("year"):
                a, p = auc_perm(g["s"].values, g["y"].values, 20)
                cells.append({
                    "market": mk, "ticker": t, "target": target, "year": int(Y), "n": len(g),
                    "acc": float(np.mean((g["s"].values > 0.5).astype(int) == g["y"].values)),
                    "base_acc": float(np.mean(g["y"].values == g["major"].values)),
                    "auc": a, "p": p,
                })
            a, p = auc_perm(w["s"].values, w["y"].values, 60)
            stocks.append({"market": mk, "ticker": t, "target": target, "n": len(w), "auc": a, "p": p})
            for h, name in [(1.0, "high_vix"), (0.0, "low_vix")]:
                g = w[w["hi"] == h]
                if len(g) > 300:
                    regime.append({"market": mk, "ticker": t, "target": target, "regime": name, "auc": auc_only(g["s"].values, g["y"].values)})
        print(f"  {t} done", flush=True)

cells = pd.DataFrame(cells)
stocks = pd.DataFrame(stocks)
regime = pd.DataFrame(regime)
cells["q"] = np.nan
for tg in ["dir", "vol"]:
    m = cells["target"] == tg
    cells.loc[m, "q"] = bh(cells.loc[m, "p"].values)
cells["significant"] = cells["q"] < 0.05
stocks["q"] = np.nan
for tg in ["dir", "vol"]:
    m = stocks["target"] == tg
    stocks.loc[m, "q"] = bh(stocks.loc[m, "p"].values)
stocks["significant"] = stocks["q"] < 0.05
cells.to_csv(f"{OUT}/pmap_cells.csv", index=False)
stocks.to_csv(f"{OUT}/pmap_stocks.csv", index=False)
regime.to_csv(f"{OUT}/pmap_regime.csv", index=False)

summary = []
for (mk, tg), g in cells.groupby(["market", "target"]):
    summary.append({
        "market": mk, "target": tg, "stock_years": len(g),
        "mean_auc": float(g["auc"].mean()),
        "share_auc_above_half": float((g["auc"] > 0.5).mean()),
        "raw_p_below_0.05": int((g["p"] < 0.05).sum()),
        "expected_by_chance": round(0.05 * len(g), 1),
        "fdr_discoveries": int(g["significant"].sum()),
    })
summary = pd.DataFrame(summary)

sb = []
for (mk, tg), g in stocks.groupby(["market", "target"]):
    sb.append({
        "market": mk, "target": tg, "stocks": len(g), "mean_stock_auc": float(g["auc"].mean()),
        "stocks_significant_fdr": int(g["significant"].sum()),
    })
sb = pd.DataFrame(sb)

diff = []
if {"us", "india"} <= set(stocks["market"]):
    for tg in ["dir", "vol"]:
        a = stocks[(stocks.market == "india") & (stocks.target == tg)]["auc"].dropna().values
        b = stocks[(stocks.market == "us") & (stocks.target == tg)]["auc"].dropna().values
        bs = [rs.choice(a, len(a)).mean() - rs.choice(b, len(b)).mean() for _ in range(5000)]
        diff.append({
            "target": tg, "india_minus_us_auc": float(a.mean() - b.mean()),
            "ci95_low": float(np.percentile(bs, 2.5)), "ci95_high": float(np.percentile(bs, 97.5)),
        })
diff = pd.DataFrame(diff)

rg = regime.groupby(["market", "target", "regime"])["auc"].agg(["mean", "std", "count"]).reset_index()
rg["se"] = rg["std"] / np.sqrt(rg["count"])

pd.set_option("display.width", 160)
fmt = lambda v: f"{v:.3f}"
print("\n== Stock-year cells (per-cell permutation p, BH false discovery rate 5%) ==")
print(summary.to_string(index=False, float_format=fmt))
print("\n== Whole-period AUC per stock ==")
print(sb.to_string(index=False, float_format=fmt))
if len(diff):
    print("\n== India minus US (bootstrap over stocks) ==")
    print(diff.to_string(index=False, float_format=fmt))
print("\n== AUC by VIX regime (mean over stocks) ==")
print(rg.to_string(index=False, float_format=fmt))
summary.to_csv(f"{OUT}/pmap_summary.csv", index=False)
rg.to_csv(f"{OUT}/pmap_regime_summary.csv", index=False)
if len(diff):
    diff.to_csv(f"{OUT}/pmap_india_vs_us.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for a, tg in zip(ax, ["dir", "vol"]):
    for mk, col in [("us", "tab:blue"), ("india", "tab:orange")]:
        g = cells[(cells.market == mk) & (cells.target == tg)]
        if g.empty:
            continue
        by = g.groupby("year")["auc"].agg(["mean", "std", "count"])
        se = by["std"] / np.sqrt(by["count"])
        a.plot(by.index, by["mean"], label=mk, color=col)
        a.fill_between(by.index, by["mean"] - se, by["mean"] + se, color=col, alpha=0.2)
    a.axhline(0.5, color="black", linewidth=0.8)
    a.set_title("next-day direction" if tg == "dir" else "next-day volatility")
    a.set_xlabel("test year")
    a.xaxis.set_major_locator(MaxNLocator(integer=True))
    a.legend()
ax[0].set_ylabel("mean AUC across stocks (0.5 = no skill)")
plt.tight_layout()
plt.savefig(f"{OUT}/pmap_auc_by_year.png", dpi=130)
plt.close()

mks = [m for m in USE if m in set(cells["market"])]
fig, ax = plt.subplots(len(mks), 2, figsize=(14, 5 * len(mks)), squeeze=False)
for r_, mk in enumerate(mks):
    for c_, tg in enumerate(["dir", "vol"]):
        g = cells[(cells.market == mk) & (cells.target == tg)]
        piv = g.pivot(index="ticker", columns="year", values="auc")
        sig = g.pivot(index="ticker", columns="year", values="significant").reindex_like(piv).fillna(False).astype(bool)
        im = ax[r_][c_].imshow(piv.values, cmap="RdBu_r", vmin=0.4, vmax=0.6, aspect="auto")
        ax[r_][c_].set_xticks(range(len(piv.columns)))
        ax[r_][c_].set_xticklabels(piv.columns, rotation=90, fontsize=7)
        ax[r_][c_].set_yticks(range(len(piv.index)))
        ax[r_][c_].set_yticklabels(piv.index, fontsize=7)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                if sig.values[i, j]:
                    ax[r_][c_].text(j, i, "*", ha="center", va="center", fontsize=8)
        ax[r_][c_].set_title(f"{mk}: {'direction' if tg == 'dir' else 'volatility'} AUC (* = significant after FDR)")
        plt.colorbar(im, ax=ax[r_][c_], fraction=0.04)
plt.tight_layout()
plt.savefig(f"{OUT}/pmap_heatmap.png", dpi=130)
plt.close()
print("\nsaved tables and charts in results/ (pmap_*)")
