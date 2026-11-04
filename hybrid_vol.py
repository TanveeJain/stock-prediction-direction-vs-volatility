import json
import os
import sys
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

from countries import COUNTRIES
from panel_data import load_weekly

warnings.filterwarnings("ignore")


def arg(name, default):
    for a in sys.argv:
        if a.startswith(f"--{name}="):
            return a.split("=")[1]
    return default


START = arg("start", "2006-01-01")
MINN = int(arg("minstocks", 5))
FIRSTW = int(arg("firstw", 312))
BLOCK = int(arg("block", 52))
EPOCHS = int(arg("epochs", 30))
L = int(arg("window", 26))
H = int(arg("hidden", 32))
SEED = int(arg("seed", 0))
USE = arg("countries", ",".join(COUNTRIES)).split(",")
OUT = "results"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

vol, ret_w, vix = load_weekly(USE, START, MINN)
lrv = np.log(vol)
cols = list(lrv.columns)
T, C = lrv.shape
base = lrv.rolling(13, min_periods=10).mean()
l1, l4 = lrv, lrv.rolling(4, min_periods=3).mean()
l13, l52 = base, lrv.rolling(52, min_periods=40).mean()
lev = lrv.expanding(min_periods=52).mean()
y = lrv.shift(-1)
rel = lrv.sub(lrv.mean(axis=1), axis=0)
vixc = pd.DataFrame({c: vix for c in cols})

raw = np.stack([lrv.values, ret_w.values, rel.values, vixc.values, vixc.diff().values], axis=-1)
M = lrv.notna().values
Y = y.values
B = base.values
ok = M & ~np.isnan(Y) & ~np.isnan(B)
har_cols = [l1.values, l4.values, l13.values, l52.values]
HAR = np.stack(har_cols, axis=-1)


class Net(nn.Module):
    def __init__(self, f, c, h):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(f, h, 3, padding=1), nn.GELU(), nn.Conv1d(h, h, 3, padding=1), nn.GELU())
        self.lstm = nn.LSTM(h, h, batch_first=True)
        self.emb = nn.Embedding(c, h)
        self.att = nn.MultiheadAttention(h, 4, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.Dropout(0.2), nn.Linear(h, 1))

    def forward(self, x, m):
        b, c, f, l = x.shape
        z = self.conv(x.reshape(b * c, f, l)).transpose(1, 2)
        o, _ = self.lstm(z)
        e = self.emb.weight[None].expand(b, -1, -1)
        h = o[:, -1].reshape(b, c, -1) + e
        a, _ = self.att(h, h, h, key_padding_mask=~m)
        return self.head(torch.cat([h, a, e], dim=-1)).squeeze(-1)


def windows(Xs):
    return sliding_window_view(Xs, L, axis=0)


def fit_block(i):
    last = i - 2
    tr = np.arange(L - 1, last + 1)
    tr = tr[ok[tr].any(axis=1)]
    mu = np.nanmean(raw[:last + 1].reshape(-1, raw.shape[-1]), axis=0)
    sd = np.nanstd(raw[:last + 1].reshape(-1, raw.shape[-1]), axis=0) + 1e-8
    Xs = np.nan_to_num((raw - mu) / sd)
    Xs = np.concatenate([Xs, M[..., None].astype(float)], axis=-1)
    W = windows(Xs)
    cut = int(len(tr) * 0.85)
    net = Net(Xs.shape[-1], C, H)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)

    def batch(idx):
        x = torch.tensor(W[idx - (L - 1)], dtype=torch.float32)
        m = torch.tensor(M[idx])
        t = torch.tensor(np.nan_to_num(Y[idx]), dtype=torch.float32)
        b = torch.tensor(np.nan_to_num(B[idx]), dtype=torch.float32)
        k = torch.tensor(ok[idx])
        return x, m, t, b, k

    def loss_of(idx):
        x, m, t, b, k = batch(idx)
        p = b + net(x, m)
        return (((p - t) ** 2) * k).sum() / k.sum()

    best, state, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        net.train()
        perm = np.random.permutation(tr[:cut])
        for s in range(0, len(perm), 32):
            opt.zero_grad()
            loss_of(perm[s:s + 32]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_of(tr[cut:]).item()
        if v < best:
            best, bad = v, 0
            state = {k_: t_.clone() for k_, t_ in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    net.load_state_dict(state)
    net.eval()
    trh = np.arange(0, last + 1)
    xs = HAR[trh].reshape(-1, 4)
    ys = Y[trh].reshape(-1)
    g = np.isfinite(xs).all(axis=1) & np.isfinite(ys)
    har = LinearRegression().fit(xs[g], ys[g])
    return net, W, har


rows = []
i = FIRSTW
while i < T - 1:
    j = min(i + BLOCK, T - 1)
    net, W, har = fit_block(i)
    te = np.arange(i, j)
    with torch.no_grad():
        x = torch.tensor(W[te - (L - 1)], dtype=torch.float32)
        d = net(x, torch.tensor(M[te])).numpy()
    for a, t in enumerate(te):
        for c in range(C):
            if ok[t, c] and np.isfinite(HAR[t, c]).all():
                rows.append((t, c, Y[t, c], B[t, c] + d[a, c], float(har.predict(HAR[t, c][None])[0]), l1.values[t, c], lev.values[t, c]))
    print(f"  block {i}/{T} done", flush=True)
    i = j

R = pd.DataFrame(rows, columns=["t", "c", "y", "hybrid", "har", "persistence", "lev"])
R["trailing"] = [B[int(t), int(c)] for t, c in zip(R["t"], R["c"])]
R["ensemble"] = (R["hybrid"] + R["har"]) / 2
names = ["persistence", "trailing", "har", "hybrid", "ensemble"]


def dm(e_a, e_b, lag=4):
    d = (e_a ** 2 - e_b ** 2).groupby(R["t"]).mean().values
    n = len(d)
    dc = d - d.mean()
    v = (dc @ dc) / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    z = d.mean() / np.sqrt(v / n)
    return float(z), float(norm.sf(z))


fc = []
for k in names:
    e = R[k] - R["y"]
    fv = np.exp(2 * R[k])
    rv = np.exp(2 * R["y"])
    q = (rv / fv - np.log(rv / fv) - 1).mean()
    up = np.sign(R[k] - R["persistence"])
    ac = np.sign(R["y"] - R["persistence"])
    g = (up != 0) & (ac != 0)
    hi = (R["y"] > R["lev"]).astype(int)
    fc.append({
        "model": k,
        "rmse_log_vol": float(np.sqrt((e ** 2).mean())),
        "qlike": float(q),
        "skill_vs_persistence": float(1 - (e ** 2).mean() / ((R["persistence"] - R["y"]) ** 2).mean()),
        "change_direction_acc": float((up[g] == ac[g]).mean()) if g.sum() and k != "persistence" else np.nan,
        "high_risk_auc": float(roc_auc_score(hi, R[k] - R["lev"])),
    })
fc = pd.DataFrame(fc)
fc.to_csv(f"{OUT}/hyb_metrics.csv", index=False)

pd.set_option("display.width", 160)
print(f"\nTest: {R['t'].nunique()} weeks, {C} countries, {len(R)} country-weeks")
print("\n== Volatility forecast quality (lower RMSE and QLIKE are better) ==")
print(fc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\n== Diebold-Mariano tests on squared error (positive z: first model is better) ==")
for a, b_ in [("hybrid", "har"), ("hybrid", "persistence"), ("ensemble", "har"), ("har", "persistence")]:
    z, p = dm(R[b_] - R["y"], R[a] - R["y"])
    print(f"  {a} vs {b_}: z = {z:.2f}, one-sided p = {p:.4f}")

bt = R.copy()
bt["wk"] = bt["t"].astype(int)
nxt = np.expm1(ret_w.shift(-1).values)
bt["ret"] = [nxt[int(t), int(c)] for t, c in zip(bt["t"], bt["c"])]
bt = bt[np.isfinite(bt["ret"])]
out = pd.DataFrame({
    "week": bt["wk"], "country": bt["c"].astype(int), "ret": bt["ret"],
    "p_trail": np.exp(bt["trailing"]), "p_har": np.exp(bt["har"]), "p_hyb": np.exp(bt["hybrid"]), "p_ens": np.exp(bt["ensemble"]),
})
out.sort_values(["week", "country"]).to_csv(f"{OUT}/bt_input.csv", index=False)
json.dump({"countries": cols, "weeks": [str(w.date()) for w in lrv.index]}, open(f"{OUT}/bt_meta.json", "w"))
print(f"\nwrote results/bt_input.csv ({len(out)} rows). Next: g++ -O2 -o backtest backtest.cpp && ./backtest 10 < results/bt_input.csv")
