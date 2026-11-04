import json
import os
import sys
import time
import warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import binomtest
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import make_dataset

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None


def arg(name, default):
    for a in sys.argv:
        if a.startswith(f"--{name}="):
            return a.split("=")[1]
    return default


TICKER = arg("ticker", "AAPL")
START = arg("start", "2005-01-01")
TARGET = arg("target", "dir")
FIRST = int(arg("first", 1260))
STEP = int(arg("step", 126))
WINDOW = int(arg("window", 20))
EPOCHS = int(arg("epochs", 15))
SEED = int(arg("seed", 0))
ONLY = [s for s in arg("only", "").split(",") if s]
SKIP_DL = "--skip-dl" in sys.argv
DEEP = arg("deep", "basic")
TAG = arg("tag", "")
OUT = "results"
os.makedirs(OUT, exist_ok=True)
np.random.seed(SEED)

tab = {
    "logreg": lambda: LogisticRegression(C=0.1, max_iter=1000),
    "svm_rbf": lambda: SVC(C=0.5, gamma="scale"),
    "knn": lambda: KNeighborsClassifier(n_neighbors=75),
    "naive_bayes": lambda: GaussianNB(),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=25, n_jobs=-1, random_state=SEED),
    "extra_trees": lambda: ExtraTreesClassifier(n_estimators=200, max_depth=6, min_samples_leaf=25, n_jobs=-1, random_state=SEED),
    "hist_gbm": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=SEED),
    "mlp": lambda: MLPClassifier((64, 32), alpha=1e-2, early_stopping=True, max_iter=300, random_state=SEED),
}
if XGBClassifier is not None:
    tab["xgboost"] = lambda: XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=4, random_state=SEED)
else:
    print("xgboost not available, skipping it")
if LGBMClassifier is not None:
    tab["lightgbm"] = lambda: LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=SEED)
else:
    print("lightgbm not available, skipping it")

EXTRA = [
    "lstm_bi", "lstm_2path", "gru_bi", "gru_2path", "rnn", "rnn_bi", "rnn_2path",
    "lstm_s2s", "lstm_bi_s2s", "lstm_vae", "gru_s2s", "gru_bi_s2s", "gru_vae",
    "cnn_s2s", "dcnn_s2s",
]
seq_kinds = ["lstm", "gru", "cnn1d", "transformer"] if (torch is not None and not SKIP_DL) else []
if DEEP == "all" and seq_kinds:
    seq_kinds = seq_kinds + EXTRA
if torch is None and not SKIP_DL:
    print("torch not available, skipping deep learning models")

names = list(tab) + seq_kinds
if ONLY:
    names = [k for k in names if k in ONLY]


def proba(m, x):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(x)[:, 1]
    return 1 / (1 + np.exp(-m.decision_function(x)))


if torch is not None:
    class SeqNet(nn.Module):
        def __init__(self, kind, d, h=32):
            super().__init__()
            self.kind = kind
            if kind == "lstm":
                self.rnn = nn.LSTM(d, h, batch_first=True)
            elif kind == "gru":
                self.rnn = nn.GRU(d, h, batch_first=True)
            elif kind == "cnn1d":
                self.conv = nn.Sequential(nn.Conv1d(d, h, 3, padding=1), nn.ReLU(), nn.Conv1d(h, h, 3, padding=1), nn.ReLU())
            else:
                self.inp = nn.Linear(d, h)
                self.pos = nn.Parameter(torch.randn(1, WINDOW, h) * 0.02)
                layer = nn.TransformerEncoderLayer(h, 4, 64, 0.2, batch_first=True)
                self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(h, 1))

        def forward(self, x):
            if self.kind in ("lstm", "gru"):
                o, _ = self.rnn(x)
                z = o[:, -1]
            elif self.kind == "cnn1d":
                z = self.conv(x.transpose(1, 2)).mean(2)
            else:
                z = self.enc(self.inp(x) + self.pos).mean(1)
            return self.head(z).squeeze(-1)

    SPEC = {
        "lstm_bi": ("lstm", True, "plain"), "lstm_2path": ("lstm", False, "path2"),
        "gru_bi": ("gru", True, "plain"), "gru_2path": ("gru", False, "path2"),
        "rnn": ("rnn", False, "plain"), "rnn_bi": ("rnn", True, "plain"), "rnn_2path": ("rnn", False, "path2"),
        "lstm_s2s": ("lstm", False, "s2s"), "lstm_bi_s2s": ("lstm", True, "s2s"), "lstm_vae": ("lstm", False, "vae"),
        "gru_s2s": ("gru", False, "s2s"), "gru_bi_s2s": ("gru", True, "s2s"), "gru_vae": ("gru", False, "vae"),
        "cnn_s2s": ("cnn", False, "cnn"), "dcnn_s2s": ("cnn", False, "dcnn"),
    }
    CELLS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

    class DeepNet(nn.Module):
        def __init__(self, kind, d, h=32):
            super().__init__()
            self.cell, self.bi, self.mode = SPEC[kind]
            self.h = h
            if self.mode in ("cnn", "dcnn"):
                dil = [1, 2, 4, 8] if self.mode == "dcnn" else [1, 1, 1, 1]
                layers, cin = [], d
                for k in dil:
                    layers += [nn.Conv1d(cin, h, 3, padding=k, dilation=k), nn.ReLU()]
                    cin = h
                self.conv = nn.Sequential(*layers)
                self.dec = nn.GRUCell(h, h)
                self.fin = h
            else:
                R = CELLS[self.cell]
                self.enc = R(d, h, batch_first=True, bidirectional=self.bi)
                oh = h * (2 if self.bi else 1)
                self.fin = oh
                if self.mode == "path2":
                    self.enc2 = R(d, h, batch_first=True, bidirectional=self.bi)
                    self.fin = oh * 2
                if self.mode in ("s2s", "vae"):
                    self.to_dec = nn.Linear(oh, h)
                    self.dec = R(h, h, batch_first=True)
                    self.fin = h
                if self.mode == "vae":
                    self.mu = nn.Linear(oh, h)
                    self.lv = nn.Linear(oh, h)
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.fin, 1))

        def last(self, o):
            if self.bi:
                return torch.cat([o[:, -1, :self.h], o[:, 0, self.h:]], dim=-1)
            return o[:, -1]

        def decode(self, ctx):
            h0 = ctx[None]
            x0 = torch.zeros(ctx.size(0), 1, ctx.size(1))
            if self.cell == "lstm":
                o, _ = self.dec(x0, (h0, torch.zeros_like(h0)))
            else:
                o, _ = self.dec(x0, h0)
            return o[:, -1]

        def forward(self, x):
            if self.mode in ("cnn", "dcnn"):
                ctx = self.conv(x.transpose(1, 2)).mean(2)
                return self.head(self.dec(torch.zeros_like(ctx), ctx)).squeeze(-1)
            o, _ = self.enc(x)
            z = self.last(o)
            if self.mode == "plain":
                return self.head(z).squeeze(-1)
            if self.mode == "path2":
                o2, _ = self.enc2(x[:, -5:])
                return self.head(torch.cat([z, self.last(o2)], dim=-1)).squeeze(-1)
            if self.mode == "s2s":
                return self.head(self.decode(torch.tanh(self.to_dec(z)))).squeeze(-1)
            mu, lv = self.mu(z), self.lv(z)
            ctx = mu + torch.exp(0.5 * lv) * torch.randn_like(mu) if self.training else mu
            kl = -0.5 * torch.mean(1 + lv - mu ** 2 - torch.exp(lv))
            return self.head(self.decode(torch.tanh(ctx))).squeeze(-1), kl

    def make_net(kind, d):
        return DeepNet(kind, d) if kind in SPEC else SeqNet(kind, d)

    def logits_of(out):
        return out[0] if isinstance(out, tuple) else out

    def fit_predict_seq(kind, xa, ya, xb):
        torch.manual_seed(SEED)
        net = make_net(kind, xa.shape[2])
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
        lossf = nn.BCEWithLogitsLoss()
        cut = int(len(xa) * 0.9)
        xt = torch.tensor(xa[:cut], dtype=torch.float32)
        yt = torch.tensor(ya[:cut], dtype=torch.float32)
        xv = torch.tensor(xa[cut:], dtype=torch.float32)
        yv = torch.tensor(ya[cut:], dtype=torch.float32)
        best, state = 1e9, None
        for ep in range(EPOCHS):
            net.train()
            perm = torch.randperm(len(xt))
            for s in range(0, len(perm), 128):
                idx = perm[s:s + 128]
                opt.zero_grad()
                out = net(xt[idx])
                loss = lossf(logits_of(out), yt[idx])
                if isinstance(out, tuple):
                    loss = loss + 1e-3 * out[1]
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = lossf(logits_of(net(xv)), yv).item()
            if v < best:
                best = v
                state = {k: t.clone() for k, t in net.state_dict().items()}
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            return torch.sigmoid(logits_of(net(torch.tensor(xb, dtype=torch.float32)))).numpy()


d = make_dataset(TICKER, START, TARGET)
cols = [c for c in d.columns if c not in ("y", "nxt")]
Xall = d[cols].values
Y = d["y"].values.astype(int)
R = d["nxt"].values
n = len(d)
L = WINDOW
FIRST = max(FIRST, L + 200)

preds = {k: [] for k in names}
failed = set()
secs = {k: 0.0 for k in names}
base, ys, rs = [], [], []

i = FIRST
while i < n:
    j = min(i + STEP, n)
    sc = StandardScaler().fit(Xall[:i])
    Xs = np.clip(sc.transform(Xall[:j]), -5, 5)
    win = sliding_window_view(Xs, (L, Xs.shape[1]))[:, 0]
    tr = np.arange(L - 1, i)
    te = np.arange(i, j)
    ya = Y[tr]
    base.append(np.full(len(te), int(ya.mean() >= 0.5)))
    ys.append(Y[te])
    rs.append(R[te])
    for k in names:
        if k in failed:
            continue
        t0 = time.time()
        try:
            if k in tab:
                m = tab[k]().fit(Xs[tr], ya)
                p = proba(m, Xs[te])
            else:
                p = fit_predict_seq(k, win[tr - (L - 1)], ya, win[te - (L - 1)])
            preds[k].append(p)
        except Exception as e:
            print(f"{k} failed: {e}")
            failed.add(k)
        secs[k] += time.time() - t0
    print(f"fold done: {i} / {n}", flush=True)
    i = j

names = [k for k in names if k not in failed]
yt = np.concatenate(ys)
bt = np.concatenate(base)
rt = np.concatenate(rs)
pt = {k: np.concatenate(preds[k]) for k in names}
pt["ensemble"] = np.mean([pt[k] for k in names], axis=0)
nt = len(yt)
base_acc = float(np.mean(bt == yt))
ci = float(1.96 * np.sqrt(base_acc * (1 - base_acc) / nt))
alpha = 0.05 / len(pt)

rows = []
for k, p in pt.items():
    hard = (p > 0.5).astype(int)
    right = hard == yt
    acc = float(right.mean())
    pv = float(binomtest(int(right.sum()), nt, base_acc, alternative="greater").pvalue)
    sure = np.abs(p - 0.5) > 0.05
    row = {
        "model": k,
        "acc": acc,
        "vs_baseline": acc - base_acc,
        "p_value": pv,
        "beats_baseline": bool(acc > base_acc and pv < alpha),
        "auc": float(roc_auc_score(yt, p)) if p.std() > 0 else 0.5,
        "log_loss": float(log_loss(yt, np.clip(p, 1e-6, 1 - 1e-6))),
        "confident_acc": float(right[sure].mean()) if sure.sum() else np.nan,
        "confident_share": float(sure.mean()),
        "minutes": round(secs.get(k, 0.0) / 60, 2),
    }
    if TARGET == "dir":
        s = (p > 0.5).astype(float) * rt
        row["strategy_sharpe"] = float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else 0.0
        row["strategy_ann_return"] = float(np.expm1(s.mean() * 252))
    rows.append(row)

res = pd.DataFrame(rows).sort_values("acc", ascending=False)
bh = {"model": "baseline", "acc": base_acc, "vs_baseline": 0.0}
if TARGET == "dir":
    bh["strategy_sharpe"] = float(rt.mean() / rt.std() * np.sqrt(252))
    bh["strategy_ann_return"] = float(np.expm1(rt.mean() * 252))
res = pd.concat([res, pd.DataFrame([bh])], ignore_index=True)

tag = f"{TICKER}_{TARGET}" + (f"_{TAG}" if TAG else "")
res.to_csv(f"{OUT}/benchmark_{tag}.csv", index=False)
pdf = pd.DataFrame(pt, index=d.index[FIRST:FIRST + nt])
pdf.insert(0, "base_pred", bt)
pdf.insert(0, "ret", rt)
pdf.insert(0, "y", yt)
pdf.index.name = "date"
pdf.to_csv(f"{OUT}/preds_{tag}.csv")

info = {
    "ticker": TICKER,
    "target": TARGET,
    "test_days": nt,
    "test_from": str(d.index[FIRST].date()),
    "test_to": str(d.index[-1].date()),
    "baseline_acc": base_acc,
    "ci95": ci,
    "bonferroni_alpha": alpha,
    "models_compared": len(pt),
}
json.dump(info, open(f"{OUT}/benchmark_{tag}.json", "w"), indent=2)

show = res.drop(columns=["log_loss", "minutes"], errors="ignore")
print(json.dumps(info, indent=2))
print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

order = res[res["model"] != "baseline"].sort_values("acc")
plt.figure(figsize=(9, 0.45 * len(order) + 1.5))
colors = ["tab:green" if b else "tab:gray" for b in order["beats_baseline"]]
plt.barh(order["model"], order["acc"], color=colors)
plt.axvline(base_acc, color="tab:red", linestyle="--", label=f"baseline {base_acc:.3f}")
plt.axvspan(base_acc - ci, base_acc + ci, color="tab:red", alpha=0.12)
lo = min(order["acc"].min(), base_acc - ci) - 0.01
hi = max(order["acc"].max(), base_acc + ci) + 0.01
plt.xlim(lo, hi)
plt.xlabel("walk-forward accuracy")
plt.title(f"{TICKER} {TARGET}: {nt} test days (green = beats baseline, Bonferroni-corrected)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/benchmark_{tag}.png", dpi=130)
plt.close()

if TARGET == "dir":
    idx = d.index[FIRST:FIRST + nt]
    plt.figure(figsize=(10, 5))
    plt.plot(idx, np.cumsum(rt), label="buy and hold", color="black")
    for k in order.sort_values("acc", ascending=False)["model"].head(3):
        plt.plot(idx, np.cumsum((pt[k] > 0.5) * rt), label=k)
    plt.title(f"{TICKER} cumulative log return, long when model says up, otherwise flat (no costs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/backtest_{tag}.png", dpi=130)
    plt.close()
