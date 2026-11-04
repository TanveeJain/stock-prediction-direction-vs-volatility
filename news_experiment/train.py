
import json
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from config import TICKER, START, WINDOW, SEED, YS, NEWS_CSV, SCORED_CSV, OUT, MODE, COLS
from data import load_prices, get_scores, daily_news, build, make_windows
from model import Net
 
torch.manual_seed(SEED)
np.random.seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)
 
prices = load_prices(TICKER, START)
daily = daily_news(get_scores(NEWS_CSV, SCORED_CSV), prices.index) if MODE == "news" else None
df = build(prices, daily, COLS).dropna(subset=["y"])
 
n = len(df)
a, b = int(0.7 * n), int(0.85 * n)
x = df[COLS].values
mu, sd = x[:a].mean(0), x[:a].std(0) + 1e-8
x = (x - mu) / sd
y = df["y"].values
close = df["Close"].values
 
X, Y, ends = make_windows(x, y, WINDOW)
tr = ends < a
va = (ends >= a) & (ends < b)
te = ends >= b
 
t = lambda z: torch.tensor(z).to(dev)
Xtr, Ytr = t(X[tr]), t(Y[tr] * YS)
Xva, Yva = t(X[va]), t(Y[va] * YS)
Xte = t(X[te])
 
net = Net(len(COLS)).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
lossf = nn.HuberLoss(delta=1.0)
 
best, bad, bs = 1e9, 0, 64
path = f"{OUT}/{MODE}.pt"
for ep in range(200):
    net.train()
    perm = torch.randperm(len(Xtr), device=dev)
    tl = 0.0
    for i in range(0, len(perm), bs):
        idx = perm[i:i + bs]
        opt.zero_grad()
        loss = lossf(net(Xtr[idx]), Ytr[idx])
        loss.backward()
        opt.step()
        tl += loss.item() * len(idx)
    net.eval()
    with torch.no_grad():
        v = lossf(net(Xva), Yva).item()
    print(f"epoch {ep + 1} train {tl / len(perm):.4f} val {v:.4f}")
    if v < best:
        best, bad = v, 0
        torch.save({"model": net.state_dict(), "mu": mu.tolist(), "sd": sd.tolist(), "cols": COLS}, path)
    else:
        bad += 1
        if bad == 12:
            break
 
net.load_state_dict(torch.load(path)["model"])
net.eval()
with torch.no_grad():
    p = net(Xte).cpu().numpy() / YS
 
ce = ends[te]
c0 = close[ce]
real = c0 * np.exp(Y[te])
pred = c0 * np.exp(p)
 
rmse = lambda u, v: float(np.sqrt(np.mean((u - v) ** 2)))
mae = lambda u, v: float(np.mean(np.abs(u - v)))
res = {
    "mode": MODE,
    "test_days": int(te.sum()),
    "model_rmse": rmse(pred, real),
    "naive_rmse": rmse(c0, real),
    "model_mae": mae(pred, real),
    "naive_mae": mae(c0, real),
    "direction_acc": float(np.mean(np.sign(p) == np.sign(Y[te]))),
    "up_rate": float(np.mean(Y[te] > 0)),
    "seed": SEED,
}
print(json.dumps(res, indent=2))
json.dump(res, open(f"{OUT}/metrics_{MODE}_{SEED}.json", "w"), indent=2)
 
plt.figure(figsize=(12, 5))
plt.plot(df.index[ce], real, label="actual")
plt.plot(df.index[ce], pred, label="model")
plt.plot(df.index[ce], c0, label="naive", alpha=0.5)
plt.title(f"{TICKER} next-day close ({MODE})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/pred_{MODE}.png", dpi=120)
