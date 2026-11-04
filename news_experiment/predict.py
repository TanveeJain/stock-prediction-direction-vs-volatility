import numpy as np
import torch

from config import TICKER, START, WINDOW, YS, NEWS_CSV, SCORED_CSV, OUT, MODE, COLS
from data import load_prices, get_scores, daily_news, build
from model import Net

ck = torch.load(f"{OUT}/{MODE}.pt")
mu, sd = np.array(ck["mu"]), np.array(ck["sd"])

prices = load_prices(TICKER, START)
daily = daily_news(get_scores(NEWS_CSV, SCORED_CSV), prices.index) if MODE == "news" else None
df = build(prices, daily, COLS)

x = (df[COLS].values[-WINDOW:] - mu) / sd
net = Net(len(COLS))
net.load_state_dict(ck["model"])
net.eval()
with torch.no_grad():
    r = net(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item() / YS

last = float(df["Close"].iloc[-1])
print(f"last date: {df.index[-1].date()}")
print(f"last close: {last:.2f}")
print(f"predicted return: {r * 100:.3f}%")
print(f"predicted next close: {last * np.exp(r):.2f}")
