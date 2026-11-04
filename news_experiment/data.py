import os
import numpy as np
import pandas as pd
import yfinance as yf

from config import PRICE_COLS, NEWS_COLS


def load_prices(ticker, start):
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).normalize()
    return df


def score(texts, batch=64):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    bert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(dev).eval()
    names = {"positive": "pos", "negative": "neg", "neutral": "neu"}
    order = [names[bert.config.id2label[i].lower()] for i in range(3)]
    out = []
    for i in range(0, len(texts), batch):
        b = tok(texts[i:i + batch], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
        with torch.no_grad():
            out.append(torch.softmax(bert(**b).logits, -1).cpu().numpy())
    out = pd.DataFrame(np.concatenate(out), columns=order)
    return out[["pos", "neg", "neu"]].values


def get_scores(path, cache):
    news = pd.read_csv(path)
    news["date"] = pd.to_datetime(news["date"])
    news = news.dropna(subset=["headline"]).drop_duplicates(["date", "headline"])
    if os.path.exists(cache):
        old = pd.read_csv(cache, parse_dates=["date"])
        news = news.merge(old, on=["date", "headline"], how="left")
    else:
        for c in ["pos", "neg", "neu"]:
            news[c] = np.nan
    miss = news["pos"].isna()
    if miss.any():
        news.loc[miss, ["pos", "neg", "neu"]] = score(news.loc[miss, "headline"].tolist())
    news.to_csv(cache, index=False)
    return news


def daily_news(news, index):
    ts = news["date"]
    late = (ts.dt.hour >= 16).astype(int)
    day = ts.dt.normalize() + pd.to_timedelta(late, unit="D")
    k = index.searchsorted(day.values)
    ok = k < len(index)
    out = news[ok].copy()
    out["day"] = index[k[ok]]
    g = out.groupby("day").agg(pos=("pos", "mean"), neg=("neg", "mean"), neu=("neu", "mean"), cnt=("pos", "size"))
    g["cnt"] = np.log1p(g["cnt"])
    return g


def build(prices, daily, cols):
    df = prices.copy()
    if daily is not None:
        df = df.join(daily)
        df["pos"] = df["pos"].fillna(0)
        df["neg"] = df["neg"].fillna(0)
        df["neu"] = df["neu"].fillna(1)
        df["cnt"] = df["cnt"].fillna(0)
    df["ret"] = np.log(df["Close"]).diff()
    df["hl"] = (df["High"] - df["Low"]) / df["Close"]
    df["vol"] = np.log(df["Volume"].replace(0, np.nan)).diff()
    df["m5"] = df["Close"] / df["Close"].rolling(5).mean() - 1
    df["m20"] = df["Close"] / df["Close"].rolling(20).mean() - 1
    df["y"] = df["ret"].shift(-1)
    return df.dropna(subset=cols)


def make_windows(x, y, L):
    ends = np.arange(L - 1, len(x))
    X = np.stack([x[e - L + 1:e + 1] for e in ends]).astype(np.float32)
    return X, y[ends].astype(np.float32), ends
