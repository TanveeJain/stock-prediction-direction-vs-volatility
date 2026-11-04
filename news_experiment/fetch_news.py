import os
import sys
import time
import requests
import pandas as pd

from config import TICKER, START, NEWS_CSV

key = os.environ["FINNHUB_API_KEY"]
end = pd.Timestamp.today().normalize()
cur = pd.Timestamp(START)
if "--full" not in sys.argv:
    cur = max(cur, end - pd.Timedelta(days=365))

rows = []
print("fetching from", cur.date(), "to", end.date(), flush=True)
while cur <= end:
    nxt = min(cur + pd.Timedelta(days=29), end)
    r = requests.get(
        "https://finnhub.io/api/v1/company-news",
        params={"symbol": TICKER, "from": cur.strftime("%Y-%m-%d"), "to": nxt.strftime("%Y-%m-%d"), "token": key},
        timeout=30,
    )
    r.raise_for_status()
    got = 0
    for item in r.json():
        if item.get("headline") and item.get("datetime"):
            rows.append((item["datetime"], item["headline"]))
            got += 1
    print(cur.date(), "to", nxt.date(), ":", got, "headlines", flush=True)
    cur = nxt + pd.Timedelta(days=1)
    time.sleep(1.1)

df = pd.DataFrame(rows, columns=["ts", "headline"])
df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
df = df[["date", "headline"]].drop_duplicates().sort_values("date")

if os.path.exists(NEWS_CSV):
    old = pd.read_csv(NEWS_CSV, parse_dates=["date"])
    df = pd.concat([old, df]).drop_duplicates().sort_values("date")

df.to_csv(NEWS_CSV, index=False)
print(len(df), "headlines saved to", NEWS_CSV)
