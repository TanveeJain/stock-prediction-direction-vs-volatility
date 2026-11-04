
import sys
 
TICKER = "GOOGL"
START = "2025-09-01"
WINDOW = 30
SEED = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--seed=")), 42)
YS = 100.0
 
NEWS_CSV = "news.csv"
SCORED_CSV = "news_scored.csv"
OUT = "results"
 
PRICE_COLS = ["ret", "hl", "vol", "m5", "m20"]
NEWS_COLS = ["pos", "neg", "neu", "cnt"]
 
MODE = "price" if "--price-only" in sys.argv else "news"
COLS = PRICE_COLS + (NEWS_COLS if MODE == "news" else [])
