# Stock prediction: benchmarking 12 models with walk-forward testing

Question: can machine learning predict (a) tomorrow's direction (up or down)
and (b) whether tomorrow's move will be large or small (volatility), for large
US stocks? Twelve models and an ensemble are compared honestly against a
majority-class baseline, on AAPL and MSFT, with about 4,160 out-of-sample test
days each (2010 to 2026).

## Models compared

Classical and tree-based (scikit-learn): logistic regression, SVM (RBF),
k-nearest neighbours, Gaussian naive Bayes, random forest, extra trees,
histogram gradient boosting, MLP.

Deep learning (PyTorch, 20-day windows): LSTM, GRU, 1D CNN, Transformer encoder.

With `--deep=all`, 15 more deep variants are added: bidirectional LSTM and GRU,
"2-path" LSTM, GRU and plain RNN (a long 20-day branch plus a short 5-day branch),
plain RNN and bidirectional RNN, encoder-decoder (seq2seq) LSTM and GRU, with and
without bidirectional encoders, variational (VAE) versions of the LSTM and GRU
encoder-decoders, and CNN and dilated-CNN encoder-decoders. The Transformer plays
the role of "attention is all you need". These are our own implementations of the
architecture types, and the decoder outputs one step because we predict one day ahead.
Use `--tag=deep` so the results do not overwrite the main run, and `--step=252
--epochs=10` to make it faster. More models means a stricter significance bar,
which the Bonferroni correction applies automatically.

Plus an ensemble that averages all model probabilities, and the baseline
(always predict the majority class of the training data).

XGBoost and LightGBM are supported by `benchmark.py` and run automatically when
installed. They were not available in the environment used for the results below.

## Method

- Data: daily prices from yfinance (AAPL and MSFT from 2005), plus the S&P 500 and VIX.
- Features: lagged returns, 10/20-day returns, 5/20/60-day volatility, lagged
  absolute returns, high-low range, volume change, moving-average gaps, RSI,
  market return, VIX change and level. All use data up to day t only.
- Targets: `dir` = next-day return > 0. `vol` = next-day absolute return is above
  the trailing 60-day median.
- Walk-forward: train on everything before a block, test on the next 126 trading
  days, move forward. Scaling is fitted on training data only, so every
  prediction is on data the model never saw.
- Metrics: accuracy, AUC, log loss, accuracy on confident days, and for `dir` a
  long-or-flat backtest with Sharpe ratio (no transaction costs).
- Significance: one-sided binomial test of each model against the baseline
  accuracy, Bonferroni-corrected for the number of models compared.

## Setup

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

On macOS, XGBoost and LightGBM need OpenMP: `brew install libomp`.

## Run

    python benchmark.py --ticker=AAPL --target=dir
    python benchmark.py --ticker=AAPL --target=vol
    python benchmark.py --ticker=MSFT --target=dir
    python benchmark.py --ticker=MSFT --target=vol
    python summarize.py

Options: `--step=252` (faster), `--epochs=15`, `--window=20`, `--skip-dl`,
`--only=logreg,lstm`, `--seed=0`. Outputs go to `results/`: per-run CSV and JSON,
an accuracy chart against the baseline band, a backtest chart for `dir`, and
`summary.md`.

### Cross-sectional ranking (stock selection)

Instead of predicting one stock's direction, rank 32 large US stocks each day
and test whether the top-ranked 20% beat the bottom-ranked 20%.

    python rank_benchmark.py --horizon=5
    python rank_benchmark.py --horizon=1
    python rank_benchmark.py --horizon=21

Models: ridge, lasso, random forest, extra trees, histogram gradient boosting,
MLP, PyTorch MLP, XGBoost and LightGBM (if installed), an ensemble, and simple
factor baselines (momentum, 5-day and 1-day reversal, low volatility) plus a
random-score control. Features are ranked within each day, the target is the
forward return rank, and walk-forward training leaves a gap of `horizon` days
before each test block so no training target overlaps the test period.

Metrics: daily rank information coefficient (IC) with its t-statistic, the
long-short return of top versus bottom 20%, Sharpe ratio before and after
costs (`--cost=5` basis points per side), and portfolio turnover. A strategy is
marked significant only if its IC t-statistic exceeds 3, because many
strategies are compared. Outputs: `results/rank_<h>d.csv`, `rank_ic_<h>d.png`
and `rank_curve_<h>d.png`.

Caveat: the stock list is today's large companies, so it has survivorship bias
and returns look better than a real portfolio would have been.

## Results

Walk-forward accuracy. The baseline row is the majority-class guess. The margin
of error is about +/-1.5 points.

| model | AAPL_dir | AAPL_vol | MSFT_dir | MSFT_vol |
|---|---|---|---|---|
| extra_trees | 0.532 | 0.557 | 0.515 | 0.554 |
| cnn1d | 0.528 | 0.550 | 0.510 | 0.550 |
| svm_rbf | 0.526 | 0.550 | 0.504 | 0.557 |
| ensemble | 0.523 | 0.558 | 0.512 | 0.565 |
| random_forest | 0.521 | 0.549 | 0.500 | 0.555 |
| transformer | 0.521 | 0.549 | 0.515 | 0.564 |
| lstm | 0.520 | 0.553 | 0.492 | 0.565 |
| mlp | 0.520 | 0.555 | 0.502 | 0.541 |
| naive_bayes | 0.519 | 0.544 | 0.515 | 0.531 |
| gru | 0.518 | 0.554 | 0.510 | 0.570 |
| hist_gbm | 0.515 | 0.549 | 0.497 | 0.551 |
| logreg | 0.514 | 0.561 | 0.510 | 0.570 |
| knn | 0.511 | 0.535 | 0.505 | 0.534 |
| baseline | 0.530 | 0.507 | 0.513 | 0.495 |

AUC (0.5 = no skill) is 0.49 to 0.51 for every model on `dir`, and 0.55 to 0.60
on `vol`.

### Findings

1. Next-day direction is not predictable. No model beats the baseline for AAPL
   or MSFT after the multiple-comparison correction, and AUC is at chance level
   for all of them. Differences between models in these columns are noise.
2. Volatility is predictable. Every model beats the baseline by roughly 3 to
   7 points on both stocks, with AUC around 0.57 to 0.60.
3. Complexity does not help. Logistic regression is at or near the top on both
   stocks (56.1% and 57.0%), matching the LSTM, GRU and Transformer, and the
   differences among the top models are within the margin of error.
4. The least flexible models (KNN, naive Bayes) are the weakest on volatility.

### Predictability map (US versus India, by year and regime)

    python predictability_map.py

Trains a logistic regression walk-forward on 32 US and 30 Indian (NSE) large
caps for each target, tests it on every calendar year, and asks where and when
the market is predictable. Each stock-year gets an AUC and a p-value from a
circular-shift null (the scores are shifted against the labels, which keeps the
serial dependence of volatility), with a Benjamini-Hochberg false discovery
rate across all cells. It also reports whole-period AUC per stock, the
India-minus-US difference with a bootstrap interval over stocks, and AUC in
high versus low VIX regimes (US VIX and India VIX). Outputs are `pmap_*` files
in `results/`, including a year-by-year AUC chart and a stock-by-year heatmap.

### Ranking results (5-day horizon, 32 stocks, 2010 to 2026)

794 non-overlapping 5-day periods, top versus bottom 20%, 5 bps cost per side.

| strategy | mean IC | IC t-stat | Sharpe gross | Sharpe net | turnover |
|---|---|---|---|---|---|
| f_momentum | 0.028 | 2.61 | 0.24 | 0.17 | 0.52 |
| hist_gbm | 0.017 | 2.15 | 0.27 | -0.08 | 2.27 |
| random_forest | 0.018 | 2.14 | 0.23 | -0.09 | 2.12 |
| mlp | 0.015 | 2.10 | 0.58 | 0.09 | 2.71 |
| extra_trees | 0.018 | 2.08 | 0.31 | 0.01 | 2.04 |
| ensemble | 0.017 | 2.06 | 0.25 | -0.07 | 2.19 |
| lightgbm | 0.016 | 2.06 | 0.26 | -0.13 | 2.40 |
| xgboost | 0.016 | 2.06 | 0.24 | -0.15 | 2.40 |
| lasso | 0.018 | 2.00 | 0.22 | -0.07 | 2.18 |
| random (control) | 0.012 | 1.84 | 0.68 | 0.04 | 3.24 |
| ridge | 0.016 | 1.82 | 0.21 | -0.10 | 2.20 |
| mlp_torch | 0.014 | 1.67 | 0.30 | -0.05 | 2.37 |
| f_reversal1 | 0.014 | 1.43 | 0.27 | -0.15 | 3.20 |
| f_reversal5 | 0.011 | 1.20 | 0.21 | -0.19 | 3.19 |
| f_lowvol | -0.006 | -0.54 | -0.06 | -0.10 | 0.43 |

Findings:

1. No strategy passes the |IC t| > 3 bar. The best is plain momentum
   (t = 2.6); the ML models cluster at t of about 2.0 to 2.1.
2. The random-score control scores IC t = 1.84 and the highest gross Sharpe of
   any row (0.68), so gross differences of this size are indistinguishable
   from luck.
3. Turnover of 200% to 320% per period removes most gross returns. Only
   momentum, with turnover of 0.52, keeps a positive net Sharpe, and it is small.
4. Cumulative net curves for the tree models fade from a small early gain to a
   loss, and momentum's gains are concentrated in 2024 to 2026.

## Limitations

- Part of the volatility signal is volatility persistence (quiet days follow
  quiet days). A simple rule such as "big move if the VIX is above its median"
  was not tested as an extra baseline.
- Two stocks (direction and volatility) or 32 stocks (ranking) over one time period; results may not transfer elsewhere. The ranking universe is today's large caps, so it has survivorship bias.
- Price-based features only. The news-sentiment scripts in this folder had too
  little history to test properly.
- The backtest ignores transaction costs and slippage, and its Sharpe
  differences are not evidence of skill because direction AUC is near 0.5.
