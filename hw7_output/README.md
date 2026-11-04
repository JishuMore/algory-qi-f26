# Algory QI HW7

Run `python hw7.py` after installing numpy, pandas, matplotlib, scikit-learn and yfinance. It downloads 15 years of daily SPY (adjusted) and writes `results.txt`, `q4_tuning.png`, `q6_backtest.png` and `spy_prices.csv` here. Offline: `python hw7.py --csv prices.csv` (columns Date, Close, Volume).

Row t's features use closes and volumes up to day t's close; the label is close t to close t+1, so every feature is known the morning of t+1. `hand_check` recomputes one row by explicit slicing and again after deleting every later day, and asserts all three agree. The split is chronological 60/20/20. Tuning uses only the validation set, the chosen model is trained on the training set and scored once on test. Q6 reuses `backtest` and `metrics` from `hw4.py`, with no costs or cash yield.

Written answer to Q8: `written_answers.md`.
