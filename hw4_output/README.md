# Algory QI HW4

Run `python hw4.py` after installing numpy, pandas, matplotlib and yfinance. Downloads ^SP500TR daily closes since 1990, splits at 2011-09-10, validates the backtester and logs eight variants. Selects among four daily PO3 proxies on training data, then evaluates the final strategy once on held-out data. Every signal sees exactly 60 previous closes. This does not test the intraday 10AM PO3 method. Output includes plots, metrics, data CSVs and a PDF. Complete the personal reaction in Q10 before submission. No costs or cash yield are modelled. Do not retune after inspecting the test.

Data API: https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
