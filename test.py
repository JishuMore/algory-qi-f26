"""Homework 1 self-check. Does your environment actually work?

    python check_hw01.py

Checks your setup rather than your answers, because that is what Homework 1 is
really about. Self-contained: you only need this one file.
"""
import sys, pathlib

G, R, D, B, OFF = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
ok = True

print(f"\n{B}Homework 1 self-check: your environment{OFF}\n")

v = sys.version_info
good = v >= (3, 9)
print(f"  {G + 'pass' + OFF if good else R + 'FAIL' + OFF}  Python {v.major}.{v.minor}.{v.micro}")
if not good:
    print(f"        {D}You need Python 3.9 or newer.{OFF}")
    ok = False

for pkg, why in [("yfinance", "pulls market data"),
                 ("pandas", "holds the data"),
                 ("numpy", "does the arithmetic"),
                 ("matplotlib", "draws the plots"),
                 ("sklearn", "the models, from Session 7 onward")]:
    try:
        mod = __import__(pkg)
        print(f"  {G}pass{OFF}  {pkg:<12} {D}{getattr(mod, '__version__', '?'):<10} {why}{OFF}")
    except ImportError:
        install = "scikit-learn" if pkg == "sklearn" else pkg
        print(f"  {R}FAIL{OFF}  {pkg:<12} {D}not installed. Run: pip install {install}{OFF}")
        ok = False

try:
    import yfinance as yf
    hist = yf.Ticker("AAPL").history(period="10d")
    closes = hist["Close"].dropna() if len(hist) else hist
    if len(closes):
        # the most recent bar is often today's, still open, and comes back
        # empty. Drop it, or you will chase the same thing in your own code.
        print(f"  {G}pass{OFF}  market data   {D}pulled {len(hist)} days of AAPL, last complete "
              f"close {float(closes.iloc[-1]):.2f} on {closes.index[-1].date()}{OFF}")
    else:
        print(f"  {R}FAIL{OFF}  market data   {D}yfinance returned no usable prices. "
              f"Check your connection.{OFF}")
        ok = False
except Exception as e:
    print(f"  {R}FAIL{OFF}  market data   {D}{type(e).__name__}: {e}{OFF}")
    ok = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(); ax.plot([1, 2, 3])
    out = pathlib.Path(__file__).parent / "_plot_test.png"
    fig.savefig(out); plt.close(fig)
    size = out.stat().st_size; out.unlink()
    print(f"  {G}pass{OFF}  plotting      {D}wrote and removed a {size:,}-byte PNG{OFF}")
except Exception as e:
    print(f"  {R}FAIL{OFF}  plotting      {D}{type(e).__name__}: {e}{OFF}")
    ok = False

print()
if ok:
    print(f"{G}Your environment is ready.{OFF}")
    print(f"{D}Now do the rest of Homework 1: pick a ticker, pull a year, and look at it.{OFF}\n")
else:
    print(f"{R}Fix what failed above before Homework 2.{OFF}")
    print(f"{D}Come to office hours rather than losing an evening to an install problem.{OFF}\n")
sys.exit(0 if ok else 1)