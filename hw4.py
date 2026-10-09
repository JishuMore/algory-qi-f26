from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

WINDOW = 60
SPLIT = pd.Timestamp("2011-09-10")
OUT = Path(__file__).resolve().parent / "hw4_output"


def download_data():
    """Q1: Return training and test closes with inclusive date boundaries."""
    import yfinance as yf
    # End is exclusive: omit today's possibly incomplete daily bar.
    today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    raw = yf.download("^SP500TR", start="1990-01-01", end=today,
                      interval="1d", auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo returned no ^SP500TR data. Retry later; do not substitute SPY.")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise ValueError("Expected one ticker in Close data.")
        close = close.iloc[:, 0]
    close = close.dropna().astype(float).sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    if close.index.has_duplicates or (close <= 0).any():
        raise ValueError("Duplicate dates or nonpositive prices in downloaded data.")
    close.name = "Close"
    train, test = close.loc[close.index <= SPLIT], close.loc[close.index >= SPLIT]
    if min(len(train), len(test)) <= WINDOW:
        raise ValueError("Not enough prices in both sets.")
    if train.index[0] > pd.Timestamp("1990-01-10"):
        raise ValueError("Downloaded history does not begin near January 1990.")
    return train, test


def always_hold(prices):
    return True


def always_cash(prices):
    return False


def random_strategy(seed=0):
    rng = np.random.default_rng(seed)
    return lambda prices: bool(rng.random() < 0.7)


def class_strategy(prices):
    # Five returns require six closes.
    return bool(prices[-1] / prices[-6] - 1 <= 0)


def po3_strategy(range_days=20, search_days=5, tight_range=None):
    """Long/cash daily range-sweep-recovery proxy, using only this window.

    Accumulation: reference range of closes before the recent search segment.
    Sweep: a later close below that reference range's low.
    Confirmation: a subsequent close above the low and above its prior close.
    Hold after confirmation while above the low and below the range high;
    otherwise cash. No claim that closing-price extremes are candle wicks.
    """
    def rule(prices):
        p = np.asarray(prices, dtype=float)
        recent = p[-search_days:]
        reference = p[-(range_days + search_days):-search_days]
        low, high = reference.min(), reference.max()
        if tight_range is not None and high / low - 1 > tight_range:
            return False
        swept = False
        confirmed = False
        for j, price in enumerate(recent):
            if price < low:
                swept = True
                confirmed = False
            elif swept and j > 0 and price > low and price > recent[j - 1]:
                confirmed = True
            if price >= high:
                # The opposite closing-price boundary has been reached.
                swept = False
                confirmed = False
        return bool(confirmed and low < recent[-1] < high)
    return rule


def backtest(data, strategy):
    p = np.asarray(data, dtype=float)
    if len(p) <= WINDOW or not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("Need more than 60 finite, positive closes.")
    positions = np.array([bool(strategy(p[i-WINDOW:i].copy()))
                          for i in range(WINDOW, len(p))])
    index_returns = p[WINDOW:] / p[WINDOW-1:-1] - 1
    strategy_returns = np.where(positions, index_returns, 0.0)
    dates = data.index[WINDOW-1:]
    equity = pd.DataFrame({
        "Strategy": np.r_[1.0, np.cumprod(1 + strategy_returns)],
        "Index": np.r_[1.0, np.cumprod(1 + index_returns)]}, index=dates)
    return {"equity": equity, "positions": positions,
            "returns": strategy_returns, "index_returns": index_returns}


def metrics(result, column="Strategy"):
    e = result["equity"][column]
    n = len(e) - 1
    return {"total": float(e.iloc[-1] - 1),
            "annual": float(e.iloc[-1] ** (252 / n) - 1),
            "drawdown": float(-(e / e.cummax() - 1).min()),
            "invested": float(np.mean(result["positions"]))}


def describe(result, label, emit):
    emit(label)
    for column in ("Strategy", "Index"):
        m = metrics(result, column)
        emit(f"  {column}: total {m['total']:.2%}; annualised {m['annual']:.2%}; "
             f"maximum drawdown {m['drawdown']:.2%}")
    emit(f"  Days invested: {metrics(result)['invested']:.2%}")


def plot_result(result, title, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    result["equity"].plot(ax=ax, linewidth=1.5)
    ax.set(title=title, ylabel="Portfolio value (starts at 1)", xlabel="Date")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    return fig


def text_pages(pdf, title, content):
    lines = []
    for paragraph in content.splitlines():
        lines.extend(textwrap.wrap(paragraph, width=95) or [""])
    for start in range(0, len(lines), 48):
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.07, 0.95, title, fontsize=15, weight="bold", va="top")
        fig.text(0.07, 0.90, "\n".join(lines[start:start+48]),
                 fontsize=9, va="top", linespacing=1.5)
        pdf.savefig(fig)
        plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    log = []
    def emit(message=""):
        print(message)
        log.append(message)

    train, test = download_data()
    full = pd.concat([train, test]).loc[lambda s: ~s.index.duplicated()].sort_index()
    full.to_csv(OUT / "downloaded_closes.csv")
    emit("HW4 - Jay Jiang | ^SP500TR | annualisation: 252 trading days")
    emit("Q1: Training/test split")
    for name, data in [("Training", train), ("Test", test)]:
        emit(f"{name}: {len(data):,} closes; mean {data.mean():.4f}; "
             f"{data.index[0].date()} through {data.index[-1].date()}")
    emit("2011-09-10 is a Saturday: inclusive slicing creates no shared observation.")
    emit("Each set has its own 60-close warm-up; only later returns are evaluated.")

    hold = backtest(train, always_hold)
    cash = backtest(train, always_cash)
    random = backtest(train, random_strategy(0))
    np.testing.assert_allclose(hold["equity"]["Strategy"], hold["equity"]["Index"], rtol=0, atol=0)
    assert metrics(cash)["total"] == 0.0
    assert metrics(cash)["annual"] == 0.0
    emit("\nQ2-Q4: Always-hold matches index exactly; always-cash is exactly 0%.")
    describe(hold, "Always hold (training)", emit)
    describe(cash, "Always cash (training)", emit)
    describe(random, "Random 70% hold, fixed seed 0 (training)", emit)
    r, b = metrics(random)["total"], metrics(hold)["total"]
    emit(f"Random result between cash and index: {min(0, b) <= r <= max(0, b)}")
    emit("That ordering is a diagnostic, not a mathematical guarantee for random exposure.")
    full_hold = backtest(full, always_hold)
    emit(f"Full-period index annualised return: {metrics(full_hold)['annual']:.2%}")

    class_result = backtest(train, class_strategy)
    emit("\nQ5: Hold unless the previous five-trading-day return is positive.")
    describe(class_result, "Class strategy (training)", emit)

    candidates = [
        ("PO3: 15-close range, 3-close sweep/recovery search", po3_strategy(15, 3), True),
        ("PO3: 20-close range, 5-close sweep/recovery search", po3_strategy(20, 5), True),
        ("PO3: 30-close range, 5-close sweep/recovery search", po3_strategy(30, 5), True),
        ("PO3: 20-close range, 5-close search, range width <= 5%", po3_strategy(20, 5, .05), True),
        ("Trend: latest close above 20-close moving average", lambda p: bool(p[-1] > np.mean(p[-20:])), False),
        ("Momentum: positive previous 15-day return", lambda p: bool(p[-1] > p[-16]), False),
        ("Mean reversion: latest close below 20-close moving average", lambda p: bool(p[-1] < np.mean(p[-20:])), False),
        ("Slope: positive least-squares slope across 60 closes", lambda p: bool(np.polyfit(np.arange(60), p, 1)[0] > 0), False),
    ]
    emit("\nQ6: Every attempted variant (training only)")
    records, eligible = [], []
    for name, rule, is_po3 in candidates:
        result = backtest(train, rule)
        m = metrics(result)
        emit(f"{name}: annualised {m['annual']:.2%}; invested {m['invested']:.2%}")
        records.append({"rule": name, **m, "eligible_for_final": is_po3})
        if is_po3:
            eligible.append((m["annual"], name, rule, result))
    pd.DataFrame(records).to_csv(OUT / "variants.csv", index=False)
    _, final_name, final_rule, final_train = max(eligible, key=lambda item: item[0])
    rationale = (
        "This daily PO3-inspired rule looks for a close below an earlier closing-price "
        "range, followed by a recovery above its lower boundary with an upward daily close. "
        "It holds the index while that confirmed recovery remains inside the range, "
        "using the upper boundary as a closing-price proxy for opposing liquidity. "
        "The hypothesis is that a failed downside break may precede a rebound; this "
        "hypothesis remains unproven and does not establish the intraday PO3 claims.")
    emit("\nQ7: FINAL RULE LOCKED BEFORE TEST")
    emit(final_name)
    emit("Selected by highest training annualised return among FOUR PO3 proxies; "
         "eight total variants tried, including four alternative ideas.")
    emit(rationale)
    describe(final_train, "Final strategy (training)", emit)

    # The ONLY evaluation of the selected final rule on the test set.
    final_test = backtest(test, final_rule)
    emit("\nQ8: Final strategy, test set - one evaluation, no retuning")
    describe(final_test, "Held-out test", emit)
    final_test["equity"].to_csv(OUT / "test_equity.csv")
    pd.DataFrame({"hold": final_test["positions"],
                  "strategy_return": final_test["returns"],
                  "index_return": final_test["index_returns"]},
                 index=test.index[WINDOW:]).to_csv(OUT / "test_daily_returns.csv")
    emit("\nQ9: Latest completed-close signal")
    emit(f"{'HOLD' if final_rule(full.iloc[-WINDOW:].to_numpy()) else 'CASH'}; "
         f"last close used: {full.index[-1].date()}")
    emit("This is the next-session signal using completed daily closes, not a 10AM signal.")

    tr, te, ix = metrics(final_train), metrics(final_test), metrics(final_test, "Index")
    outcome = "beat" if te["annual"] > ix["annual"] else "lost to" if te["annual"] < ix["annual"] else "matched"
    reflection = (
        f"My daily PO3-inspired strategy returned {tr['annual']:.2%} annualised in "
        f"training and {te['annual']:.2%} in the test period, a change of "
        f"{(te['annual']-tr['annual'])*100:+.2f} percentage points. In the test set, "
        f"it {outcome} buy-and-hold, which returned {ix['annual']:.2%} annualised. "
        f"The strategy's maximum drawdown was {te['drawdown']:.2%}, compared with "
        f"{ix['drawdown']:.2%} for the index; it was invested {te['invested']:.2%} "
        "of evaluated days.\n\n"
        "I tested eight variants and selected the best training return from four "
        "PO3 proxies before evaluating the final rule once on the test set. "
        "Selection creates a risk of fitting noise, and holding cash can miss "
        "large gains. This experiment tests a daily range-sweep-recovery idea, "
        "not the actual 10AM method: closes do not reveal intraday wicks, "
        "fair value gaps, entry fills, stops, or session targets.\n\n"
        "I would not commit my own money based on this result alone. I would "
        "need an independently reserved future period, plausible execution "
        "after each signal, transaction costs, and stability across nearby "
        "parameters. Testing the original method would additionally require "
        "intraday OHLC data and explicit entry/exit definitions.\n\n"
        "PERSONAL RESPONSE TO COMPLETE: Was the result surprising to you, "
        "and why? Replace this line with your own reaction before submitting.")
    emit("\nQ10: Data-grounded reflection draft (add your personal reaction)")
    emit(reflection)
    (OUT / "reflection.txt").write_text(reflection, encoding="utf-8")
    emit("\nAssumptions: no fees, slippage, taxes or cash interest; one unit of long "
         "exposure or cash; close-to-close execution; no shorts or intraday stops.")
    emit("Rerunning after inspecting the test or changing the rule contaminates the holdout.")
    figures = [plot_result(hold, "Q3: Always hold vs index - training", "q3_always_hold.png"),
               plot_result(class_result, "Q5: Class rule vs index - training", "q5_class_rule.png"),
               plot_result(final_train, "Q7: Final daily PO3 proxy - training", "q7_training.png"),
               plot_result(final_test, "Q8: Final daily PO3 proxy - test", "q8_test.png")]
    (OUT / "results.txt").write_text("\n".join(log), encoding="utf-8")
    with PdfPages(OUT / "report.pdf") as pdf:
        text_pages(pdf, "HW4 - Results and method | Jay Jiang", "\n".join(log[:log.index("\nQ10: Data-grounded reflection draft (add your personal reaction)")]))
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)
        text_pages(pdf, "Q10 - Reflection draft (edit before submission)", reflection)
    (OUT / "README.md").write_text(
        "# Algory QI HW4\n\nRun `python hw4.py` after installing numpy, pandas, "
        "matplotlib and yfinance. Downloads ^SP500TR daily closes since 1990, "
        "splits at 2011-09-10, validates the backtester and logs eight variants. "
        "Selects among four daily PO3 proxies on training data, then evaluates "
        "the final strategy once on held-out data. Every signal sees exactly "
        "60 previous closes. This does not test the intraday 10AM PO3 method. "
        "Output includes plots, metrics, data CSVs and a PDF. Complete the "
        "personal reaction in Q10 before submission. No costs or cash yield "
        "are modelled. Do not retune after inspecting the test.\n\n"
        "Data API: https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html\n",
        encoding="utf-8")
    print(f"\nSaved results and plots to: {OUT}")


if __name__ == "__main__":
    main()
