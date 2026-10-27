
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import math
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm


def payoff(spot_at_expiry, strike, premium, kind="call", position="long"):
    """Profit per underlying share, including the option premium."""
    s = np.asarray(spot_at_expiry, dtype=float)
    if kind == "call":
        intrinsic = np.maximum(s - strike, 0.0)
    elif kind == "put":
        intrinsic = np.maximum(strike - s, 0.0)
    else:
        raise ValueError("kind must be 'call' or 'put'")
    if position not in ("long", "short"):
        raise ValueError("position must be 'long' or 'short'")
    return (intrinsic - premium) * (1 if position == "long" else -1)


def black_scholes(spot, strike, time, rate, volatility, kind="call"):
    """European Black-Scholes price; time in years, no dividends."""
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    if spot <= 0 or strike <= 0 or time < 0 or volatility < 0:
        raise ValueError("Positive spot/strike and nonnegative time/volatility required")
    if not all(math.isfinite(x) for x in (spot, strike, time, rate, volatility)):
        raise ValueError("All numerical inputs must be finite")
    if time == 0:
        return max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
    discounted_strike = strike * math.exp(-rate * time)
    if volatility == 0:
        return (max(spot - discounted_strike, 0.0) if kind == "call"
                else max(discounted_strike - spot, 0.0))
    d1 = (math.log(spot / strike) + (rate + volatility ** 2 / 2) * time) / (
        volatility * math.sqrt(time))
    d2 = d1 - volatility * math.sqrt(time)
    if kind == "call":
        return float(spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2))
    return float(discounted_strike * norm.cdf(-d2) - spot * norm.cdf(-d1))


def implied_volatility(market_price, spot, strike, time, rate, kind):
    """Invert Black-Scholes by bisection; return annualized decimal IV.

    Reject impossible prices and quotes outside the prescribed volatility
    bracket instead of silently returning one of the bracket endpoints.
    """
    if not math.isfinite(market_price) or market_price < 0 or time <= 0:
        raise ValueError("Finite nonnegative price and positive time required")
    lower = black_scholes(spot, strike, time, rate, 0.0, kind)
    upper = spot if kind == "call" else strike * math.exp(-rate * time)
    if market_price < lower - 1e-10 or market_price >= upper:
        raise ValueError("Quote violates European no-dividend price bounds")
    if abs(market_price - lower) < 1e-10:
        return 0.0
    lo, hi = 0.001, 5.0
    if not (black_scholes(spot, strike, time, rate, lo, kind) <= market_price <=
            black_scholes(spot, strike, time, rate, hi, kind)):
        raise ValueError("IV is outside the [0.001, 5.0] search bracket")
    for _ in range(200):
        mid = (lo + hi) / 2
        price = black_scholes(spot, strike, time, rate, mid, kind)
        if abs(price - market_price) < 1e-10 or hi - lo < 1e-10:
            return mid
        if price < market_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def combinations():
    # Illustrative premiums, not downloaded quotes. One contract = 100 shares.
    # Stock entry: $100; strikes: $95, $100 and $105.
    # Call(100) premium $6; call(105) $3; put(100) $5; put(95) $2.
    s = np.linspace(0, 160, 1601)
    positions = [
        ("Covered call", 100 * (s - 100 + payoff(s, 105, 3, "call", "short")),
         "$800", "$9,700", "$97", "100 shares at $100 + short 1 call, K=105, premium=3"),
        ("Protective put", 100 * (s - 100 + payoff(s, 95, 2, "put")),
         "Unlimited", "$700", "$102", "100 shares at $100 + long 1 put, K=95, premium=2"),
        ("Long straddle", 100 * (payoff(s, 100, 6, "call") + payoff(s, 100, 5, "put")),
         "Unlimited", "$1,100", "$89 and $111", "Long 1 call (premium=6) + 1 put (premium=5), K=100"),
        ("Bull call spread", 100 * (payoff(s, 100, 6, "call") + payoff(s, 105, 3, "call", "short")),
         "$200", "$300", "$103", "Long 1 call K=100 (premium=6); short 1 K=105 (premium=3)"),
    ]
    lines = ["Q1. COMBINATIONS", "All results are expiration PROFIT, including premiums.",
             "Illustrative inputs; no fees, interest on premiums, or dividends.",
             "One option contract controls 100 shares; stock cannot fall below zero."]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (name, profit, maximum, loss, breakeven, description) in zip(axes.flat, positions):
        lines.extend([f"\n{name}: {description}", f"Maximum profit: {maximum}; maximum loss: {loss}; break-even: {breakeven}."])
        ax.plot(s, profit, color="#214e70", linewidth=2)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(title=name, xlabel="Stock price at expiration ($)", ylabel="Position profit ($)")
        ax.grid(alpha=0.2)
    fig.suptitle("Q1: Combination profits at expiration (100-share contract size)")
    return "\n".join(lines), fig


def live_analysis(ticker, rate):
    import pandas as pd
    import yfinance as yf

    now = datetime.now(ZoneInfo("America/New_York"))
    stock = yf.Ticker(ticker)
    # Use completed daily observations so Q6 has exactly 30 daily returns.
    history = stock.history(period="6mo", auto_adjust=True)
    if history.empty:
        raise RuntimeError(f"No price history received for {ticker}")
    closes = history["Close"].dropna()
    if closes.index.tz is not None:
        dates = closes.index.tz_convert("America/New_York").date
    else:
        dates = closes.index.date
    closes = closes[np.asarray(dates) < now.date()]
    if len(closes) < 31:
        raise RuntimeError("Need 31 completed closes for 30 trading-day returns")
    returns = np.log(closes / closes.shift(1)).dropna().tail(30)
    realized = float(returns.std(ddof=1) * np.sqrt(252))

    intraday = stock.history(period="5d", interval="1m", auto_adjust=False)
    if intraday.empty or intraday["Close"].dropna().empty:
        raise RuntimeError("No recent intraday spot quote received")
    spot_series = intraday["Close"].dropna()
    spot = float(spot_series.iloc[-1])
    spot_timestamp = str(spot_series.index[-1])
    expiries = []
    for expiry in stock.options:
        expiration = datetime.strptime(expiry, "%Y-%m-%d").replace(
            hour=16, tzinfo=ZoneInfo("America/New_York"))
        days = (expiration - now).total_seconds() / 86400
        if days > 0:
            expiries.append((expiry, days))
    if not expiries:
        raise RuntimeError("No unexpired listed options received")
    expiry, days = min(expiries, key=lambda item: abs(item[1] - 30))
    time = days / 365.0
    calls = stock.option_chain(expiry).calls.copy()
    calls = calls[np.isfinite(calls["bid"]) & (calls["bid"] > 0)].copy()
    if calls.empty:
        raise RuntimeError("No calls with positive bids in selected expiration")
    rows = []
    for _, row in calls.iterrows():
        bid, ask = float(row["bid"]), float(row["ask"])
        record = {"contract": row["contractSymbol"], "strike": float(row["strike"]),
                  "bid": bid, "ask": ask, "market_price": float("nan"),
                  "price_source": "", "iv": float("nan"), "status": ""}
        if np.isfinite(ask) and 0 < ask < bid:
            record["status"] = "Skipped: crossed bid/ask quote"
        else:
            market_price = (bid + ask) / 2 if np.isfinite(ask) and ask >= bid else bid
            record.update(market_price=market_price,
                          price_source="midpoint" if np.isfinite(ask) and ask >= bid else "bid")
            try:
                record["iv"] = implied_volatility(market_price, spot, record["strike"], time, rate, "call")
                record["status"] = "OK"
            except ValueError as exc:
                record["status"] = str(exc)
        rows.append(record)
    table = pd.DataFrame(rows).sort_values("strike")
    valid = table[table["status"] == "OK"]
    if valid.empty:
        raise RuntimeError("No positive-bid calls have solvable IV under this model")
    central = valid[valid["strike"].between(0.8 * spot, 1.2 * spot)]
    left = central[central["strike"] <= spot]
    right = central[central["strike"] >= spot]
    shape = "Too few strikes near spot to identify a stable curve shape."
    if len(left) >= 3 and len(right) >= 3:
        left_slope = np.polyfit(left["strike"], left["iv"], 1)[0]
        right_slope = np.polyfit(right["strike"], right["iv"], 1)[0]
        if left_slope < 0 < right_slope:
            shape = "The near-spot curve is consistent with a volatility smile (down then up)."
        elif left_slope < 0 and right_slope <= 0:
            shape = "The near-spot curve is consistent with a downside volatility skew, or smirk."
        elif left_slope > 0 and right_slope >= 0:
            shape = "The near-spot curve is consistent with an upward volatility skew."
        else:
            shape = "The near-spot curve has an irregular or inverted shape; inspect quote quality."
    nearest = table.loc[(table["strike"] - spot).abs().idxmin()]
    # Do not silently substitute a farther strike if the nearest quote is invalid.
    atm_iv = float(nearest["iv"])
    vix = float("nan")
    vix_timestamp = "Unavailable"
    try:
        vix_history = yf.Ticker("^VIX").history(period="5d", interval="1m")
        vix_values = vix_history["Close"].dropna()
        if not vix_values.empty:
            vix = float(vix_values.iloc[-1])
            vix_timestamp = str(vix_values.index[-1])
    except Exception as exc:
        vix_timestamp = f"Download failed: {exc}"

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(valid["strike"], 100 * valid["iv"], ".-", color="#214e70", markersize=4)
    ax.axvline(spot, color="#a95e24", linestyle="--", label=f"Spot: ${spot:.2f}")
    ax.set(title=f"Q5: {ticker} call implied volatility | expiration {expiry}",
           xlabel="Strike ($)", ylabel="Annualized implied volatility (%)")
    ax.grid(alpha=0.2)
    ax.legend()

    lines = ["Q5. LIVE IMPLIED VOLATILITY", f"Downloaded: {now.isoformat(timespec='seconds')}",
             f"Source: Yahoo Finance via yfinance; https://finance.yahoo.com/quote/{ticker}/options/",
             f"Ticker: {ticker}; spot: ${spot:.2f}; spot timestamp: {spot_timestamp}",
             f"Expiration: {expiry}; time: {time:.6f} years; assumed annual risk-free rate: {rate:.2%}",
             "Use bid/ask midpoint when available, otherwise bid. All positive-bid calls are attempted.",
             f"Positive-bid calls: {len(table)}; solved: {len(valid)}; skipped: {len(table)-len(valid)}.",
             f"Nearest-spot positive-bid strike: ${nearest['strike']:.2f}; IV: {atm_iv:.2%}."]
    if nearest["status"] != "OK":
        lines.append(f"Nearest-strike IV unavailable: {nearest['status']}")
    if math.isfinite(vix):
        lines.append(f"Latest available VIX: {vix:.2f} ({vix_timestamp}).")
    else:
        lines.append(f"VIX unavailable: {vix_timestamp}.")
    lines.extend([
        "VIX source: https://finance.yahoo.com/quote/%5EVIX/",
        "VIX measures approximately 30-day S&P 500 option-implied volatility, not this single call's IV.",
        "Yahoo quotes may be delayed or stale; inspect timestamps and wide spreads in the CSV.",
        "Model limitation: SPY options are American and SPY pays dividends; this assignment's",
        "European, zero-dividend model gives approximate IVs. Bounds failures are recorded, not forced.",
        "A U-shaped curve is a volatility smile; a downward-sloping equity curve is a volatility skew",
        "or smirk. Inspect the actual plot to identify today's shape: sparse or stale quotes can distort it.",
        f"Data-based description (linear slopes within 20% of spot): {shape}",
        "Different IVs across strikes contradict the constant-volatility Black-Scholes model.",
        "Equity downside skew reflects expensive downside protection and heavier left-tail risk",
        "than that model implies. Black-Scholes assumes normal log returns (lognormal prices),",
        "not normally distributed stock prices; the curve describes risk-neutral pricing, not",
        "a direct estimate of the real-world return distribution.",
        "\nQ6. REALIZED VERSUS IMPLIED VOLATILITY",
        f"Realized volatility (30 log returns, sample SD x sqrt(252)): {realized:.2%}",
        f"ATM implied volatility (nearest positive-bid strike): {atm_iv:.2%}",
        f"Realized-return window: {returns.index[0].date()} through {returns.index[-1].date()}.",
        "This compares backward-looking 30-trading-day RV with forward-looking roughly 30-calendar-day IV.",
        "\nQ7. THE VOLATILITY RISK PREMIUM",
    ])
    if math.isfinite(atm_iv):
        relation = "higher than" if atm_iv > realized else "lower than" if atm_iv < realized else "equal to"
        comparison = (f"For {ticker} in this download, ATM IV ({atm_iv:.2%}) is {relation} "
                      f"past-30-return realized volatility ({realized:.2%}). ")
    else:
        comparison = "The nearest-strike quote has no valid model IV, so this download cannot establish the comparison. "
    lines.append(comparison +
        "An option seller may demand a premium for insuring uncertain future volatility, jumps, "
        "and crash losses, especially when losses arrive during market stress and hedging becomes "
        "costly. Historical realized volatility is not future realized volatility, so this gap is "
        "neither a guaranteed profit nor a pure measure of the risk premium. Someone repeatedly "
        "selling options can collect small premiums in quiet weeks, but a large adverse move "
        "can wipe out months or years of income. A short straddle loses on large moves in either "
        "direction; an uncovered short call has unlimited upside loss, while a short put has "
        "substantial downside loss. Volatility spikes and margin calls may force liquidation "
        "before prices recover, and leverage can bankrupt the seller. Delta hedging removes "
        "some directional exposure but leaves short-gamma and jump risk; it does not make the "
        "weekly premium safe income.")
    return "\n".join(lines), fig, table


def save_report(path, text, figures):
    """Create a readable PDF containing all written answers and plots."""
    wrapped = []
    for line in text.splitlines():
        wrapped.extend(textwrap.wrap(line, width=94) or [""])
    with PdfPages(path) as pdf:
        for start in range(0, len(wrapped), 48):
            page = plt.figure(figsize=(8.5, 11))
            page.text(0.07, 0.96, "QI Homework 6 - Answers", fontsize=14, va="top", weight="bold")
            page.text(0.07, 0.91, "\n".join(wrapped[start:start+48]), fontsize=9,
                      va="top", linespacing=1.5, fontfamily="monospace")
            pdf.savefig(page)
            plt.close(page)
        for fig in figures:
            pdf.savefig(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--rate", type=float, default=0.04,
                        help="Assumed annual risk-free rate, as a decimal (default 0.04)")
    parser.add_argument("--offline", action="store_true", help="Run Q1-4 only")
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening windows")
    args = parser.parse_args()
    out = Path("hw06_output")
    out.mkdir(exist_ok=True)
    q1, strategy_figure = combinations()
    q2 = ("Q2. CALENDAR SPREAD\n"
          "A calendar spread cannot be represented by a single expiration payoff diagram because "
          "the calls expire at different times: when the one-week short call expires, the three-month "
          "long call still has time value that depends on the stock price and remaining implied volatility. "
          "If the stock stays near the strike for a week, the short call loses its time value while "
          "the longer-dated call generally loses less, which can benefit the spread. The position is "
          "typically betting on limited near-term movement and favorable relative time decay, with "
          "exposure to changes in longer-term volatility; its profit also depends on the initial debit.")
    call = black_scholes(100, 100, 1, 0.04, 0.25, "call")
    put = black_scholes(100, 100, 1, 0.04, 0.25, "put")
    q3 = (f"Q3. BLACK-SCHOLES\nS=100, K=100, T=1, r=0.04, sigma=0.25\n"
          f"Call: ${call:.2f}; put: ${put:.2f}\n"
          f"Put-call parity: C-P = {call-put:.6f}; S-K*exp(-rT) = {100-100*math.exp(-0.04):.6f}")
    recovered_call = implied_volatility(call, 100, 100, 1, 0.04, "call")
    recovered_put = implied_volatility(put, 100, 100, 1, 0.04, "put")
    q4 = (f"Q4. IMPLIED-VOLATILITY ROUND TRIP\nOriginal: 0.2500; "
          f"recovered from call: {recovered_call:.4f}; recovered from put: {recovered_put:.4f}")
    figures = [strategy_figure]
    sections = [q1, q2, q3, q4]
    if args.offline:
        sections.append("Q5-Q7: Not run (--offline). Run without --offline to obtain live results.")
    else:
        try:
            live_text, iv_figure, table = live_analysis(args.ticker.upper(), args.rate)
            sections.append(live_text)
            figures.append(iv_figure)
            table.to_csv(out / "option_implied_volatility.csv", index=False)
        except Exception as exc:
            sections.append(f"Q5-Q7 INCOMPLETE: {type(exc).__name__}: {exc}\n"
                            "No substitute data was used. Retry when Yahoo Finance provides quotes.")
    answers = "\n\n".join(sections)
    print(answers)
    (out / "answers.txt").write_text(answers, encoding="utf-8")
    strategy_figure.savefig(out / "combination_profits.png", dpi=180)
    if len(figures) > 1:
        figures[1].savefig(out / "implied_volatility.png", dpi=180)
    save_report(out / "HW6_answers.pdf", answers, figures)
    print(f"\nSaved answers and plots to: {out.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
