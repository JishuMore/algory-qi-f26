"""Homework 8 — Algory QI Education, Fall 2026."""

import argparse
import csv
import io
import json
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from scipy.stats import kurtosis, norm, skew


HERE = Path(__file__).resolve().parent
START = "2006-11-13"
END = "2026-11-12"
ARCHIVE_END = "2026-07-23"
SPOT = 760.88
DAYS = 21
TRADING_DAYS = 252
RATE = 0.0412  # November 12, 2026 one-month Treasury par yield.
YAHOO_ARCHIVE = (
    "https://raw.githubusercontent.com/Setooooo/etf-insight/master/data/raw/SPY.csv"
)
NASDAQ_HISTORY = (
    "https://api.nasdaq.com/api/quote/SPY/historical"
    "?assetclass=etf&fromdate=2026-07-24&todate=2026-11-12&limit=7000"
)


def download_closes(destination):
    """Download raw SPY closes and save a date-sorted CSV."""
    with urlopen(YAHOO_ARCHIVE, timeout=30) as response:
        archive = csv.DictReader(io.StringIO(response.read().decode("utf-8")))
        daily = {
            row["Date"]: float(row["Close"])
            for row in archive
            if START <= row["Date"] <= ARCHIVE_END
        }

    with urlopen(NASDAQ_HISTORY, timeout=30) as response:
        recent = json.load(response)["data"]["tradesTable"]["rows"]
    for row in recent:
        date = datetime.strptime(row["date"], "%m/%d/%Y").date().isoformat()
        if ARCHIVE_END < date <= END:
            daily[date] = float(row["close"].replace(",", "").lstrip("$"))

    if min(daily, default="") != START or max(daily, default="") != END:
        raise RuntimeError("Downloaded SPY history does not cover the analysis dates.")
    with destination.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["Date", "Close"])
        writer.writerows((date, f"{daily[date]:.6f}") for date in sorted(daily))


def load_closes(path):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    dates = np.array([row["Date"] for row in rows], dtype="datetime64[D]")
    prices = np.array([float(row["Close"]) for row in rows])
    if not (len(dates) > 60 and np.all(np.diff(dates) > np.timedelta64(0, "D"))):
        raise ValueError("Expected date-sorted SPY closes with no duplicate dates.")
    return dates, prices


def n_day_returns(prices, n):
    """Return all overlapping simple returns, P[t+n] / P[t] - 1."""
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("prices must be a one-dimensional series of positive closes.")
    if not isinstance(n, (int, np.integer)) or not 0 < n < prices.size:
        raise ValueError("n must be an integer from 1 to len(prices) - 1.")
    return prices[n:] / prices[:-n] - 1


def option_direction(kind):
    if kind == "call":
        return 1
    if kind == "put":
        return -1
    raise ValueError("kind must be 'call' or 'put'.")


def price_from_history(prices, spot, strike, days, kind):
    """Average the option's expiration payoffs per share without discounting."""
    terminal = spot * (1 + n_day_returns(prices, days))
    payoff = np.maximum(option_direction(kind) * (terminal - strike), 0)
    return float(payoff.mean())


def black_scholes(kind, spot, strike, years, rate, vol):
    """Homework 6 European pricing formula with no dividend yield."""
    direction = option_direction(kind)
    strike = np.asarray(strike)
    vol_time = vol * np.sqrt(years)
    d1 = (np.log(spot / strike) + (rate + vol**2 / 2) * years) / vol_time
    d2 = d1 - vol_time
    discounted_strike = strike * np.exp(-rate * years)
    return direction * (
        spot * norm.cdf(direction * d1)
        - discounted_strike * norm.cdf(direction * d2)
    )


def plot_returns(returns, mean, std, output):
    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    ax.hist(returns, bins=55, density=True, color="#397b9d", alpha=0.75,
            label="Observed 21-day returns")
    x = np.linspace(returns.min(), returns.max(), 500)
    ax.plot(x, norm.pdf(x, mean, std), color="#ac4e35", linewidth=2,
            label="Normal with matching mean and standard deviation")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title="SPY 21-day returns", xlabel="Simple return", ylabel="Density")
    ax.legend()
    fig.savefig(output / "01_return_distribution.png", dpi=170)
    plt.close(fig)


def plot_terminal_prices(terminal, strike, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True,
                             layout="constrained")
    edges = np.unique(np.append(np.linspace(terminal.min(), terminal.max(), 56), strike))
    counts, _ = np.histogram(terminal, bins=edges)
    for ax, kind in zip(axes, ("call", "put")):
        pays = option_direction(kind) * ((edges[:-1] + edges[1:]) / 2 - strike) > 0
        colors = np.where(pays, "#397b9d", "#c9cfd5")
        ax.bar(edges[:-1], counts, width=np.diff(edges), color=colors, align="edge")
        ax.axvline(strike, color="#ac4e35", linewidth=1.6, linestyle="--",
                   label=f"Strike: ${strike:,.2f}")
        ax.set(title=f"{kind.title()} payoff region shaded blue",
               xlabel="SPY at expiration ($)")
        ax.legend()
    axes[0].set_ylabel("Historical windows")
    fig.savefig(output / "02_terminal_prices.png", dpi=170)
    plt.close(fig)


def compare_strikes(prices, years, vol, output):
    strikes = SPOT * np.linspace(0.85, 1.15, 301)
    best_dollar_gap = (0.0, "", 0.0, 0.0)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex="col",
                             layout="constrained")
    for column, kind in enumerate(("call", "put")):
        historical = np.array([
            price_from_history(prices, SPOT, strike, DAYS, kind) for strike in strikes
        ])
        bs = black_scholes(kind, SPOT, strikes, years, RATE, vol)
        gap = historical - bs
        index = np.argmax(np.abs(gap))
        result = (abs(gap[index]), kind, strikes[index], gap[index])
        if result[0] > best_dollar_gap[0]:
            best_dollar_gap = result

        otm = option_direction(kind) * (strikes - SPOT) > 0
        otm_strikes, otm_gap = strikes[otm], gap[otm]
        otm_index = np.argmax(np.abs(otm_gap))
        relative = gap / bs
        relative_index = np.argmax(np.abs(relative))
        print(f"{kind.title()} largest absolute dollar gap: "
              f"${strikes[index]:.2f}, history minus BS = ${gap[index]:+.2f}")
        print(f"  OTM only: ${otm_strikes[otm_index]:.2f}, "
              f"history minus BS = ${otm_gap[otm_index]:+.2f}")
        print(f"  Largest absolute percentage gap: ${strikes[relative_index]:.2f}, "
              f"{relative[relative_index]:+.2%} vs BS")

        price_axis, gap_axis = axes[:, column]
        price_axis.plot(strikes, historical, color="#397b9d", label="Historical")
        price_axis.plot(strikes, bs, color="#ac4e35", linestyle="--",
                        label="Black–Scholes")
        price_axis.set(title=f"{kind.title()} prices", ylabel="Price ($ per share)")
        price_axis.legend()
        gap_axis.plot(strikes, gap, color="#397b9d")
        gap_axis.axhline(0, color="#6b7278", linewidth=0.9)
        gap_axis.set(xlabel="Strike ($)", ylabel="History minus BS ($)")
        for axis in (price_axis, gap_axis):
            axis.axvline(SPOT, color="#6b7278", linewidth=0.9, linestyle=":")
    _, kind, strike, gap = best_dollar_gap
    print(f"Overall largest absolute dollar gap: {kind}, strike ${strike:.2f}, "
          f"history minus BS = ${gap:+.2f}")
    fig.savefig(output / "03_strike_comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Price SPY options from historical closes.")
    parser.add_argument("--download", action="store_true", help="Rebuild the SPY close CSV.")
    args = parser.parse_args()
    close_file = HERE / "spy_daily_closes.csv"
    if args.download or not close_file.exists():
        download_closes(close_file)
    dates, prices = load_closes(close_file)

    returns = n_day_returns(prices, DAYS)
    mean = returns.mean()
    std = returns.std(ddof=1)
    p01, p99 = np.percentile(returns, [1, 99])
    print(f"SPY closes: {prices.size:,} from {dates[0]} to {dates[-1]}")
    print(f"21-day windows: {returns.size:,}")
    print(f"Mean: {mean:.4%} | Sample standard deviation: {std:.4%}")
    print(f"1st percentile: {p01:.4%} | 99th percentile: {p99:.4%}")
    print(f"Skewness: {skew(returns, bias=False):.4f}")
    print(f"Excess kurtosis: {kurtosis(returns, fisher=True, bias=False):.4f}")
    plot_returns(returns, mean, std, HERE)

    strike = SPOT * 1.03
    terminal = SPOT * (1 + returns)
    years = DAYS / TRADING_DAYS
    vol = np.diff(np.log(prices))[-60:].std(ddof=1) * np.sqrt(TRADING_DAYS)
    print(f"\nSpot: ${SPOT:.2f} | Strike: ${strike:.4f} | Expiration: {DAYS} trading days")
    print(f"60-day annualized log-return volatility: {vol:.4%} | Rate: {RATE:.2%}")
    print("Kind      Historical    Black–Scholes     % vs BS         ITM share")
    for kind in ("call", "put"):
        historical = price_from_history(prices, SPOT, strike, DAYS, kind)
        bs = black_scholes(kind, SPOT, strike, years, RATE, vol)
        itm = np.count_nonzero(option_direction(kind) * (terminal - strike) > 0)
        print(f"{kind.title():<10}${historical:>8.4f}     ${bs:>8.4f}"
              f"       {(historical - bs) / bs:>+7.2%}     {itm / returns.size:>7.2%}"
              f" ({itm:,}/{returns.size:,})")
    plot_terminal_prices(terminal, strike, HERE)

    print()
    compare_strikes(prices, years, vol, HERE)

    print("\n20% out-of-the-money contracts")
    for kind in ("call", "put"):
        tail_strike = SPOT * (1 + 0.20 * option_direction(kind))
        historical = price_from_history(prices, SPOT, tail_strike, DAYS, kind)
        bs = black_scholes(kind, SPOT, tail_strike, years, RATE, vol)
        itm = np.count_nonzero(option_direction(kind) * (terminal - tail_strike) > 0)
        print(f"{kind.title()} ${tail_strike:.2f}: history ${historical:.8f}, "
              f"BS ${bs:.8g}, {itm} paying windows")


if __name__ == "__main__":
    main()
