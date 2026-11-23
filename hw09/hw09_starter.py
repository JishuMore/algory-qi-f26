import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

TRADING_DAYS = 252
OUT = Path(__file__).resolve().parent / "hw09_output"


def recovery_needed(loss_fraction):
    if not 0.0 <= loss_fraction < 1.0:
        raise ValueError("loss_fraction must be in [0, 1)")

    return loss_fraction / (1.0 - loss_fraction)


def annualise_vol(daily_vol, days=TRADING_DAYS):
    return daily_vol * math.sqrt(days)


def var(returns, confidence=0.95):
    r = np.asarray(returns, dtype=float)
    return float(np.quantile(r, 1.0 - confidence))


def cvar(returns, confidence=0.95):
    r = np.asarray(returns, dtype=float)
    threshold = var(r, confidence)
    tail = r[r <= threshold]
    return float(tail.mean())


def simulate_paths(start, drift, vol, days=TRADING_DAYS, paths=1000, seed=0):
    rng = np.random.default_rng(seed)
    dt = 1.0 / days
    z = rng.standard_normal((paths, days))
    log_steps = (drift - 0.5 * vol**2) * dt + vol * math.sqrt(dt) * z
    return start * np.exp(log_steps.sum(axis=1))


def bootstrap_paths(start, daily_returns, days=TRADING_DAYS, paths=1000, seed=0):
    rng = np.random.default_rng(seed)
    r = np.asarray(daily_returns, dtype=float)
    idx = rng.integers(0, len(r), size=(paths, days))
    return start * np.prod(1.0 + r[idx], axis=1)


def load_closes(csv_path=None, years=20):
    if csv_path:
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
        closes = df["Close"]
        closes = closes.loc[closes.index >= closes.index[-1] - pd.DateOffset(years=years)]
    else:
        raw = yf.download("SPY", period=f"{years}y", auto_adjust=True, progress=False)
        closes = raw["Close"]
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
    closes = closes.dropna().sort_index().astype(float)
    closes.name = "Close"
    return closes


def say(log, text=""):
    print(text)
    log.append(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="offline CSV with Date,Close columns")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    log = []
    closes = load_closes(args.csv)
    daily = closes.pct_change().dropna()
    n = len(daily)
    say(log, f"Data: {closes.index[0].date()} to {closes.index[-1].date()}, {n} daily returns")
    say(log, "")

    say(log, "=== Q1/Q2 volatility scaling ===")
    daily_vol = daily.std(ddof=1)
    ann_scaled = annualise_vol(daily_vol)
    say(log, f"Daily volatility:            {daily_vol:.4%}")
    say(log, f"Annualised (x sqrt 252):     {ann_scaled:.2%}")
    year_end = closes.resample("YE").last()
    yearly = year_end.pct_change().dropna()
    last_year = closes.index[-1].year
    yearly = yearly[yearly.index.year < last_year]
    ann_direct = yearly.std(ddof=1)
    say(log, f"Direct annual vol (calendar-year returns, n={len(yearly)}): {ann_direct:.2%}")
    say(log, f"Ratio direct / scaled:       {ann_direct / ann_scaled:.3f}")

    r = daily.values
    down = r < 0
    p_down = down.mean()
    p_dd = (down[1:] & down[:-1]).sum() / down[:-1].sum()
    rho1 = pd.Series(r).autocorr(lag=1)
    vr5 = pd.Series(r).rolling(5).sum().dropna()
    variance_ratio_5 = vr5.var(ddof=1) / (5 * r.var(ddof=1))
    say(log, f"P(down) = {p_down:.3f}   P(down tomorrow | down today) = {p_dd:.3f}")
    say(log, f"Lag-1 autocorrelation of daily returns: {rho1:+.4f}")
    say(log, f"Variance ratio VR(5) = {variance_ratio_5:.3f}   (1.0 means days are independent)")
    say(log, "")

    say(log, "=== Q3 rolling 60-day volatility (annualised) ===")
    roll = daily.rolling(60).std(ddof=1).dropna() * math.sqrt(TRADING_DAYS)
    hi_date, hi_val = roll.idxmax(), roll.max()
    lo_date, lo_val = roll.idxmin(), roll.min()
    say(log, f"Highest: {hi_val:.2%} on {hi_date.date()}")
    say(log, f"Lowest:  {lo_val:.2%} on {lo_date.date()}")

    spikes = []
    for date in roll.sort_values(ascending=False).index:
        if all(abs((date - s).days) > 180 for s in spikes):
            spikes.append(date)
        if len(spikes) == 2:
            break
    for i, s in enumerate(spikes, 1):
        say(log, f"Spike {i}: {roll[s]:.2%} on {s.date()}")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(roll.index, roll.values, lw=0.9)
    ax.scatter([hi_date, lo_date], [hi_val, lo_val], color="red", zorder=3)
    for s in spikes:
        ax.annotate(s.date().isoformat(), (s, roll[s]), xytext=(8, 8),
                    textcoords="offset points", fontsize=8)
    ax.set_title("Rolling 60-day volatility (annualised)")
    ax.set_ylabel("Volatility")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    fig.savefig(OUT / "q3_rolling_vol.png", dpi=150)
    plt.close(fig)
    say(log, "")

    say(log, "=== Q4/Q5 shape and four-sigma days ===")
    mu, sd = daily.mean(), daily.std(ddof=1)
    skew = daily.skew()
    exc_kurt = daily.kurt()
    z = (daily - mu) / sd
    actual_4s = int((z.abs() > 4).sum())
    say(log, f"Skewness: {skew:+.3f}")
    say(log, f"Excess kurtosis: {exc_kurt:+.3f}")
    say(log, f"Days beyond 4 standard deviations (two-sided): {actual_4s}")
    expected_4s = n * math.erfc(4 / math.sqrt(2))
    say(log, f"Expected under normality: {expected_4s:.3f}")
    say(log, f"Actual: {actual_4s}")
    ratio = actual_4s / expected_4s if expected_4s > 0 else float("inf")
    say(log, f"Ratio actual / expected: {ratio:.1f}")
    xs = np.linspace(daily.min(), daily.max(), 400)
    pdf = np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(daily.values, bins=150, density=True, alpha=0.6, label="SPY daily returns")
    ax.plot(xs, pdf, color="black", lw=1.5, label="Fitted normal")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-3)
    ax.set_title("Daily returns vs fitted normal (log density)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "q4_return_histogram.png", dpi=150)
    plt.close(fig)
    say(log, "")

    say(log, "=== Q6 loss and recovery ===")
    losses = [0.10, 0.25, 0.50, 0.75, 0.90]
    recoveries = [recovery_needed(L) for L in losses]
    for L, g in zip(losses, recoveries):
        say(log, f"Lose {L:.0%} -> need +{g:.1%} to get back to even")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([f"-{L:.0%}" for L in losses], [g * 100 for g in recoveries], color="steelblue")
    ax.bar_label(bars, labels=[f"+{g:.0%}" for g in recoveries], padding=3)
    ax.set_xlabel("Loss")
    ax.set_ylabel("Gain required to recover (%)")
    ax.set_title("Asymmetry of compounding")
    fig.tight_layout()
    fig.savefig(OUT / "q6_recovery.png", dpi=150)
    plt.close(fig)
    say(log, "")

    say(log, "=== Q7/Q8/Q9 Monte Carlo, VaR and CVaR ===")
    start = 100_000.0
    paths = 10_000
    log_daily = np.log1p(daily.values)
    sigma_ann = daily_vol * math.sqrt(TRADING_DAYS)
    mu_log_ann = log_daily.mean() * TRADING_DAYS
    drift_ann = mu_log_ann + 0.5 * sigma_ann**2
    say(log, f"GBM inputs: drift (arithmetic) = {drift_ann:.2%}, vol = {sigma_ann:.2%}")
    gbm_final = simulate_paths(start, drift_ann, sigma_ann, paths=paths, seed=1)
    say(log, "Q7 GBM one-year final value ($):")
    say(log, f"  mean    {gbm_final.mean():>12,.0f}")
    say(log, f"  median  {np.median(gbm_final):>12,.0f}")
    say(log, f"  5th pct {np.percentile(gbm_final, 5):>12,.0f}")
    say(log, f"  95th pct{np.percentile(gbm_final, 95):>12,.0f}")
    boot_final = bootstrap_paths(start, daily.values, paths=paths, seed=2)
    gbm_ret = gbm_final / start - 1.0
    boot_ret = boot_final / start - 1.0
    say(log, "")
    say(log, "Q8/Q9 one-year VaR and CVaR at 95% (loss shown as $ and % of $100,000)")
    say(log, f"{'method':<12}{'VaR $':>14}{'VaR %':>9}{'CVaR $':>14}{'CVaR %':>9}")
    for name, rets in [("GBM", gbm_ret), ("Bootstrap", boot_ret)]:
        v, c = var(rets, 0.95), cvar(rets, 0.95)
        say(log, f"{name:<12}{-v * start:>14,.0f}{-v:>9.2%}{-c * start:>14,.0f}{-c:>9.2%}")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(gbm_final, bins=120, alpha=0.5, label="GBM (normal)", density=True)
    ax.hist(boot_final, bins=120, alpha=0.5, label="Bootstrap (actual returns)", density=True)
    ax.axvline(np.mean(gbm_final), color="C0", ls="--", label="GBM mean")
    ax.axvline(np.median(gbm_final), color="C0", ls=":", label="GBM median")
    ax.set_xlabel("Portfolio value after one year ($)")
    ax.set_title("One-year Monte Carlo, 10,000 paths")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "q7_monte_carlo.png", dpi=150)
    plt.close(fig)
    (OUT / "results.txt").write_text("\n".join(log) + "\n")
    print(f"\nWrote plots and results.txt to {OUT}")


if __name__ == "__main__":
    main()
