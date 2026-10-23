
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta



def payoff(kind, strike, premium, spot):
    if kind == "long_call":
        return max(spot - strike, 0) - premium
    elif kind == "long_put":
        return max(strike - spot, 0) - premium
    elif kind == "short_call":
        return premium - max(spot - strike, 0)
    elif kind == "short_put":
        return premium - max(strike - spot, 0)
    else:
        raise ValueError("Invalid option kind")


kinds = ["long_call", "long_put", "short_call", "short_put"]

print("Q1: Payoffs")
for kind in kinds:
    for spot in [80, 100, 120]:
        print(f"{kind}, spot={spot}: {payoff(kind, 100, 6, spot):.2f}")




spots = np.linspace(60, 140, 400)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for ax, kind in zip(axes.flat, kinds):
    profits = [payoff(kind, 100, 6, s) for s in spots]

    ax.plot(spots, profits)
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(100, color="gray", linestyle="--")
    ax.set_title(kind.replace("_", " ").title())
    ax.set_xlabel("Spot Price at Expiry")
    ax.set_ylabel("Profit per Share")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Q2_payoffs.png", dpi=300)



ticker = "SPY"
t = yf.Ticker(ticker)

expiries = list(t.options)
print("\nQ3: Available Expiries")
print(expiries)

if not expiries:
    raise RuntimeError("No option expiries returned by Yahoo Finance.")

today = datetime.today().date()

dates = [datetime.strptime(e, "%Y-%m-%d").date() for e in expiries]

short_candidates = [d for d in dates if d > today]
short_date = min(
    short_candidates,
    key=lambda d: abs((d - today).days - 30)
)

long_candidates = [d for d in dates if (d - today).days >= 190]
if not long_candidates:
    raise RuntimeError("No expiry at least six months away was found.")

long_date = min(
    long_candidates,
    key=lambda d: abs((d - today).days - 210)
)

short_expiry = short_date.isoformat()
long_expiry = long_date.isoformat()


try:
    spot = float(t.fast_info["last_price"])
except Exception:
    history = t.history(period="5d")
    spot = float(history["Close"].iloc[-1])

short_chain = t.option_chain(short_expiry)
long_chain = t.option_chain(long_expiry)

print("\nQ3: Chain Information")
print("Ticker:", ticker)
print("Current Spot:", round(spot, 2))
print("Short Expiry:", short_expiry)
print("Number of Calls:", len(short_chain.calls))
print("Number of Puts:", len(short_chain.puts))



def calculate_values(calls, spot):
    df = calls.copy()

    for col in ["strike", "bid", "ask"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["strike", "bid", "ask"])
    df = df[(df["bid"] > 0) & (df["ask"] >= df["bid"])].copy()

    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["intrinsic"] = np.maximum(spot - df["strike"], 0)
    df["extrinsic"] = df["mid"] - df["intrinsic"]

    return df.sort_values("strike")


def print_nearest(df, spot, title):
    nearest = df.copy()
    nearest["distance"] = abs(nearest["strike"] - spot)
    nearest = nearest.nsmallest(10, "distance")
    nearest = nearest.sort_values("strike")

    print(f"\n{title}")
    print(
        nearest[
            ["strike", "mid", "intrinsic", "extrinsic"]
        ].to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )


short_calls = calculate_values(short_chain.calls, spot)
long_calls = calculate_values(long_chain.calls, spot)

if short_calls.empty or long_calls.empty:
    raise RuntimeError("No valid positive-bid calls for one of the expiries.")

print_nearest(short_calls, spot, "Q4: Short Expiry - 10 Nearest Strikes")

assert np.allclose(
    short_calls["intrinsic"] + short_calls["extrinsic"],
    short_calls["mid"]
)



short_max = short_calls.loc[short_calls["extrinsic"].idxmax()]

print("\nQ5: Short-Dated Maximum Extrinsic")
print("Strike:", short_max["strike"])
print("Maximum Extrinsic:", round(short_max["extrinsic"], 2))
print("Current Spot:", round(spot, 2))

plt.figure(figsize=(10, 6))
plt.plot(
    short_calls["strike"],
    short_calls["extrinsic"],
    label=short_expiry
)
plt.axvline(spot, color="red", linestyle="--", label="Current Spot")
plt.axhline(0, color="gray", linewidth=1)

plt.xlabel("Strike Price")
plt.ylabel("Extrinsic Value")
plt.title("SPY Call Extrinsic Value - Short Expiry")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Q5_short_extrinsic.png", dpi=300)



print("\nQ6: Long Expiry Information")
print("Long Expiry:", long_expiry)
print("Number of Calls:", len(long_chain.calls))
print("Number of Puts:", len(long_chain.puts))

print_nearest(long_calls, spot, "Q6: Long Expiry - 10 Nearest Strikes")

long_max = long_calls.loc[long_calls["extrinsic"].idxmax()]

print("\nQ6: Maximum Extrinsic Comparison")
print(
    f"Short Expiry ({short_expiry}): "
    f"{short_max['extrinsic']:.2f}"
)
print(
    f"Long Expiry ({long_expiry}): "
    f"{long_max['extrinsic']:.2f}"
)
print("Long Expiry Maximum Strike:", long_max["strike"])
print("Current Spot:", round(spot, 2))

# Plot both curves
plt.figure(figsize=(11, 6))

plt.plot(
    short_calls["strike"],
    short_calls["extrinsic"],
    label=f"Short: {short_expiry}"
)

plt.plot(
    long_calls["strike"],
    long_calls["extrinsic"],
    label=f"Long: {long_expiry}"
)

plt.axvline(
    spot,
    color="red",
    linestyle="--",
    label=f"Spot: {spot:.2f}"
)

plt.axhline(0, color="gray", linewidth=1)

plt.xlabel("Strike Price")
plt.ylabel("Extrinsic Value")
plt.title("SPY Call Extrinsic Value: Short vs Long Expiry")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Q6_extrinsic_comparison.png", dpi=300)

# Additional checks
assert np.allclose(
    long_calls["intrinsic"] + long_calls["extrinsic"],
    long_calls["mid"]
)

print("\nAll calculations completed.")
print("Saved plots:")
print("Q2_payoffs.png")
print("Q5_short_extrinsic.png")
print("Q6_extrinsic_comparison.png")

plt.show()
