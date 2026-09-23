"""Homework 2 — Algory QI Education, Fall 2026"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def present_value(cash_flows, rate):
    """Present value of cash flows, with the first cash flow one year from now."""
    total = 0

    for year, cash_flow in enumerate(cash_flows, start=1):
        total += cash_flow / (1 + rate) ** year

    return total



def bond_price(face, coupon_rate, years, market_rate):
    """Price a bond with annual coupons and face value repaid at maturity."""
    coupon = face * coupon_rate
    cash_flows = [coupon] * years

    # Final year pays both the last coupon and the face value.
    cash_flows[-1] += face

    return present_value(cash_flows, market_rate)


def annualised_return(prices):
    """Annualised return from a price series, using 252 trading days."""
    prices = pd.Series(prices).dropna()
    daily_returns = prices.pct_change(fill_method=None).dropna()

    if len(daily_returns) == 0:
        return np.nan

    total_growth = prices.iloc[-1] / prices.iloc[0]
    return total_growth ** (252 / len(daily_returns)) - 1


def annualised_volatility(prices):
    """Annualised standard deviation of daily returns."""
    prices = pd.Series(prices).dropna()
    daily_returns = prices.pct_change(fill_method=None).dropna()

    return daily_returns.std() * np.sqrt(252)


def beta(stock_prices, market_prices):
    """Beta = covariance(stock returns, market returns) / variance(market returns)."""
    stock_prices = pd.Series(stock_prices).dropna()
    market_prices = pd.Series(market_prices).dropna()

    stock_returns = stock_prices.pct_change(fill_method=None).dropna().rename("stock")
    market_returns = market_prices.pct_change(fill_method=None).dropna().rename("market")

    returns = pd.concat(
        [stock_returns, market_returns],
        axis=1
    ).dropna()

    market_variance = returns["market"].var()

    if market_variance == 0:
        return np.nan

    return (
        returns["stock"].cov(returns["market"])
        / market_variance
    )


def main():

    # Q1
    q1 = present_value([10, 15, 20], 0.10)

    print(
        "Q1 present_value([10, 15, 20], 0.10) =",
        round(q1, 2)
    )


    q2 = bond_price(1000, 0.04, 10, 0.04)

    print(
        "Q2 bond_price(1000, 0.04, 10, 0.04) =",
        round(q2, 2)
    )



    print("\nQ3 Bond prices")

    market_rates = [0.02, 0.04, 0.0496]

    for rate in market_rates:

        price = bond_price(
            1000,
            0.04,
            10,
            rate
        )

        print(
            f"Market rate {rate:.2%}: "
            f"${price:.2f}"
        )

    rates = np.linspace(
        0,
        0.10,
        101
    )

    bond_prices = []

    for rate in rates:

        price = bond_price(
            1000,
            0.04,
            10,
            rate
        )

        bond_prices.append(price)


    plt.figure(figsize=(8, 5))

    plt.plot(
        rates * 100,
        bond_prices
    )

    plt.xlabel("Market rate (%)")
    plt.ylabel("Bond price ($)")

    plt.title(
        "10-Year 4% Coupon Bond: "
        "Price vs. Market Rate"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "bond_price_curve.png",
        dpi=200
    )

    plt.show()


    print(
        "Shape: The bond price curve bends rather than "
        "forming a straight line; this is convexity, so "
        "price rises more when rates fall than it falls "
        "when rates rise by the same amount."
    )


    tickers = [
        "RL",
        "JPM",
        "XOM"
    ]

    all_tickers = tickers + ["SPY"]


    prices = yf.download(
        all_tickers,
        period="5y",
        progress=False,
        auto_adjust=False
    )["Close"]


    print("\nQ4 Trading days")

    for ticker in all_tickers:

        trading_days = (
            prices[ticker]
            .dropna()
            .shape[0]
        )

        print(
            f"{ticker}: "
            f"{trading_days} trading days"
        )



    rows = []


    for ticker in tickers:

        stock_prices = (
            prices[ticker]
            .dropna()
        )

        spy_prices = (
            prices["SPY"]
            .dropna()
        )


        stock_return = annualised_return(
            stock_prices
        )

        stock_volatility = annualised_volatility(
            stock_prices
        )

        stock_beta = beta(
            stock_prices,
            spy_prices
        )


        rows.append({

            "Ticker": ticker,

            "Annualised Return":
                stock_return,

            "Annualised Volatility":
                stock_volatility,

            "Beta":
                stock_beta
        })


    results = pd.DataFrame(rows)

    results = results.set_index(
        "Ticker"
    )


    print(
        "\nQ5 Return, volatility, and beta"
    )


    print(

        results.to_string(

            formatters={

                "Annualised Return":
                    lambda x: f"{x:.2%}",

                "Annualised Volatility":
                    lambda x: f"{x:.2%}",

                "Beta":
                    lambda x: f"{x:.2f}"
            }

        )

    )



    spy_beta = beta(
        prices["SPY"],
        prices["SPY"]
    )

    print(
        f"\nSPY beta against itself: "
        f"{spy_beta:.2f}"
    )


 
    beta_ranking = (

        results["Beta"]
        .sort_values(
            ascending=False
        )

    )


    volatility_ranking = (

        results[
            "Annualised Volatility"
        ]
        .sort_values(
            ascending=False
        )

    )


    print(
        "\nQ6 Beta ranking, "
        "highest to lowest"
    )


    for rank, (
        ticker,
        value
    ) in enumerate(
        beta_ranking.items(),
        start=1
    ):

        print(
            f"{rank}. "
            f"{ticker}: "
            f"{value:.2f}"
        )


    print(
        "\nQ6 Volatility ranking, "
        "highest to lowest"
    )


    for rank, (
        ticker,
        value
    ) in enumerate(
        volatility_ranking.items(),
        start=1
    ):

        print(
            f"{rank}. "
            f"{ticker}: "
            f"{value:.2%}"
        )


if __name__ == "__main__":
    main()