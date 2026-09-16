import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ticker = "RL"
market = "SPY"


stock = yf.Ticker(ticker).history(period="1y")["Close"]
spy = yf.Ticker(market).history(period="1y")["Close"]

prices = pd.concat([stock, spy], axis=1)
prices.columns = [ticker, market]

# Remove empty rows
prices = prices.dropna()

print("Tickers:", ticker, "and", market)
print("Number of trading days:", len(prices))
print("First date:", prices.index[0].date())
print("Last date:", prices.index[-1].date())



daily_returns = prices.pct_change().dropna()

for symbol in [ticker, market]:

    last_close = prices[symbol].iloc[-1]

    yearly_return = (
        prices[symbol].iloc[-1] /
        prices[symbol].iloc[0] - 1
    )

    annualized_volatility = (
        daily_returns[symbol].std() *
        np.sqrt(252)
    )

    print()
    print(symbol)
    print("Last close:", round(last_close, 2))
    print("Return over year:", round(yearly_return * 100, 2), "%")
    print(
        "Annualized volatility:",
        round(annualized_volatility * 100, 2),
        "%"
    )


rebased = prices / prices.iloc[0] * 100

plt.figure(figsize=(10, 6))

plt.plot(
    rebased.index,
    rebased[ticker],
    label=ticker
)

plt.plot(
    rebased.index,
    rebased[market],
    label=market 
)

plt.xlabel("Date")
plt.ylabel("Rebased Price (Starting Value = 100)")
plt.title(ticker + " vs SPY")
plt.legend()

plt.show()



stock_returns = daily_returns[ticker]

biggest_date = stock_returns.abs().idxmax()
biggest_move = stock_returns.loc[biggest_date]

print()
print("Biggest single-day move for", ticker)
print("Date:", biggest_date.date())
print("Move:", round(biggest_move * 100, 2), "%")