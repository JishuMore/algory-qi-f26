import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def simulate_dice(trials, seed=0):
    rng = np.random.default_rng(seed)

    picked_four = rng.random(trials) < 0.5
    sides = np.where(picked_four, 4, 6)

    rolls = (rng.random(trials) * sides).astype(int) + 1

    rolled_one = rolls == 1

    return picked_four[rolled_one].mean()


def simulate_coins(trials, seed=0):
    rng = np.random.default_rng(seed)

    flips = rng.integers(0, 2, size=(trials, 3))

    heads = flips.sum(axis=1)
    tails = 3 - heads

    payout = heads * tails

    return payout.mean()



def p_down(returns):
    returns = np.asarray(returns, dtype=float)

    down_count = 0
    total_count = 0

    for r in returns:
        if np.isnan(r):
            continue

        total_count += 1

        if r < 0:
            down_count += 1

    return down_count / total_count


def p_down_given_down(returns):
    returns = np.asarray(returns, dtype=float)

    today_down_count = 0
    tomorrow_down_count = 0

    for i in range(len(returns) - 1):
        today = returns[i]
        tomorrow = returns[i + 1]

        if np.isnan(today) or np.isnan(tomorrow):
            continue

        if today < 0:
            today_down_count += 1

            if tomorrow < 0:
                tomorrow_down_count += 1

    if today_down_count == 0:
        return np.nan

    return tomorrow_down_count / today_down_count


def p_down_given_big_drop(returns, threshold=-0.02):
    returns = np.asarray(returns, dtype=float)

    big_drop_count = 0
    tomorrow_down_count = 0

    for i in range(len(returns) - 1):
        today = returns[i]
        tomorrow = returns[i + 1]

        if np.isnan(today) or np.isnan(tomorrow):
            continue

        if today < threshold:
            big_drop_count += 1

            if tomorrow < 0:
                tomorrow_down_count += 1

    if big_drop_count == 0:
        return np.nan, 0

    probability = tomorrow_down_count / big_drop_count

    return probability, big_drop_count


def expected_present_value(cash_flows, rate, survival_prob):
    epv = 0.0

    for i, cash_flow in enumerate(cash_flows):
        year = i + 1

        survival_to_year = survival_prob ** i

        discounted_cash_flow = cash_flow / ((1 + rate) ** year)

        epv += survival_to_year * discounted_cash_flow

    return epv


def main():

    simulated_dice = simulate_dice(100_000)
    exact_dice = 0.6

    print("Q1 simulated =", simulated_dice)
    print("Q1 exact =", exact_dice)
    print("Q1 difference =", simulated_dice - exact_dice)


    
    simulated_coins = simulate_coins(100_000)

    print("Q2 simulated =", simulated_coins)
    print("Q2 exact = 1.5")


    trial_counts = [100, 1_000, 10_000, 100_000]
    estimates = []

    for trials in trial_counts:
        estimate = simulate_coins(trials, seed=0)
        estimates.append(estimate)

        print(
            f"Q3 {trials:,} trials: {estimate:.4f}"
        )

    plt.plot(
        trial_counts,
        estimates,
        marker="o"
    )

    plt.axhline(
        1.5,
        linestyle="--",
        label="Exact = 1.5"
    )

    plt.xscale("log")
    plt.xlabel("Number of trials")
    plt.ylabel("Estimated payout")
    plt.title("Simulation convergence")
    plt.legend()

    plt.show()


    spy = yf.Ticker("SPY").history(
        period="10y",
        auto_adjust=False
    )

    closes = spy["Close"]

    returns = closes.pct_change().dropna()

    print("Q4 trading days =", len(returns))
    print("Q4 mean daily return =", returns.mean())

    
    prob_down = p_down(returns)

    prob_after_down = p_down_given_down(
        returns
    )

    returns_array = np.asarray(
        returns,
        dtype=float
    )

    down_condition_count = 0

    for i in range(len(returns_array) - 1):
        if returns_array[i] < 0:
            down_condition_count += 1

    print("Q5 P(down) =", prob_down)
    print("Q5 count =", len(returns))

    print(
        "Q5 P(down tomorrow | down today) =",
        prob_after_down
    )

    print(
        "Q5 conditional count =",
        down_condition_count
    )


    
    prob_big_drop, big_drop_count = (
        p_down_given_big_drop(returns)
    )

    print(
        "Q6 P(down tomorrow | >2% drop today) =",
        prob_big_drop
    )

    print(
        "Q6 count =",
        big_drop_count
    )


    
    epv = expected_present_value(
        [10, 10, 10],
        0.10,
        0.5
    )

    print("Q7 EPV =", epv)


if __name__ == "__main__":
    main()