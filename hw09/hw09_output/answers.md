# Homework 9 — Volatility, tails, and value at risk

**Data caveat.** The sandbox could not reach Yahoo Finance, so these numbers come from `hw4_output/downloaded_closes.csv`, which is the **S&P 500 total-return index (`^SP500TR`)**, not SPY. The window is 2006-10-09 to 2026-10-07 (5,029 daily returns). Its volatility matches SPY's expected range, but the figures should be regenerated with SPY on your machine:

```bash
python hw09_starter.py
```

## Volatility and scaling

**Q1.** Daily volatility is 1.2312%. Scaled to annual by √252 it is **19.54%**.

**Q2.** Directly measured annual volatility from 19 full calendar-year returns is **17.77%**. The ratio of direct to scaled is **0.909**. The direct figure is lower.

Days are not independent here. Lag-1 autocorrelation is −0.123 and the variance ratio VR(5) is 0.796. Both say daily moves partly reverse, so variance accumulates more slowly than √time predicts, and the scaled figure overstates annual risk. The direction matches what I found in hw3: P(down tomorrow | down today) = 0.429, below the unconditional P(down) = 0.449. I did not have hw3's printed output saved, so I recomputed the comparison here rather than quote it. Nineteen annual observations is a small sample, so the direct figure is itself noisy.

**Q3.** Rolling 60-day volatility, annualised (plot: `q3_rolling_vol.png`):
- Highest: **73.99%** on **2008-12-08**
- Lowest: **4.97%** on **2017-11-15**
- Largest spike 1: 2008-12-08 (73.99%), the global financial crisis
- Largest spike 2: **2020-05-18** (63.60%), the COVID crash

The plot shows clustering. The calm stretch around 2017 sits between spikes, and each spike decays over months rather than volatility wandering randomly.

## The shape of returns

**Q4.** Skewness is **−0.205**. Excess kurtosis is **+12.54**. There are **37** days beyond four standard deviations (two-sided). Plot: `q4_return_histogram.png`.

**Q5.** Under normality over 5,029 days, the expected count of four-sigma days is **0.319**. Actual is **37**. The ratio is **116×**. This is the number to remember: a normal distribution makes a four-sigma day roughly once every 16,000 trading days, and this sample had 37.

**Q6.** Loss and the gain needed to get back to even:

| Loss | Gain needed |
|---|---|
| −10% | +11.1% |
| −25% | +33.3% |
| −50% | +100.0% |
| −75% | +300.0% |
| −90% | +900.0% |

Losses and gains are not symmetric. Each additional percentage point of loss needs more recovery than the last, so a deep drawdown can take years to repair. This is why maximum drawdown matters more than average return.

## Monte Carlo and value at risk

GBM inputs estimated from the data: arithmetic drift **12.60%**, volatility **19.54%**. Starting value $100,000, 10,000 one-year paths.

**Q7.** One-year final value:

| Statistic | Value |
|---|---|
| Mean | $113,696 |
| Median | $111,581 |
| 5th percentile | $81,150 |
| 95th percentile | $153,687 |

An investor should care about the **median** (or more precisely the distribution of the outcome they will actually live through), not the mean. The mean is an average of exponentially compounded outcomes, and a few very large winners pull it up. The median is what a typical investor experiences. The gap between them is the volatility drag, roughly exp(−σ²T/2) ≈ 2% a year at this volatility. Averaging returns across lucky and unlucky paths overstates what a single investor will earn.

**Q8.** One-year VaR and CVaR at 95% under GBM:
- VaR: **$18,850 loss (18.85%)**
- CVaR: **$24,933 loss (24.93%)**

**Q9.** Side by side (bootstrap resamples actual daily returns, 252 per path):

| Method | VaR $ | VaR % | CVaR $ | CVaR % |
|---|---|---|---|---|
| GBM (normal) | 18,850 | 18.85% | 24,933 | 24.93% |
| Bootstrap (actual) | 19,059 | 19.06% | 25,505 | 25.51% |

## Written

**Q10a. Which VaR is larger, and why?** The bootstrap VaR (19.06%) is slightly larger than the GBM VaR (18.85%). The difference is about 0.2 percentage points, which is within Monte Carlo noise for 10,000 paths: the standard error of a 5% quantile here is roughly 0.4 points. So I would not read a real difference into it. The bootstrap does carry the real fat tails and negative skew of daily returns, which the normal model lacks, but resampling 252 days independently also drops volatility clustering, and summing 252 draws pulls much of the daily tail shape back toward normal (the CLT). Most of the gap is sampling noise and the drift estimate.

**Q10b. What VaR does not tell you, and why CVaR exists.** VaR at 95% says that losses will exceed the threshold about one year in twenty. It says nothing about how large the loss is beyond that threshold. A 5% chance of losing 19% and a 5% chance of losing 60% produce the same VaR. CVaR averages the losses in that tail (here 24.9% against 18.9%), so it reports how bad the bad case is.

That is also why VaR can be gamed. An institution can keep its VaR below a limit by holding positions that rarely lose a little and occasionally lose catastrophically, since those losses sit beyond the 95% line where VaR cannot see them. CVaR, which averages the tail, is harder to manipulate this way.

## Check

- Daily SPY-range volatility: 1.23% (within 0.7–1.3%) ✓
- Annualised: 19.54% (within 12–20%) ✓
- Lose 50% → +100% needed ✓
- CVaR worse than VaR in both methods ✓
- Self-check: 9 of 9 pass (`check_hw09.py`)
