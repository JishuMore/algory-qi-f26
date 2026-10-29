# AMBER / JADE trader

`trader.py` is the submission file (self-contained apart from `datamodel`). `baseline_trader.py` is the previous version, kept for comparison.

## What the data says (days -2, -1)

| Finding | Evidence | How it's used |
|---|---|---|
| Every book level = latent fair + one common iid shock (sd 1.55 AMBER / 1.02 JADE) | mid-diff lag-1 autocorr ≈ -0.27; averaging levels doesn't reduce the noise | Kalman filter fair + `noise = mid - fair` feature |
| Slow mean reversion to 1000 / 500 | ~60% of a deviation gone after 100 ticks | `dev` feature, OU in the filter |
| **Informed bot trades exactly 7 lots**, at least 20 ticks apart, same-sign repeats at exactly +20 | fwd 29-tick move after a 7-lot trade: AMBER +12 ± 5, JADE +10 ± 3 (10th pct still +6.5) | event-age buckets incl. a dedicated age-20 bucket; detection counts a 7-lot trade split between our quote and the book |
| **JADE = 500 + 0.5·(AMBER[t-25] − 1000) + G**, G independent OU | corr(ΔJ_t, ΔA_{t-25}) = 0.36, nothing at other lags; G ⟂ AMBER | horizon-specific `lead` = 0.5·(Ã_{t-25+h} − Ã_{t-25}), already observed for h ≤ 25 |
| No market trades in blackouts; prices jump 7–20 across them | 0 trades in 4 blackouts | no blackout quoting (it can't fill); filter coasts through gaps |
| Book volumes / spread width carry no information | ΔR² ≤ 0.0003 | not used |

## Results

Honest = each day traded with coefficients fit only on the other day.

| | day -2 | day -1 | total |
|---|---|---|---|
| baseline, original (in-sample) coefs | 193,446 | 191,964 | 385,410 |
| baseline, honest coefs | 191,572 | 188,803 | 380,374 |
| **new, honest coefs** | **195,661** | **193,170** | **388,831** |
| new, coefs fit on both days | 195,470 | 194,999 | 390,468 |

Out-of-sample forecast R² (h=10): AMBER 0.433 vs 0.428, JADE 0.605 vs 0.570.

Fill diagnostics (`research/markout.py`): about 95% of volume is taken, at about 3.4/lot spread on AMBER. Those trades earn it back by +30 ticks. More aggressive taking (longer horizons, multi-horizon "best rate" entry) and longer-horizon quoting were both tested and lost money or did nothing. Passive capacity is capped by bot flow.

## Live day 1 post-mortem (`data/my_trades_day_1.csv`)

Live: AMBER ≈ −7k, JADE ≈ +49k (marked at last fill), vs ~110k / ~85k per backtest day. The log was empty because the trader printed nothing, and the file only has our own fills, so the cause is inferred by replaying days -2/-1 under possible engine differences (`backtest.py --feed ... --fresh`):

| scenario (old trader) | AMBER/day | JADE/day |
|---|---|---|
| normal | +110k | +85k |
| **no informed events registered at all** | **−10k … −17k** | **+44k … +51k** ← matches live |
| no market trades (own fills still seen) | +30k … +52k | +54k |
| trades repeated until replaced (`sticky`) | −15k … −18k | −4k … +2k |
| new Trader instance every tick | −5k … −7k | +6k |

So live, the 7-lot informed signal never reached the model; JADE survived on the price-only AMBER→JADE lead. Fixes in `trader.py`:

* events placed by each trade's own `timestamp`, re-sent trades de-duplicated, side judged vs the mid at the trade's time, `qty >= 7`;
* fallback `COEF_NOEV` (event-free fit) if no informed trade is seen for 300 live ticks (expected gap ~17): zero-event scenario becomes AMBER +12…18k, JADE +62k;
* one compact log line per tick (`ts|mode|SYM pos mid m[qty@px/dt] o[n] e[(age,sign)]`) plus `NEW_NAME` when trades carry counterparty names.

New trader: normal / sticky / dup feeds 392k (identical), late 386k, no market trades 193k, zero events 155k.

## Live day 2 post-mortem (`data/live_log_day_2_tail.txt`)

PnL +101.7k, about half the previous run. The log shows market trades arriving normally (7-lot trades with `dt=-1`). The detected events were wrong: 36 of 45 in the excerpt came from our own fills. Live, bots fill our resting quotes in 8–15-lot chunks, which never happens in the historical data, and `own >= 7` turned each of those fills into a fake informed trade pointing against our own fill.

Fixes:
* informed trades = **market trades of exactly 7 lots only**; own fills are ignored;
* **taker-only** by default (`Trader.passive`). Our inside quote used to absorb the bot's 7 lots, so the signal never printed as a market trade. Taking only: 383k OOS, identical under both fill rules and every feed mode, so it no longer depends on how live bots fill resting quotes;
* the log now also prints own fills (`o[+qty@px/dt]`).

## Reproduce

```bash
python backtest.py trader baseline_trader          # local backtest (--match worse, --feed none|dup|late|sticky, --fresh)
python research/fit.py --write                     # refit + embed coefficients (add new days in fit.main)
python research/evaluate.py                        # cross-day honest evaluation
python research/grid_product.py                    # execution-parameter grid
```
