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

## Reproduce

```bash
python backtest.py trader baseline_trader          # local backtest (--match worse for stricter fills)
python research/fit.py --write                     # refit + embed coefficients (add new days in fit.main)
python research/evaluate.py                        # cross-day honest evaluation
python research/grid_product.py                    # execution-parameter grid
```
