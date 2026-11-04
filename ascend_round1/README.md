# Ascend Algo – Round 1 trader

| File | Purpose |
| --- | --- |
| `trader.py` | Submission file. Uses only the standard library and `datamodel`. EWMA and history state live in `traderData`, updated in O(1) per tick. |
| `fit_ridge.py` | Replays the round CSVs through `trader.update_features`, runs a walk-forward Ridge fit by day, and writes `COEFS` into `trader.py`. |

## Usage

```bash
pip install numpy pandas
python fit_ridge.py --data-dir /path/to/round_1            # report only
python fit_ridge.py --data-dir /path/to/round_1 --write    # embed coefficients
```

Then submit `trader.py`. If `COEFS` is empty, the trader quotes around the Deep-MM mid (`mid + outer_dev`).

## Before submitting

1. Set `LIMITS` in `trader.py` from the round rules. The defaults of 50 are placeholders.
2. Check `_obs_from_state` against the round's `datamodel.py`. If live weather observations
   don't arrive in that shape, the weather features stay at 0 even though the model was trained with them.
3. Compare each product's OOS r in the report with the `outer_dev alone` baseline. If Ridge doesn't beat it,
   the extra features are only adding noise.
4. Tune `TAKE_EDGE`, `MAKE_EDGE`, `GAMMA`, `MAKE_SIZE`, and `HORIZON_WEIGHTS` in a backtester.
   They have not been tuned.
