"""
Fit per-product Ridge models for the Round 1 trader and embed them in trader.py.

    python fit_ridge.py --data-dir round_1            # report only
    python fit_ridge.py --data-dir round_1 --write    # also rewrite COEFS in trader.py

Features are produced by replaying every tick through trader.update_features, i.e. the
exact code that runs live, so there is no train/live feature mismatch.

Validation: expanding-window walk-forward by day (train on all earlier days, test on the
next unseen day). Targets are mid(t+h) - mid(t) computed within a day, so no target window
crosses a day boundary and no extra purge is needed. Alpha is chosen on the pooled OOS
correlation; that makes the reported OOS number mildly optimistic (alpha is one scalar
picked from a grid, so the bias is small). A horizon whose OOS correlation is below
--min-corr is written with zero weights, and the trader then ignores it.

Dependencies: numpy, pandas (no sklearn).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import trader as T  # noqa: E402

HORIZONS = {"h10": 10, "h50": 50}
ALPHAS = np.logspace(0, 7, 15)


def _day(path):
    return int(re.search(r"day_(-?\d+)", path).group(1))


def load(data_dir):
    def read_all(pattern):
        files = sorted(glob.glob(os.path.join(data_dir, pattern)), key=_day)
        if not files:
            raise SystemExit(f"No files matching {pattern} in {data_dir}")
        out = []
        for fp in files:
            d = pd.read_csv(fp, sep=";")
            d["day"] = _day(fp)  # trades files have no day column
            out.append(d)
        return pd.concat(out, ignore_index=True)

    return (read_all("prices_round_1_day_*.csv"),
            read_all("trades_round_1_day_*.csv"),
            read_all("observations_round_1_day_*.csv"))


def _level(row, side, i):
    p, v = row.get(f"{side}_price_{i}"), row.get(f"{side}_volume_{i}")
    if p is None or v is None or pd.isna(p) or pd.isna(v) or v == 0:
        return None
    return (float(p), abs(float(v)))


def replay(prices, trades, obs):
    """Run every tick of every day through the live feature engine. Returns a DataFrame."""
    price_rows = {}
    for r in prices.to_dict("records"):
        price_rows.setdefault((r["day"], r["timestamp"]), {})[r["product"]] = r
    trade_rows = {}
    for r in trades.itertuples(index=False):
        trade_rows.setdefault((r.day, r.timestamp, r.symbol), []).append((float(r.price), abs(int(r.quantity))))
    obs_rows = {(r["day"], r["timestamp"]): r for r in obs.to_dict("records")}

    out = []
    for day in sorted(prices["day"].unique()):
        S = {}  # fresh state each day, exactly like a fresh live session
        prev_ts = None
        for ts in sorted(prices.loc[prices["day"] == day, "timestamp"].unique()):
            books = {}
            for p, r in price_rows[(day, ts)].items():
                bids = [lv for lv in (_level(r, "bid", i) for i in (1, 2, 3)) if lv]
                asks = [lv for lv in (_level(r, "ask", i) for i in (1, 2, 3)) if lv]
                books[p] = (bids, asks)
            tr = {} if prev_ts is None else {p: trade_rows.get((day, prev_ts, p), []) for p in T.PRODUCTS}
            o = obs_rows.get((day, ts))
            f = T.update_features(S, ts, books, tr, o)
            f["day"], f["timestamp"] = day, ts
            out.append(f)
            prev_ts = ts
    return pd.DataFrame(out)


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0)
    keep = sd > 1e-12
    sd = np.where(keep, sd, 1.0)
    Z = (X - mu) / sd
    Z[:, ~keep] = 0.0
    ym = y.mean()
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    beta = np.linalg.solve(A, Z.T @ (y - ym))
    beta[~keep] = 0.0
    w_raw = beta / sd
    return w_raw, ym - float(w_raw @ mu)


def corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def fit_product(df, p, h, min_train_days):
    feats = T.FEATURES[p]
    d = df.copy()
    d["y"] = d.groupby("day")[p + "_mid"].shift(-h) - d[p + "_mid"]
    d = d.dropna(subset=["y"] + feats)
    days = sorted(d["day"].unique())
    test_days = days[min_train_days:]
    if not test_days:
        raise SystemExit(f"Need more than {min_train_days} days for walk-forward")
    X, y, dy = d[feats].to_numpy(float), d["y"].to_numpy(float), d["day"].to_numpy()

    by_alpha = {}
    for a in ALPHAS:
        preds, ys = [], []
        for td in test_days:
            tr, te = dy < td, dy == td
            w, b = ridge_fit(X[tr], y[tr], a)
            preds.append(X[te] @ w + b)
            ys.append(y[te])
        by_alpha[a] = (corr(np.concatenate(preds), np.concatenate(ys)), preds, ys)
    best_a = max(by_alpha, key=lambda a: np.nan_to_num(by_alpha[a][0], nan=-1))
    best_r, preds, ys = by_alpha[best_a]
    per_day = [corr(pp, yy) for pp, yy in zip(preds, ys)]
    p_all, y_all = np.concatenate(preds), np.concatenate(ys)
    hit = float(np.mean(np.sign(p_all[y_all != 0]) == np.sign(y_all[y_all != 0])))
    te_mask = np.isin(dy, test_days)
    base_r = corr(d.loc[te_mask, p + "_outer_dev"].to_numpy(), y[te_mask])

    w, b = ridge_fit(X, y, best_a)  # final model on all days
    return {
        "alpha": float(best_a), "oos_r": best_r, "per_day": per_day, "hit": hit,
        "baseline_outer_dev_r": base_r, "test_days": [int(t) for t in test_days],
        "intercept": float(b), "w": {k: float(v) for k, v in zip(feats, w)},
    }


def write_coefs(coefs, path):
    src = open(path).read()
    block = ("# === BEGIN FITTED COEFFICIENTS (written by fit_ridge.py; raw feature space) ===\n"
             f"COEFS = {json.dumps(coefs, indent=1)}\n"
             "# === END FITTED COEFFICIENTS ===")
    new, n = re.subn(r"# === BEGIN FITTED COEFFICIENTS.*?# === END FITTED COEFFICIENTS ===",
                     lambda _: block, src, flags=re.S)
    if n != 1:
        raise SystemExit("Coefficient markers not found in trader.py")
    open(path, "w").write(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="round_1")
    ap.add_argument("--trader", default=os.path.join(HERE, "trader.py"))
    ap.add_argument("--min-train-days", type=int, default=3)
    ap.add_argument("--min-corr", type=float, default=0.02)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--report", default=os.path.join(HERE, "fit_report.json"))
    args = ap.parse_args()

    prices, trades, obs = load(args.data_dir)
    print(f"Loaded {len(prices)} book rows, {len(trades)} trades, {len(obs)} obs rows; replaying...")
    df = replay(prices, trades, obs)

    coefs, report = {}, {}
    for p in T.PRODUCTS:
        coefs[p] = {}
        for hk, h in HORIZONS.items():
            r = fit_product(df, p, h, args.min_train_days)
            use = r["oos_r"] == r["oos_r"] and r["oos_r"] >= args.min_corr
            coefs[p][hk] = {"intercept": r["intercept"] if use else 0.0,
                            "w": {k: round(v, 8) for k, v in r["w"].items() if use and v != 0.0}}
            report[f"{p}_{hk}"] = {k: v for k, v in r.items() if k not in ("w", "intercept")} | {"used": use}
            pd_str = " ".join(f"{x:+.3f}" for x in r["per_day"])
            print(f"{p:15s} {hk}: OOS r={r['oos_r']:+.4f} (outer_dev alone {r['baseline_outer_dev_r']:+.4f}) "
                  f"hit={r['hit']*100:.1f}% alpha={r['alpha']:.0e} {'USED' if use else 'DROPPED'}\n"
                  f"{'':15s}      per-day: {pd_str}")

    json.dump(report, open(args.report, "w"), indent=1)
    print(f"Report written to {args.report}")
    if args.write:
        write_coefs(coefs, args.trader)
        print(f"Coefficients embedded in {args.trader}")


if __name__ == "__main__":
    main()
