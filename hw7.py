"""Algory QI HW7: predicting SPY's next-day return with scikit-learn.

Run:  python hw7.py                 (downloads 15 years of SPY from Yahoo)
      python hw7.py --csv FILE.csv  (offline: CSV with Date, Close, Volume)

Convention: row t holds features computed from closes/volumes up to and
including day t's close. The label is the return from close t to close t+1.
So every feature is known the morning of day t+1, before the return being
predicted begins.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hw4 import WINDOW, backtest, metrics  # Homework 4 backtester

OUT = Path(__file__).resolve().parent / "hw7_output"
FEATURES = ["ret_1", "ret_5", "ret_20", "ret_60", "vol_20", "px_vs_ma50",
            "volume_vs_avg20"]
SEED = 0


# ---------------------------------------------------------------- Q1: data
def load_prices(csv=None):
    if csv:
        df = pd.read_csv(csv, parse_dates=["Date"], index_col="Date")
    else:
        import yfinance as yf
        end = pd.Timestamp.today().normalize()  # exclusive: drops today's bar
        start = end - pd.DateOffset(years=15)
        raw = yf.download("SPY", start=start, end=end, auto_adjust=True,
                          progress=False)
        if raw.empty:
            raise RuntimeError("Download failed; rerun or use --csv.")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Close", "Volume"]]
    df = df[["Close", "Volume"]].sort_index().dropna()
    if df.index.duplicated().any() or (df["Close"] <= 0).any():
        raise ValueError("Duplicate dates or nonpositive prices.")
    return df


def build_features(df):
    c, v = df["Close"], df["Volume"]
    daily = c.pct_change()
    X = pd.DataFrame(index=df.index)
    X["ret_1"] = c / c.shift(1) - 1
    X["ret_5"] = c / c.shift(5) - 1
    X["ret_20"] = c / c.shift(20) - 1
    X["ret_60"] = c / c.shift(60) - 1
    X["vol_20"] = daily.rolling(20).std() * np.sqrt(252)
    X["px_vs_ma50"] = c / c.rolling(50).mean() - 1
    X["volume_vs_avg20"] = v / v.rolling(20).mean()
    # Label: next day's return. shift(-1) looks forward -- only in the label.
    X["label"] = c.shift(-1) / c - 1
    return X.dropna()


def hand_check(df, table, emit):
    """Recompute one row with explicit positional slicing, then confirm the
    row is unchanged when every later day is deleted from the raw data."""
    date = table.index[len(table) // 2]
    i = df.index.get_loc(date)
    c = df["Close"].to_numpy()
    v = df["Volume"].to_numpy()
    past = c[: i + 1]                       # closes up to and including day t
    r = np.diff(c[i - 20: i + 1]) / c[i - 20: i]   # last 20 daily returns
    manual = {
        "ret_1": past[-1] / past[-2] - 1,
        "ret_5": past[-1] / past[-6] - 1,
        "ret_20": past[-1] / past[-21] - 1,
        "ret_60": past[-1] / past[-61] - 1,
        "vol_20": r.std(ddof=1) * np.sqrt(252),
        "px_vs_ma50": past[-1] / past[-50:].mean() - 1,
        "volume_vs_avg20": v[i] / v[i - 19: i + 1].mean(),
    }
    truncated = build_features(df.iloc[: i + 2])  # keep t+1 only for label
    emit(f"\nHand check for row {date.date()} (raw data index {i}):")
    emit(f"  {'feature':<16}{'table':>12}{'by hand':>12}{'truncated':>12}")
    for f in FEATURES:
        a, b, t = table.at[date, f], manual[f], truncated.at[date, f]
        emit(f"  {f:<16}{a:>12.6f}{b:>12.6f}{t:>12.6f}")
        assert np.isclose(a, b) and np.isclose(a, t), f
    lab = c[i + 1] / c[i] - 1
    assert np.isclose(table.at[date, "label"], lab)
    emit(f"  label = close[{df.index[i+1].date()}]/close[{date.date()}]-1 = "
         f"{lab:.6f}")
    emit("  All features match; deleting every day after t leaves them "
         "unchanged, so none uses future data.")


# --------------------------------------------------------------- Q2: split
def split(table):
    n = len(table)
    a, b = int(n * 0.6), int(n * 0.8)
    return table.iloc[:a], table.iloc[a:b], table.iloc[b:]   # no shuffling


# -------------------------------------------------------------- Q3: models
def make_model(name, param=None):
    if name == "Linear":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=param or 1.0))
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=300, max_depth=param,
                                     min_samples_leaf=1, n_jobs=-1,
                                     random_state=SEED)
    if name == "GradientBoosting":
        return GradientBoostingRegressor(n_estimators=param or 100,
                                         learning_rate=0.1, max_depth=3,
                                         random_state=SEED)
    raise ValueError(name)


GRIDS = {  # Q4: one hyperparameter each
    "Ridge": ("alpha", list(np.logspace(-2, 5, 15))),
    "RandomForest": ("max_depth", [1, 2, 3, 4, 5, 6, 8, 10, 15, None]),
    "GradientBoosting": ("n_estimators", [5, 10, 20, 40, 60, 100, 200, 400]),
}


def fit_score(model, train, val):
    model.fit(train[FEATURES], train["label"])
    return (r2_score(train["label"], model.predict(train[FEATURES])),
            r2_score(val["label"], model.predict(val[FEATURES])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="offline price file (Date, Close, Volume)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    log = []

    def emit(s=""):
        print(s)
        log.append(str(s))

    # Q1
    df = load_prices(args.csv)
    df.to_csv(OUT / "spy_prices.csv")
    emit(f"Q1. Raw SPY data: {df.index[0].date()} to {df.index[-1].date()}, "
         f"{len(df)} days")
    table = build_features(df)
    emit(f"Feature table shape: {table.shape}")
    emit(f"Columns: {list(table.columns)}")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        emit(table.head(3).to_string())
    hand_check(df, table, emit)

    # Q2
    train, val, test = split(table)
    emit("\nQ2. Chronological split (no shuffling):")
    for name, part in (("Train", train), ("Validation", val), ("Test", test)):
        emit(f"  {name:<11}{part.index[0].date()} to {part.index[-1].date()}"
             f"  rows={len(part)}")
    assert train.index[-1] < val.index[0] and val.index[-1] < test.index[0]

    # Q3
    emit("\nQ3. Default models, R-squared:")
    emit(f"  {'model':<18}{'train':>10}{'validation':>12}")
    for name in ("Linear", "Ridge", "RandomForest", "GradientBoosting"):
        tr, va = fit_score(make_model(name), train, val)
        emit(f"  {name:<18}{tr:>10.4f}{va:>12.4f}")
    emit("  (RandomForest: max_depth=None; GradientBoosting: 100 trees, "
         "depth 3, lr 0.1; Ridge alpha=1)")

    # Q4
    emit("\nQ4. Tuning on the validation set only:")
    best = {"Linear": (None, fit_score(make_model("Linear"), train, val)[1])}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (name, (pname, grid)) in zip(axes, GRIDS.items()):
        scores = [fit_score(make_model(name, p), train, val)[1] for p in grid]
        k = int(np.argmax(scores))
        best[name] = (grid[k], scores[k])
        xs = [p if p is not None else 30 for p in grid]
        ax.plot(xs, scores, marker="o")
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.scatter([xs[k]], [scores[k]], color="red", zorder=3, label="chosen")
        if name == "Ridge":
            ax.set_xscale("log")
        if name == "RandomForest":
            ax.set_xticks(xs, [str(p) for p in grid])
        ax.set(title=name, xlabel=pname, ylabel="Validation R-squared")
        ax.legend()
        ax.grid(alpha=0.25)
        shown = f"{grid[k]:.4g}" if isinstance(grid[k], float) else grid[k]
        emit(f"  {name:<18}{pname}={shown}  validation R2={scores[k]:.5f}")
    fig.tight_layout()
    fig.savefig(OUT / "q4_tuning.png", dpi=150)
    plt.close(fig)

    # Q5
    best_name = max(best, key=lambda m: best[m][1])
    model = make_model(best_name, best[best_name][0])
    model.fit(train[FEATURES], train["label"])          # train set only
    pred = pd.Series(model.predict(test[FEATURES]), index=test.index)
    emit(f"\nQ5. Best by validation: {best_name} "
         f"(param={best[best_name][0]}). Run once on test:")
    emit(f"  Test R-squared: {r2_score(test['label'], pred):.5f}")
    emit(f"  Correlation(predicted, actual): "
         f"{np.corrcoef(pred, test['label'])[0, 1]:.4f}")
    emit(f"  Predictions >0 on {np.mean(pred > 0):.1%} of test days; "
         f"sign accuracy {np.mean(np.sign(pred) == np.sign(test['label'])):.1%}")

    # Q6: HW4 backtester. Its position on step i comes from the WINDOW closes
    # ending at p[i-1] and earns p[i-1] -> p[i]. Feed it closes from WINDOW-1
    # days before the first test date through the day after the last one, so
    # step i's window ends on test date t and it earns exactly test's label.
    close = df["Close"]
    s = close.index.get_loc(test.index[0]) - (WINDOW - 1)
    e = close.index.get_loc(test.index[-1]) + 1
    data = close.iloc[s: e + 1]
    signal = iter(zip(test.index, pred > 0))

    def model_rule(window):
        date, go = next(signal)
        assert window[-1] == close.at[date]   # window ends on the signal date
        return bool(go)

    res = backtest(data, model_rule)
    assert np.allclose(res["index_returns"], test["label"])
    sm, bm = metrics(res, "Strategy"), metrics(res, "Index")
    emit(f"\nQ6. Backtest on test set ({test.index[0].date()} to "
         f"{data.index[-1].date()}), HW4 backtester, no costs, no cash yield:")
    emit(f"  {'':<14}{'annualised':>12}{'max drawdown':>14}{'invested':>10}")
    emit(f"  {'Model rule':<14}{sm['annual']:>12.2%}{sm['drawdown']:>14.2%}"
         f"{sm['invested']:>10.1%}")
    emit(f"  {'Buy and hold':<14}{bm['annual']:>12.2%}{bm['drawdown']:>14.2%}"
         f"{1:>10.1%}")
    fig, ax = plt.subplots(figsize=(10, 5))
    res["equity"].rename(columns={"Index": "Buy and hold"}).plot(ax=ax)
    ax.set(title=f"Test set: {best_name} long/cash rule vs buy and hold",
           ylabel="Value (starts at 1)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "q6_backtest.png", dpi=150)
    plt.close(fig)

    # Q7: both views, so the answer is available whichever model won.
    emit("\nQ7. Feature reliance")
    lin = make_model("Linear").fit(train[FEATURES], train["label"])
    coefs = pd.Series(lin[-1].coef_, index=FEATURES)
    emit("  Linear regression on standardised features (return per 1 s.d.), "
         "ranked by |coef|:")
    for f, val_ in coefs.reindex(coefs.abs().sort_values(ascending=False)
                                 .index).items():
        emit(f"    {f:<18}{val_:+.6f}")
    tree_name = best_name if best_name in ("RandomForest", "GradientBoosting") \
        else max(("RandomForest", "GradientBoosting"), key=lambda m: best[m][1])
    tree = make_model(tree_name, best[tree_name][0]).fit(train[FEATURES],
                                                          train["label"])
    imp = pd.Series(tree.feature_importances_, index=FEATURES)
    emit(f"  {tree_name} (tuned) impurity importances, ranked:")
    for f, val_ in imp.sort_values(ascending=False).items():
        emit(f"    {f:<18}{val_:.4f}")
    if best_name in ("Linear", "Ridge"):
        rc = pd.Series(model[-1].coef_, index=FEATURES)
        emit(f"  Chosen {best_name} model's standardised coefficients:")
        for f, val_ in rc.reindex(rc.abs().sort_values(ascending=False)
                                  .index).items():
            emit(f"    {f:<18}{val_:+.6f}")

    (OUT / "results.txt").write_text("\n".join(log) + "\n")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
