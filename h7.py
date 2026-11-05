"""
Homework 7 - Predicting next-day SPY returns with machine learning.

Run:  python h7.py
Needs: yfinance, pandas, numpy, matplotlib, scikit-learn, and hw4.py in the
same folder (its backtester is reused for Section 6).
Outputs (plots + printed tables) are written to ./hw7_output/.

ASSUMPTIONS (flagged as the assignment asks)
  A1. "15 years" = calendar window ending at the last COMPLETED trading day
      (today's possibly-partial bar is excluded).
  A2. All price-based features AND the label use the ADJUSTED close (dividends
      and splits folded in), because the label is a total-return quantity.
      Volume is raw share volume (yfinance does not adjust it for dividends).
  A3. Row alignment: a row is indexed by trading day t.
        features(t) = computed from data up to and including close(t-1)
        label(t)    = close(t) / close(t-1) - 1
      i.e. the label is the "next day's" return relative to the information
      set, and the position for day t can be chosen before day t opens.
  A4. "Realized volatility" = sample std (ddof=1) of the last 20 daily simple
      returns, NOT annualized.
  A5. The 60/20/20 split is by row count of the final (NaN-free) feature table.
  A6. Every model is wrapped in a Pipeline(StandardScaler -> model) so the
      scaler is always fit on training data only. Scaling does not change tree
      models; it makes linear coefficients comparable (standardized).
  A7. No cross-validation is used; tuning uses the single chronological
      validation set, so nothing is ever shuffled. (RandomForest's bootstrap
      resamples rows within the training set only, which is part of the
      estimator, not a split of the data.)
  A8. Backtest: no transaction costs, slippage, taxes, or interest on cash;
      annualization uses 252 trading days (inherited from hw4.metrics).
"""

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

# Homework 4 backtester (hw4.py lives in the same repo folder).
from hw4 import backtest, metrics, always_hold, WINDOW

SEED = 42
TICKER = "SPY"
YEARS = 15
OUT = Path(__file__).resolve().parent / "hw7_output"
FEATURES = ["ret_1", "ret_5", "ret_20", "ret_60",
            "vol_20", "px_to_ma50", "volu_to_ma20"]


def header(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def make_model(estimator):
    """Scaler + model in one Pipeline -> scaler only ever sees training X."""
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


# =============================================================================
# 1. DATA: download, build features, verify no lookahead
# =============================================================================
def download_spy():
    import yfinance as yf
    today = datetime.now(ZoneInfo("America/New_York")).date()
    start = today.replace(year=today.year - YEARS)
    # `end` is exclusive -> today's unfinished bar is excluded (A1).
    raw = yf.download(TICKER, start=start.isoformat(), end=today.isoformat(),
                      interval="1d", auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo returned no SPY data; retry later.")
    if isinstance(raw.columns, pd.MultiIndex):        # newer yfinance
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Adj Close", "Volume"]].rename(
        columns={"Adj Close": "adj_close", "Volume": "volume"})
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.dropna().astype(float).sort_index()
    assert not df.index.has_duplicates and (df["adj_close"] > 0).all()
    return df


def build_features(raw):
    px, vol = raw["adj_close"], raw["volume"]
    daily_ret = px.pct_change()            # ret on day t uses close(t), close(t-1)

    # Every quantity below is first computed "as of close(t)", then shifted by
    # one row so that the value stored on row t only uses data through t-1.
    as_of_close = pd.DataFrame({
        "ret_1": px.pct_change(1),
        "ret_5": px.pct_change(5),
        "ret_20": px.pct_change(20),
        "ret_60": px.pct_change(60),
        "vol_20": daily_ret.rolling(20).std(),
        "px_to_ma50": px / px.rolling(50).mean(),
        "volu_to_ma20": vol / vol.rolling(20).mean(),
    })
    table = as_of_close.shift(1)
    table["target"] = daily_ret            # NOT shifted: this is day t's return
    return table.dropna()


def verify_no_lookahead(raw, table, date=None):
    """Recompute every feature for one row by hand from raw positional data."""
    px = raw["adj_close"].to_numpy()
    vol = raw["volume"].to_numpy()
    date = table.index[len(table) // 2] if date is None else pd.Timestamp(date)
    i = raw.index.get_loc(date)            # position of day t in raw data
    p = i - 1                               # position of the PREVIOUS close
    rets = px[1:] / px[:-1] - 1             # rets[k] = return on raw day k+1
    manual = {
        "ret_1": px[p] / px[p - 1] - 1,
        "ret_5": px[p] / px[p - 5] - 1,
        "ret_20": px[p] / px[p - 20] - 1,
        "ret_60": px[p] / px[p - 60] - 1,
        "vol_20": np.std(rets[p - 20:p], ddof=1),   # returns on days p-19..p
        "px_to_ma50": px[p] / px[p - 49:p + 1].mean(),
        "volu_to_ma20": vol[p] / vol[p - 19:p + 1].mean(),
        "target": px[i] / px[i - 1] - 1,
    }
    print(f"\nVerification row: {date.date()} (row t). Features must only use "
          f"data through {raw.index[p].date()} (t-1).")
    print(f"Latest raw date touched by any feature: {raw.index[p].date()}; "
          f"label uses close of {raw.index[i].date()}.")
    check = pd.DataFrame({"pipeline": table.loc[date, FEATURES + ["target"]],
                          "manual": pd.Series(manual)})
    check["abs_diff"] = (check["pipeline"] - check["manual"]).abs()
    print(check.to_string(float_format=lambda x: f"{x: .10f}"))
    assert np.allclose(check["pipeline"], check["manual"], rtol=0, atol=1e-12)
    print("All features match manual recomputation -> no lookahead.")


# =============================================================================
# 2. CHRONOLOGICAL SPLIT 60 / 20 / 20
# =============================================================================
def chrono_split(table):
    n = len(table)
    a, b = int(n * 0.6), int(n * 0.8)
    parts = {"train": table.iloc[:a], "val": table.iloc[a:b], "test": table.iloc[b:]}
    for name, d in parts.items():
        print(f"{name:>5}: {d.index[0].date()} -> {d.index[-1].date()}  "
              f"rows = {len(d):,}")
    assert parts["train"].index[-1] < parts["val"].index[0] < parts["test"].index[0]
    return parts


def xy(df):
    return df[FEATURES], df["target"]


# =============================================================================
# 3. BASELINE MODELS
# =============================================================================
def baseline_models():
    return {
        "LinearRegression": make_model(LinearRegression()),
        "Ridge": make_model(Ridge(alpha=1.0)),
        "RandomForest": make_model(RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1)),
        "GradientBoosting": make_model(GradientBoostingRegressor(random_state=SEED)),
    }


def fit_baselines(parts):
    Xtr, ytr = xy(parts["train"])
    Xva, yva = xy(parts["val"])
    rows = []
    for name, model in baseline_models().items():
        model.fit(Xtr, ytr)
        rows.append({"model": name,
                     "train_R2": r2_score(ytr, model.predict(Xtr)),
                     "val_R2": r2_score(yva, model.predict(Xva))})
    res = pd.DataFrame(rows).set_index("model")
    print(res.to_string(float_format=lambda x: f"{x: .5f}"))
    return res


# =============================================================================
# 4. HYPERPARAMETER TUNING ON VALIDATION SET ONLY
# =============================================================================
def tune(parts):
    Xtr, ytr = xy(parts["train"])
    Xva, yva = xy(parts["val"])

    grids = {
        "Ridge": ("alpha", list(np.logspace(-3, 5, 25)),
                  lambda v: Ridge(alpha=v)),
        "RandomForest": ("max_depth", [1, 2, 3, 4, 5, 6, 8, 10, 15, None],
                         lambda v: RandomForestRegressor(
                             n_estimators=300, max_depth=v, min_samples_leaf=5,
                             random_state=SEED, n_jobs=-1)),
        "GradientBoosting": ("n_estimators", [5, 10, 25, 50, 100, 200, 400],
                             lambda v: GradientBoostingRegressor(
                                 n_estimators=v, learning_rate=0.05, max_depth=2,
                                 random_state=SEED)),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    best = {}
    for ax, (name, (param, grid, build)) in zip(axes, grids.items()):
        scores = []
        for v in grid:
            m = make_model(build(v)).fit(Xtr, ytr)
            scores.append(r2_score(yva, m.predict(Xva)))
        k = int(np.argmax(scores))
        best[name] = {"param": param, "value": grid[k], "val_R2": scores[k],
                      "build": build}
        print(f"{name:<17} best {param} = {grid[k]!s:<10} validation R2 = {scores[k]: .5f}")

        if name == "Ridge":
            ax.semilogx(grid, scores, marker="o")
        else:   # categorical x-axis so max_depth=None can be shown
            ax.plot(range(len(grid)), scores, marker="o")
            ax.set_xticks(range(len(grid)), [str(g) for g in grid])
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set(title=f"{name}: validation R² vs {param}", xlabel=param,
               ylabel="validation R²")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "q4_tuning.png", dpi=150)
    plt.close(fig)
    print(f"Plot saved: {OUT / 'q4_tuning.png'}")
    return best


# =============================================================================
# 5. MODEL SELECTION, REFIT, SINGLE TEST EVALUATION
# =============================================================================
def select_and_test(parts, baseline, best):
    Xtr, ytr = xy(parts["train"])
    Xva, yva = xy(parts["val"])
    Xte, yte = xy(parts["test"])

    # Candidates: untuned LinearRegression (it has no hyperparameter) + the
    # three tuned models. Selection uses validation R2 only.
    cands = {"LinearRegression": (baseline.loc["LinearRegression", "val_R2"],
                                  lambda: LinearRegression())}
    for name, b in best.items():
        cands[name] = (b["val_R2"], (lambda b=b: b["build"](b["value"])))
    table = pd.Series({k: v[0] for k, v in cands.items()}, name="val_R2")
    print(table.sort_values(ascending=False).to_string(float_format=lambda x: f"{x: .5f}"))
    winner = table.idxmax()
    print(f"\nSelected model: {winner}"
          + (f" ({best[winner]['param']} = {best[winner]['value']})" if winner in best else ""))

    # REFIT CHOICE: train + validation.
    # The hyperparameter is now frozen, so validation data has done its job.
    # Refitting on train+val uses 25% more data and, crucially for markets,
    # the most recent regime right before the test period. The test set is
    # still untouched, so the test score remains an honest out-of-sample number.
    trval = pd.concat([parts["train"], parts["val"]])
    final = make_model(cands[winner][1]()).fit(*xy(trval))
    print("Refit on TRAIN + VALIDATION "
          f"({trval.index[0].date()} -> {trval.index[-1].date()}, {len(trval):,} rows).")

    # ---- the ONE AND ONLY evaluation on the test set ----
    pred = pd.Series(final.predict(Xte), index=Xte.index, name="pred")
    test_r2 = r2_score(yte, pred)
    corr = np.corrcoef(pred, yte)[0, 1]
    print(f"TEST R2 = {test_r2: .5f}")
    print(f"TEST Pearson corr(predicted, actual) = {corr: .5f}")
    print(f"Predictions > 0 on {np.mean(pred > 0):.1%} of test days; "
          f"pred std = {pred.std():.6f} vs actual std = {yte.std():.6f}")
    return winner, final, pred


# =============================================================================
# 6. STRATEGY BACKTEST WITH THE HW4 BACKTESTER
# =============================================================================
def run_backtest(raw, pred):
    """hw4.backtest(prices, strategy) calls strategy(window) for i = WINDOW..n-1
    with window = prices[i-WINDOW:i] (ends at close(i-1)) and earns
    prices[i]/prices[i-1]-1 if strategy returns True. So the decision for
    day i only sees closes through i-1 -> decided before day i's return.

    The hw4 strategy interface only receives prices, not dates, so we feed the
    precomputed model predictions in order and assert on every call that the
    window's last close is exactly close(t-1) for the test day t being traded.
    """
    px = raw["adj_close"]
    first = px.index.get_loc(pred.index[0])
    last = px.index.get_loc(pred.index[-1])
    # 60 warm-up closes before the first test day (only used as the window
    # the hw4 interface requires; our signal ignores them), then test closes.
    prices = px.iloc[first - WINDOW:last + 1]
    assert prices.index[WINDOW:].equals(pred.index), "test dates must be contiguous"

    signal = (pred > 0).to_numpy()
    prev_close = px.iloc[first - 1:last].to_numpy()       # close(t-1) per test day
    state = {"k": 0}

    def ml_strategy(window):
        k = state["k"]
        assert window[-1] == prev_close[k], "alignment error: lookahead risk"
        state["k"] += 1
        return bool(signal[k])

    strat = backtest(prices, ml_strategy)
    bh = backtest(prices, always_hold)
    assert state["k"] == len(pred)
    # Backtester's daily returns must equal our label -> same days, same returns.
    assert np.allclose(strat["index_returns"], (prices.iloc[WINDOW:] / prices.shift(1).iloc[WINDOW:] - 1).to_numpy())

    rows = {}
    for label, res in [("ML strategy (pred > 0)", strat), ("Buy & hold SPY", bh)]:
        m = metrics(res)
        rows[label] = {"annual_return": m["annual"], "max_drawdown": m["drawdown"],
                       "total_return": m["total"], "frac_days_invested": m["invested"]}
    out = pd.DataFrame(rows).T
    print(f"Test period traded: {pred.index[0].date()} -> {pred.index[-1].date()} "
          f"({len(pred):,} days)")
    print(out.to_string(float_format=lambda x: f"{x:.2%}"))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    pd.DataFrame({"ML strategy": strat["equity"]["Strategy"],
                  "Buy & hold": bh["equity"]["Strategy"]}).plot(ax=ax)
    ax.set(title="Section 6: test-set equity (start = 1)", ylabel="Equity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "q6_backtest.png", dpi=150)
    plt.close(fig)
    print(f"Plot saved: {OUT / 'q6_backtest.png'}")
    return out


# =============================================================================
# 7. FEATURE IMPORTANCE
# =============================================================================
def feature_importance(parts, best):
    Xtr, ytr = xy(parts["train"])
    Xva, yva = xy(parts["val"])

    # Tree model: whichever tuned tree scored higher on validation.
    tree_name = max(["RandomForest", "GradientBoosting"], key=lambda n: best[n]["val_R2"])
    b = best[tree_name]
    tree = make_model(b["build"](b["value"])).fit(Xtr, ytr)
    perm = permutation_importance(tree, Xva, yva, scoring="r2",
                                  n_repeats=30, random_state=SEED)
    tree_tbl = pd.DataFrame({
        "impurity_importance": tree.named_steps["model"].feature_importances_,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    }, index=FEATURES).sort_values("perm_importance_mean", ascending=False)
    tree_tbl.insert(0, "rank", range(1, len(FEATURES) + 1))
    print(f"Tree model: {tree_name} ({b['param']} = {b['value']}), fit on train; "
          "permutation importance measured on VALIDATION (drop in R2).")
    print(tree_tbl.to_string(float_format=lambda x: f"{x: .6f}"))

    # Linear model: tuned Ridge. Features were standardized inside the
    # pipeline, so each coefficient = change in predicted daily return for a
    # one-standard-deviation move in that feature -> directly comparable.
    rb = best["Ridge"]
    ridge = make_model(Ridge(alpha=rb["value"])).fit(Xtr, ytr)
    ols = make_model(LinearRegression()).fit(Xtr, ytr)
    lin_tbl = pd.DataFrame({
        "ridge_std_coef": ridge.named_steps["model"].coef_,
        "ols_std_coef": ols.named_steps["model"].coef_,
    }, index=FEATURES)
    lin_tbl["abs_ridge"] = lin_tbl["ridge_std_coef"].abs()
    lin_tbl = lin_tbl.sort_values("abs_ridge", ascending=False).drop(columns="abs_ridge")
    lin_tbl.insert(0, "rank", range(1, len(FEATURES) + 1))
    print(f"\nLinear model: Ridge (alpha = {rb['value']:.4g}) standardized coefficients, "
          "ranked by |coef| (OLS shown for comparison), fit on train.")
    print(lin_tbl.to_string(float_format=lambda x: f"{x: .7f}"))
    return tree_tbl, lin_tbl


# =============================================================================
def main(raw=None):
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 140)

    header("1. DATA")
    raw = download_spy() if raw is None else raw
    print(f"Raw {TICKER}: {raw.index[0].date()} -> {raw.index[-1].date()}, {len(raw):,} days")
    table = build_features(raw)
    print(f"Feature table shape: {table.shape}")
    print(f"Columns: {list(table.columns)}")
    print(table.head(3).to_string(float_format=lambda x: f"{x: .6f}"))
    verify_no_lookahead(raw, table)

    header("2. CHRONOLOGICAL SPLIT (60/20/20, no shuffling)")
    parts = chrono_split(table)

    header("3. BASELINE MODELS (train vs validation R2)")
    baseline = fit_baselines(parts)

    header("4. HYPERPARAMETER TUNING (validation set only)")
    best = tune(parts)

    header("5. MODEL SELECTION + SINGLE TEST EVALUATION")
    winner, final, pred = select_and_test(parts, baseline, best)

    header("6. STRATEGY BACKTEST (HW4 backtester, test set)")
    run_backtest(raw, pred)

    header("7. FEATURE IMPORTANCE")
    feature_importance(parts, best)


if __name__ == "__main__":
    main()
