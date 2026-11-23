"""Can workload predict a running back's injury designation the following week?

Data   : nflreadpy weekly player stats + weekly injury reports, regular season 2015-2024.
Unit   : one row per RB per regular-season game he played (week w).
Label  : 1 if the RB is on the injury report for week w+1 with status
         Questionable / Doubtful / Out, else 0.
Features are all known at the end of week w (workload, history, age, and
optionally whether he is already listed in week w).

Splits (time-based, no shuffling across seasons):
    train 2016-2021  -> fit
    val   2022       -> choose tree depth / number of trees (early stopping)
    test  2023-2024  -> reported once, after all choices are fixed

Overfitting controls: shallow trees, strong min_child_weight / lambda,
small learning rate, early stopping on validation, a deliberately simple
logistic baseline, a shuffled-label control, and bootstrap CIs.
"""
import os

import nflreadpy as nfl
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

LOAD_SEASONS = list(range(2015, 2025))   # 2015 only feeds "prior season" features
TRAIN_SEASONS = list(range(2016, 2022))
VAL_SEASONS = [2022]
TEST_SEASONS = [2023, 2024]
QDO = ["Questionable", "Doubtful", "Out"]

FEATURES = [
    "carries", "targets", "touches_roll3", "season_touches", "career_touches",
    "career_games", "age", "prior_season_qdo_weeks", "listed_now",
]
NO_LISTING = [f for f in FEATURES if f != "listed_now"]
CORE = ["touches_roll3", "career_touches", "age", "prior_season_qdo_weeks", "listed_now"]

lines = []
def log(s=""):
    print(s)
    lines.append(str(s))

# ---------------------------------------------------------------- data
stats = nfl.load_player_stats(LOAD_SEASONS, summary_level="week").to_pandas()
stats = stats[(stats["season_type"] == "REG") & (stats["position"] == "RB")].copy()
stats["touches"] = stats["carries"].fillna(0) + stats["targets"].fillna(0)
stats = stats.rename(columns={"player_id": "gsis_id"})
stats = stats[["gsis_id", "season", "week", "carries", "targets", "touches"]]
stats = stats.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)

inj = nfl.load_injuries(LOAD_SEASONS).to_pandas()
inj = inj[inj["game_type"] == "REG"].copy()
inj["qdo"] = inj["report_status"].isin(QDO).astype(int)
inj_wk = inj.groupby(["gsis_id", "season", "week"], as_index=False)["qdo"].max()
qdo_lookup = {(r.gsis_id, r.season, r.week): r.qdo for r in inj_wk.itertuples()}

players = nfl.load_players().to_pandas()[["gsis_id", "birth_date"]].dropna(subset=["gsis_id"])
players["birth_year"] = pd.to_datetime(players["birth_date"], errors="coerce").dt.year
birth_year = players.drop_duplicates("gsis_id").set_index("gsis_id")["birth_year"]

# ---------------------------------------------------------------- features
g = stats.copy()
grp = g.groupby("gsis_id")
g["touches_roll3"] = grp["touches"].transform(lambda s: s.rolling(3, min_periods=1).sum())
g["season_touches"] = g.groupby(["gsis_id", "season"])["touches"].cumsum()
g["career_touches"] = grp["touches"].cumsum()
g["career_games"] = grp.cumcount() + 1
g["birth_year"] = g["gsis_id"].map(birth_year)
g["age"] = g["season"] - g["birth_year"]
g["age"] = g["age"].fillna(g["age"].median())

g["listed_now"] = [qdo_lookup.get((p, s, w), 0) for p, s, w in zip(g.gsis_id, g.season, g.week)]

# prior-season count of QDO report weeks (injury history, known before this season)
prior = inj_wk.groupby(["gsis_id", "season"])["qdo"].sum().rename("qdo_weeks").reset_index()
prior["season"] = prior["season"] + 1  # shift so it applies to the following season
g = g.merge(prior.rename(columns={"qdo_weeks": "prior_season_qdo_weeks"}),
            on=["gsis_id", "season"], how="left")
g["prior_season_qdo_weeks"] = g["prior_season_qdo_weeks"].fillna(0)

# label: on the week w+1 report with Q/D/O status
g["next_week"] = g["week"] + 1
g = g[g["next_week"] <= 18].copy()   # regular season ends at week 18
g["injured_next"] = [qdo_lookup.get((p, s, w), 0)
                     for p, s, w in zip(g.gsis_id, g.season, g.next_week)]

df = g[g["season"] >= 2016].copy()   # injury reports are complete from 2016 on
df[FEATURES + ["injured_next"]] = df[FEATURES + ["injured_next"]].astype(float)

tr = df[df.season.isin(TRAIN_SEASONS)]
va = df[df.season.isin(VAL_SEASONS)]
te = df[df.season.isin(TEST_SEASONS)]

log(f"RB-games: {len(df)}   train {len(tr)}  val {len(va)}  test {len(te)}")
log(f"Positive rate  train {tr.injured_next.mean():.3%}  val {va.injured_next.mean():.3%}  "
    f"test {te.injured_next.mean():.3%}")
log(f"Positives      train {int(tr.injured_next.sum())}  val {int(va.injured_next.sum())}  "
    f"test {int(te.injured_next.sum())}")
log("")

# ---------------------------------------------------------------- models
def xgb_params(depth):
    return dict(objective="binary:logistic", eval_metric="logloss", learning_rate=0.03,
                max_depth=depth, min_child_weight=30, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=5.0, seed=0, verbosity=0)

def fit_xgb(X, y, depth, Xv=None, yv=None, n_rounds=None):
    dtr = xgb.DMatrix(X, label=y)
    if n_rounds is None:
        dv = xgb.DMatrix(Xv, label=yv)
        booster = xgb.train(xgb_params(depth), dtr, num_boost_round=1000,
                            evals=[(dv, "val")], early_stopping_rounds=50, verbose_eval=False)
        return booster, booster.best_iteration + 1
    booster = xgb.train(xgb_params(depth), dtr, num_boost_round=n_rounds)
    return booster, n_rounds

def predict(booster, X):
    return booster.predict(xgb.DMatrix(X))

def auc(y, p):
    return roc_auc_score(y, p)

def boot_ci(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5])

# 1) choose tree depth on validation only
best = None
log("Validation (2022) model selection - XGBoost, all features:")
for depth in [1, 2, 3]:
    b, n = fit_xgb(tr[FEATURES].values, tr.injured_next.values, depth,
                   va[FEATURES].values, va.injured_next.values)
    pv = predict(b, va[FEATURES].values)
    ll = log_loss(va.injured_next, pv)
    log(f"  depth={depth}  rounds={n:3d}  val logloss={ll:.4f}  val AUC={auc(va.injured_next, pv):.3f}")
    if best is None or ll < best[0]:
        best = (ll, depth, n)
_, DEPTH, ROUNDS = best
log(f"  -> chosen depth={DEPTH}, rounds={ROUNDS}")
log("")

# 2) final fit on train+val with fixed settings, evaluate ONCE on test
trva = df[df.season.isin(TRAIN_SEASONS + VAL_SEASONS)]
results = {}

def evaluate(name, feats, model_kind):
    if model_kind == "xgb":
        b, _ = fit_xgb(trva[feats].values, trva.injured_next.values, DEPTH, n_rounds=ROUNDS)
        p = predict(b, te[feats].values)
    else:
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))
        lr.fit(trva[feats].values, trva.injured_next.values)
        p = lr.predict_proba(te[feats].values)[:, 1]
    y = te.injured_next.values
    results[name] = dict(auc=auc(y, p), pr=average_precision_score(y, p),
                         ll=log_loss(y, p), p=p, ci=boot_ci(y, p))
    r = results[name]
    lo, hi = r["ci"]
    log(f"{name:44s} AUC {r['auc']:.3f} [95% CI {lo:.3f}-{hi:.3f}]  "
        f"PR-AUC {r['pr']:.3f} (base {te.injured_next.mean():.3f})  logloss {r['ll']:.4f}")
    return r

log("TEST (2023-2024), fitted on 2016-2022:")
base_rate = te.injured_next.mean()
log(f"{'Base rate (predict the average)':44s} AUC 0.500  PR-AUC {base_rate:.3f}")
evaluate("Logistic, core features", CORE, "lr")
evaluate("XGBoost, all features", FEATURES, "xgb")
evaluate("XGBoost, no current-listing feature", NO_LISTING, "xgb")
log("")

# 3) single-feature AUCs on test (univariate signal)
log("Single-feature AUC on test (direction-agnostic max(AUC,1-AUC)):")
for f in FEATURES:
    a = auc(te.injured_next, te[f])
    log(f"  {f:26s} {max(a, 1 - a):.3f}  ({'higher' if a >= 0.5 else 'lower'} -> more injury)")
log("")

# 4) permutation importance on the all-features model (test set)
b_all, _ = fit_xgb(trva[FEATURES].values, trva.injured_next.values, DEPTH, n_rounds=ROUNDS)
base_auc = auc(te.injured_next, predict(b_all, te[FEATURES].values))
rng = np.random.default_rng(1)
log("Permutation importance (drop in test AUC when feature is shuffled, 20 repeats):")
imp = []
for f in FEATURES:
    drops = []
    for _ in range(20):
        Xp = te[FEATURES].values.copy()
        Xp[:, FEATURES.index(f)] = rng.permutation(Xp[:, FEATURES.index(f)])
        drops.append(base_auc - auc(te.injured_next, predict(b_all, Xp)))
    imp.append((f, np.mean(drops), np.std(drops)))
for f, m, s in sorted(imp, key=lambda t: -t[1]):
    log(f"  {f:26s} {m:+.4f} (sd {s:.4f})")
log("")

# 5) overfitting control: same pipeline on shuffled labels (should be ~0.5 AUC)
shuf = trva.injured_next.values.copy()
np.random.default_rng(2).shuffle(shuf)
b_sh, _ = fit_xgb(trva[FEATURES].values, shuf, DEPTH, n_rounds=ROUNDS)
log(f"Control: XGBoost trained on SHUFFLED labels -> test AUC {auc(te.injured_next, predict(b_sh, te[FEATURES].values)):.3f}")

# 6) raw rates by workload and listing status (full data, to see what drives the signal)
log("")
log("Raw injury rate next week, by touches this game (all RB-games 2016-2024):")
tb = df.assign(b=pd.cut(df.touches, [-1, 5, 10, 15, 20, 25, 100])).groupby("b", observed=True)
log(str(tb.agg(n=("injured_next", "size"), rate=("injured_next", "mean")).round(3)))
log("")
log("Raw injury rate next week, by current listing status:")
log(str(df.groupby("listed_now").agg(n=("injured_next", "size"), rate=("injured_next", "mean")).round(3)))
log("")
log("Raw injury rate next week, by prior-season QDO weeks:")
pb = df.assign(b=pd.cut(df.prior_season_qdo_weeks, [-1, 0, 2, 5, 100])).groupby("b", observed=True)
log(str(pb.agg(n=("injured_next", "size"), rate=("injured_next", "mean")).round(3)))

with open(os.path.join(OUT, "results.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
df.to_csv(os.path.join(OUT, "rb_game_panel.csv"), index=False)
