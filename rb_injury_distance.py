"""Can we predict RB injuries from how much they ran, and is there a distance threshold?

Builds on rb_games_severity.parquet from rb_injury_severity.py (run rb_injury_corr.py,
then rb_injury_severity.py first). Labels:
  event = new injury listing (Out/Doubtful/Questionable) next week or IR within 3 weeks
  major = missed 3+ games, IR, or season over

"How much they ran", three ways:
  1. yards: rushing + receiving yards (play-by-play / weekly player stats, every game)
  2. NGS distance: Next Gen Stats tracking. NGS `efficiency` = distance traveled / rushing
     yards, so distance_yds = efficiency * rush_yards. NGS only publishes weekly rows for
     10+ carry games, so this is a subset.
  3. accumulated workload BEFORE this game (season and last-3 carries/yards, career carries)

Same-game exposure is contaminated: an RB hurt mid-game stops running, so low same-game
volume partly *reflects* an injury. Prior-game (lagged) exposure is the clean test.

Threshold test: for exposure x, logistic y ~ x + max(0, x - k) for k over a grid; the best k
maximizes likelihood. Player-clustered bootstrap gives a CI for k and for the slope change.
"""
import json
import os

import matplotlib.pyplot as plt
import nflreadpy as nfl
import numpy as np
import polars as pl
import xgboost as xgb
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

TEST_SEASONS = [2024, 2025]
OUT = "rb_injury_output"
rng = np.random.default_rng(0)

df = pl.read_parquet(f"{OUT}/rb_games_severity.parquet")
GROUPS = json.load(open(f"{OUT}/feature_groups.json"))
SEASONS = sorted(df["season"].unique().to_list())

# ---------- accumulated workload from weekly stats (every game, not just the modeled rows) ----------
ws = nfl.load_player_stats(SEASONS).filter(pl.col("season_type") == "REG").select(
    "player_id", pl.col("season", "week").cast(pl.Int32),
    wk_carries=pl.col("carries").fill_null(0).cast(pl.Float64),
    wk_yards=(pl.col("rushing_yards").fill_null(0) + pl.col("receiving_yards").fill_null(0)).cast(pl.Float64),
    wk_rush_yards=pl.col("rushing_yards").fill_null(0).cast(pl.Float64),
).sort("player_id", "season", "week")
ov = ["player_id", "season"]
ws = ws.with_columns(
    prior_season_carries=pl.col("wk_carries").cum_sum().over(ov) - pl.col("wk_carries"),
    prior_season_yards=pl.col("wk_yards").cum_sum().over(ov) - pl.col("wk_yards"),
    prior_l3_carries=pl.col("wk_carries").shift(1).rolling_sum(3, min_samples=1).over(ov).fill_null(0),
    prior_l3_yards=pl.col("wk_yards").shift(1).rolling_sum(3, min_samples=1).over(ov).fill_null(0),
)

# ---------- NGS tracking distance ----------
ngs = nfl.load_nextgen_stats([s for s in SEASONS if s >= 2016], stat_type="rushing").filter(
    (pl.col("season_type") == "REG") & (pl.col("week") > 0)).select(
    player_id="player_gsis_id", season=pl.col("season").cast(pl.Int32), week=pl.col("week").cast(pl.Int32),
    ngs_dist=pl.when(pl.col("rush_yards") > 0).then(pl.col("efficiency") * pl.col("rush_yards")),
    ngs_dist_per_carry=pl.when(pl.col("rush_yards") > 0).then(
        pl.col("efficiency") * pl.col("rush_yards") / pl.col("rush_attempts")),
    ngs_efficiency="efficiency", stacked_box_pct="percent_attempts_gte_eight_defenders",
    time_to_los="avg_time_to_los").unique(["player_id", "season", "week"])
# prior NGS distance over the season (only games NGS covered)
ngs = ngs.sort("player_id", "season", "week").with_columns(
    prior_season_ngs_dist=(pl.col("ngs_dist").fill_null(0).cum_sum().over(["player_id", "season"])
                           - pl.col("ngs_dist").fill_null(0)))

df = (df.join(ws.drop("wk_carries"), on=["player_id", "season", "week"], how="left")
        .join(ngs, on=["player_id", "season", "week"], how="left")
        .with_columns(game_yards=pl.col("wk_yards"), career_carries_prior=pl.col("career_carries")))

EXPOSURE_GAME = ["carries", "touches", "game_yards", "wk_rush_yards"]
EXPOSURE_PRIOR = ["prior_season_carries", "prior_season_yards", "prior_l3_carries", "prior_l3_yards",
                  "career_carries_prior"]
EXPOSURE_NGS = ["ngs_dist", "ngs_dist_per_carry", "ngs_efficiency", "time_to_los", "stacked_box_pct",
                "prior_season_ngs_dist"]

# ---------- threshold search ----------
def _fit_ll(Xm, y):
    m = LogisticRegression(C=1e6, max_iter=500).fit(Xm, y)
    p = np.clip(m.predict_proba(Xm)[:, 1], 1e-9, 1 - 1e-9)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)), m

def hinge(x, y, grid):
    s = x.std()
    ll0, _ = _fit_ll((x / s)[:, None], y)
    best = (-np.inf, None, None)
    for k in grid:
        Xm = np.column_stack([x / s, np.maximum(0, x - k) / s])
        ll, m = _fit_ll(Xm, y)
        if ll > best[0]:
            best = (ll, k, m)
    ll, k, m = best
    b_before, b_change = m.coef_[0] / s
    return dict(k=k, lr=2 * (ll - ll0), b_before=b_before, b_after=b_before + b_change)

def threshold_test(d, col, label, n_boot=60):
    d = d.filter(pl.col(col).is_not_null())
    x, y, pid = d[col].cast(pl.Float64).to_numpy(), d[label].to_numpy(), d["player_id"].to_numpy()
    grid = np.unique(np.quantile(x, np.linspace(0.1, 0.9, 17)))
    res = hinge(x, y, grid)
    # player-clustered bootstrap for the breakpoint location
    players = np.unique(pid); idx_by = {p: np.where(pid == p)[0] for p in players}
    ks, changes = [], []
    for _ in range(n_boot):
        take = np.concatenate([idx_by[p] for p in rng.choice(players, len(players))])
        r = hinge(x[take], y[take], grid)
        ks.append(r["k"]); changes.append(r["b_after"] - r["b_before"])
    ks, changes = np.array(ks), np.array(changes)
    # 2 extra params (k, slope change); chi2(2) is only approximate when k is searched
    res.update(col=col, label=label, n=len(y), rate=y.mean(),
               p_approx=chi2.sf(res["lr"], 2),
               k_lo=np.quantile(ks, 0.05), k_hi=np.quantile(ks, 0.95),
               share_slope_up=(changes > 0).mean())
    return res

def binned(d, col, label, nb=8):
    d = d.filter(pl.col(col).is_not_null())
    edges = np.unique(np.quantile(d[col].to_numpy(), np.linspace(0, 1, nb + 1))[1:-1])
    t = (d.with_columns(b=pl.col(col).cut(edges.tolist())).group_by("b")
         .agg(n=pl.len(), x=pl.col(col).median(), rate=pl.mean(label)).sort("x"))
    z = 1.96; n, p = t["n"].to_numpy(), t["rate"].to_numpy()
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    return t.with_columns(lo=pl.Series(centre - half), hi=pl.Series(centre + half))

tests = []
for col in EXPOSURE_GAME + EXPOSURE_PRIOR + ["ngs_dist", "prior_season_ngs_dist", "ngs_dist_per_carry"]:
    for label in ["event", "major"]:
        tests.append(threshold_test(df, col, label))
tt = pl.DataFrame(tests).select("col", "label", "n", "rate", "k", "k_lo", "k_hi", "b_before", "b_after",
                                "lr", "p_approx", "share_slope_up")

# ---------- prediction: exposure-only vs full pre-game ----------
PREGAME = [f for g, fs in GROUPS.items() if g != "in-game exit" for f in fs if f != "q4_carries"]
params = dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=20, eval_metric="logloss")

def auc(d, feats, label):
    tr, te = d.filter(~pl.col("season").is_in(TEST_SEASONS)), d.filter(pl.col("season").is_in(TEST_SEASONS))
    m = xgb.XGBClassifier(**params).fit(tr.select(feats).to_numpy().astype(float), tr[label].to_numpy())
    return roc_auc_score(te[label], m.predict_proba(te.select(feats).to_numpy().astype(float))[:, 1]), m

ngs_rows = df.filter(pl.col("ngs_dist").is_not_null())
pred = []
for label in ["event", "major"]:
    for name, d, feats in [
        ("same-game volume only", df, EXPOSURE_GAME),
        ("prior workload only (clean)", df, EXPOSURE_PRIOR),
        ("same-game + prior volume", df, EXPOSURE_GAME + EXPOSURE_PRIOR),
        ("full pre-game model", df, PREGAME),
        ("full pre-game + prior volume", df, PREGAME + EXPOSURE_PRIOR),
        ("[NGS games] NGS distance only", ngs_rows, EXPOSURE_NGS),
        ("[NGS games] full pre-game", ngs_rows, PREGAME),
        ("[NGS games] full pre-game + NGS + prior", ngs_rows, PREGAME + EXPOSURE_NGS + EXPOSURE_PRIOR),
    ]:
        pred.append((label, name, d.filter(pl.col("season").is_in(TEST_SEASONS)).height, auc(d, feats, label)[0]))

# ---------- report ----------
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(14); pl.Config.set_tbl_width_chars(170)
fmt = lambda d: str(d.with_columns(pl.selectors.float().round(4)))
lines = [
    f"Seasons {SEASONS[0]}-{SEASONS[-1]}, RB-games: {df.height}; holdout {TEST_SEASONS}",
    f"  any injury rate {df['event'].mean():.3%}, major rate {df['major'].mean():.3%}",
    f"  games with NGS tracking distance: {ngs_rows.height} "
    f"(median {ngs_rows['ngs_dist'].median():.0f} yds run, {ngs_rows['ngs_dist_per_carry'].median():.1f} yds per carry)",
    "",
    "Threshold (hinge) search. k = breakpoint; b_before/b_after = log-odds slope per unit below/above k;",
    "k_lo-k_hi = 90% player-clustered bootstrap range; share_slope_up = share of bootstraps where risk",
    "steepens past k; p_approx is optimistic because k is searched (treat p > 0.01 as no threshold).",
    fmt(tt),
    "",
    "Holdout AUC:",
    *[f"  {lab:6s} {name:42s} n_test={n:5d}  AUC={a:.3f}" for lab, name, n, a in pred],
]
for col in ["prior_season_carries", "prior_season_yards", "prior_l3_carries", "game_yards", "ngs_dist",
            "prior_season_ngs_dist"]:
    lines += ["", f"Binned rates by {col}:",
              fmt(binned(df, col, "event").join(binned(df, col, "major").select(
                  "b", major_rate="rate"), on="b"))]
df.write_parquet(f"{OUT}/rb_games_distance.parquet")  # cache for rb_injury_combined.py
report = "\n".join(lines)
print(report)
open(f"{OUT}/distance.txt", "w").write(report)

# ---------- figure ----------
cols = [("prior_season_carries", "carries before this game (season)"),
        ("prior_season_yards", "scrimmage yards before this game (season)"),
        ("prior_l3_carries", "carries in previous 3 games"),
        ("game_yards", "scrimmage yards this game"),
        ("ngs_dist", "NGS distance run this game (yds)"),
        ("prior_season_ngs_dist", "NGS distance before this game (season)")]
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
for ax, (col, title) in zip(axes.flat, cols):
    for label, c in [("event", "#4C72B0"), ("major", "#B33F40")]:
        b = binned(df, col, label)
        ax.errorbar(b["x"], b["rate"], yerr=[b["rate"] - b["lo"], b["hi"] - b["rate"]],
                    marker="o", capsize=3, color=c, label="any injury" if label == "event" else "major")
        r = tt.filter((pl.col("col") == col) & (pl.col("label") == label)).row(0, named=True)
        ax.axvline(r["k"], color=c, ls=":", lw=1)
    ax.set_title(f"{title}\n(dotted = best breakpoint)"); ax.set_ylabel("P(injury)")
axes.flat[0].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/distance.png", dpi=130)
