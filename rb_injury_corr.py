"""Broader correlation scan: which RB features (beyond carries/hits) relate to a new
injury designation the following week?

Same unit and label as rb_injury.py (RB-game, new Out/Doubtful/Questionable next week).
For each feature: Spearman rho, point-biserial p-value (Benjamini-Hochberg corrected),
and single-feature AUC on the 2023-24 holdout. Then XGBoost on feature groups.
"""
import json
import os

import matplotlib.pyplot as plt
import nflreadpy as nfl
import numpy as np
import polars as pl
import xgboost as xgb
from scipy.stats import spearmanr, pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEASONS = list(range(2016, 2025))
TEST_SEASONS = [2023, 2024]
OUT = "rb_injury_output"
os.makedirs(OUT, exist_ok=True)

# ---------- play-by-play ----------
pbp = nfl.load_pbp(SEASONS).filter(pl.col("season_type") == "REG").with_columns(
    pl.col("season", "week").cast(pl.Int32))
tackler_cols = ["solo_tackle_1_player_id", "solo_tackle_2_player_id",
                "assist_tackle_1_player_id", "assist_tackle_2_player_id",
                "assist_tackle_3_player_id", "assist_tackle_4_player_id",
                "tackle_with_assist_1_player_id", "tackle_with_assist_2_player_id"]
pbp = pbp.with_columns(
    n_tacklers=pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in tackler_cols]))

# game-level context
game_ctx = pbp.group_by("game_id").agg(
    turf=(pl.col("surface").first().fill_null("").str.contains("grass").not_()).cast(pl.Int32),
    dome=pl.col("roof").first().is_in(["dome", "closed"]).cast(pl.Int32),
    temp=pl.col("temp").first().cast(pl.Float64),
    wind=pl.col("wind").first().cast(pl.Float64),
    game_date=pl.col("game_date").first().str.to_date(),
    home_team=pl.col("home_team").first(),
    div_game=pl.col("div_game").first(),
)

carry_plays = pbp.filter((pl.col("rush_attempt") == 1) & pl.col("rusher_player_id").is_not_null())
rush = carry_plays.group_by("game_id", "season", "week", player_id="rusher_player_id").agg(
    carries=pl.len(),
    team=pl.col("posteam").first(),
    rush_hits=pl.sum("n_tacklers"),
    gang_tackles=(pl.col("n_tacklers") >= 2).sum(),
    inside_share=(pl.col("run_location") == "middle").mean(),
    guard_share=(pl.col("run_gap") == "guard").mean(),
    goal_line_carries=(pl.col("yardline_100") <= 5).sum(),
    long_runs=(pl.col("rushing_yards") >= 20).sum(),
    stuffs=(pl.col("rushing_yards") <= 0).sum(),
    fumbles=pl.col("fumble").fill_null(0).sum(),
    shotgun_share=pl.col("shotgun").mean(),
    ypc=pl.col("rushing_yards").mean(),
    avg_score_diff=pl.col("score_differential").mean(),
    leading_share=(pl.col("score_differential") > 0).mean(),
    q4_carries=(pl.col("qtr") >= 4).sum(),
)
recv = pbp.filter((pl.col("pass_attempt") == 1) & pl.col("receiver_player_id").is_not_null()).group_by(
    "game_id", player_id="receiver_player_id").agg(
    targets=pl.len(), receptions=pl.col("complete_pass").sum(),
    rec_hits=pl.when(pl.col("complete_pass") == 1).then(pl.col("n_tacklers")).otherwise(0).sum(),
    air_yards_targets=pl.col("air_yards").fill_null(0).mean(),
)
games = (rush.join(recv, on=["game_id", "player_id"], how="left")
         .with_columns(pl.col("targets", "receptions", "rec_hits").fill_null(0))
         .join(game_ctx, on="game_id", how="left")
         .with_columns(home=(pl.col("team") == pl.col("home_team")).cast(pl.Int32),
                       hits=pl.col("rush_hits") + pl.col("rec_hits")))

players = nfl.load_players().select(
    player_id="gsis_id", pfr_id="pfr_id", position="position", birth_date="birth_date",
    weight=pl.col("weight").cast(pl.Float64), height=pl.col("height").cast(pl.Float64),
    draft_round=pl.col("draft_round").cast(pl.Float64), rookie_season=pl.col("rookie_season").cast(pl.Float64))
games = games.join(players, on="player_id").filter(pl.col("position") == "RB")

snaps = nfl.load_snap_counts(SEASONS).filter(pl.col("game_type") == "REG").select(
    pl.col("season", "week").cast(pl.Int32), pfr_id="pfr_player_id",
    off_snaps=pl.col("offense_snaps").cast(pl.Float64), snap_pct=pl.col("offense_pct").cast(pl.Float64),
    st_snaps=pl.col("st_snaps").cast(pl.Float64))
games = games.join(snaps, on=["season", "week", "pfr_id"], how="left")

# ---------- derived / history features ----------
games = games.sort("player_id", "season", "week").with_columns(
    touches=pl.col("carries") + pl.col("receptions"),
    hits_per_touch=pl.col("hits") / (pl.col("carries") + pl.col("receptions")).clip(1),
    age=(pl.col("game_date") - pl.col("birth_date").str.to_date(strict=False)).dt.total_days() / 365.25,
    bmi=703 * pl.col("weight") / pl.col("height") ** 2,
    years_exp=pl.col("season") - pl.col("rookie_season"),
    q4_share=pl.col("q4_carries") / pl.col("carries"),
    touches_per_snap=(pl.col("carries") + pl.col("targets")) / pl.col("off_snaps").clip(1),
)
games = games.with_columns(
    career_carries=pl.col("carries").cum_sum().over("player_id") - pl.col("carries"),
    career_hits=pl.col("hits").cum_sum().over("player_id") - pl.col("hits"),
    season_hits=pl.col("hits").cum_sum().over("player_id", "season"),
    hits_l3=pl.col("hits").rolling_mean(3, min_samples=1).over("player_id", "season"),
    carries_spike=pl.col("carries") - pl.col("carries").shift(1).rolling_mean(4, min_samples=1).over("player_id", "season"),
    days_since_last=(pl.col("game_date") - pl.col("game_date").shift(1).over("player_id", "season")).dt.total_days(),
)
# rest before the NEXT game (short week after this game)
next_game = game_ctx.select("game_id", "game_date")
team_sched = games.select("team", "season", "week", "game_date").unique().sort("team", "season", "week").with_columns(
    days_to_next=(pl.col("game_date").shift(-1).over("team", "season") - pl.col("game_date")).dt.total_days())
games = games.join(team_sched.select("team", "season", "week", "days_to_next").unique(["team", "season", "week"]),
                   on=["team", "season", "week"], how="left").with_columns(
    short_week_next=(pl.col("days_to_next") <= 5).cast(pl.Int32))

# ---------- injury history + label ----------
inj_all = nfl.load_injuries(SEASONS).filter(pl.col("game_type") == "REG").select(
    pl.col("season", "week").cast(pl.Int32), player_id="gsis_id", status="report_status",
    body=pl.col("report_primary_injury").fill_null("").str.to_lowercase())
inj = inj_all.filter(pl.col("status").is_in(["Out", "Doubtful", "Questionable"])).unique(["season", "week", "player_id"])

on_report = inj.select("season", "week", "player_id", on_report_now=pl.lit(1))
nxt = inj.with_columns(week=pl.col("week") - 1).select("season", "week", "player_id", injured_next=pl.lit(1),
                                                       next_body="body")
# prior injury history: weeks on report earlier this season / last season / career
hist = inj.select("season", "week", "player_id").with_columns(flag=pl.lit(1))
games = games.join(hist, on=["season", "week", "player_id"], how="left").with_columns(pl.col("flag").fill_null(0))
games = games.with_columns(
    inj_weeks_this_season=pl.col("flag").cum_sum().over("player_id", "season") - pl.col("flag"),
)
by_season = hist.group_by("player_id", "season").agg(prev_season_inj_weeks=pl.len()).with_columns(pl.col("season") + 1)
games = games.join(by_season, on=["player_id", "season"], how="left").with_columns(pl.col("prev_season_inj_weeks").fill_null(0))
# weeks since last on report (any time before this game)
games = games.with_columns(
    gidx=pl.col("week").cum_count().over("player_id"),
).with_columns(
    last_flag_idx=pl.when(pl.col("flag") == 1).then(pl.col("gidx")).forward_fill().shift(1).over("player_id"),
).with_columns(games_since_inj=(pl.col("gidx") - pl.col("last_flag_idx")).fill_null(99).clip(0, 99))

df = (games.drop("flag")
      .join(on_report, on=["season", "week", "player_id"], how="left")
      .join(nxt, on=["season", "week", "player_id"], how="left")
      .with_columns(pl.col("on_report_now", "injured_next").fill_null(0))
      .filter((pl.col("on_report_now") == 0) & (pl.col("week") < pl.col("week").max().over("season")))
      .filter(pl.col("carries") >= 3)
      # a bye next week means no injury report next week -> label would be 0 by construction
      .filter(pl.col("days_to_next") <= 8))

GROUPS = {
    "workload": ["carries", "targets", "receptions", "touches", "q4_carries", "goal_line_carries",
                 "off_snaps", "snap_pct", "st_snaps", "touches_per_snap", "carries_spike"],
    "contact": ["hits", "rush_hits", "rec_hits", "hits_per_touch", "gang_tackles", "stuffs",
                "inside_share", "guard_share", "hits_l3", "season_hits", "fumbles"],
    "efficiency/style": ["ypc", "long_runs", "shotgun_share", "air_yards_targets"],
    "game context": ["turf", "dome", "temp", "wind", "home", "div_game", "avg_score_diff",
                     "leading_share", "days_since_last", "days_to_next", "short_week_next", "week"],
    "player": ["age", "weight", "height", "bmi", "draft_round", "years_exp", "career_carries", "career_hits"],
    "in-game exit": ["q4_share"],
    "injury history": ["inj_weeks_this_season", "prev_season_inj_weeks", "games_since_inj"],
}
FEATURES = [f for g in GROUPS.values() for f in g]
group_of = {f: g for g, fs in GROUPS.items() for f in fs}
# cache for rb_injury_severity.py
df.write_parquet(f"{OUT}/rb_games.parquet")
json.dump(GROUPS, open(f"{OUT}/feature_groups.json", "w"), indent=1)

train = df.filter(~pl.col("season").is_in(TEST_SEASONS))
test = df.filter(pl.col("season").is_in(TEST_SEASONS))
y_all, ytr, yte = df["injured_next"].to_numpy(), train["injured_next"].to_numpy(), test["injured_next"].to_numpy()

# ---------- univariate scan ----------
rows = []
for f in FEATURES:
    x = df[f].cast(pl.Float64).to_numpy()
    m = ~np.isnan(x)
    rho, _ = spearmanr(x[m], y_all[m])
    _, pval = pointbiserialr(y_all[m], x[m])
    xtr = np.nan_to_num(train[f].cast(pl.Float64).to_numpy(), nan=np.nanmedian(x))
    xte = np.nan_to_num(test[f].cast(pl.Float64).to_numpy(), nan=np.nanmedian(x))
    lr = LogisticRegression().fit(xtr[:, None], ytr)          # direction learned on train only
    auc = roc_auc_score(yte, lr.predict_proba(xte[:, None])[:, 1])
    rows.append((f, group_of[f], rho, pval, auc, m.mean()))
scan = pl.DataFrame(rows, schema=["feature", "group", "spearman", "p", "test_auc", "coverage"], orient="row")
# Benjamini-Hochberg
order = np.argsort(scan["p"].to_numpy()); n = len(order)
q = np.empty(n); q[order] = np.minimum.accumulate((scan["p"].to_numpy()[order] * n / np.arange(1, n + 1))[::-1])[::-1]
scan = scan.with_columns(q_bh=pl.Series(np.minimum(q, 1))).sort("p")

# ---------- XGBoost by feature group ----------
def fit_auc(feats):
    m = xgb.XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=20, eval_metric="logloss")
    m.fit(train.select(feats).to_numpy().astype(float), ytr)
    p = m.predict_proba(test.select(feats).to_numpy().astype(float))[:, 1]
    return roc_auc_score(yte, p), m

base_feats = GROUPS["workload"] + GROUPS["contact"]
auc_all, model_all = fit_auc(FEATURES)
grp_rows = [("workload + contact only", fit_auc(base_feats)[0])]
for g, fs in GROUPS.items():
    grp_rows.append((f"{g} alone", fit_auc(fs)[0]))
for g in GROUPS:
    grp_rows.append((f"all minus {g}", fit_auc([f for f in FEATURES if f not in GROUPS[g]])[0]))
grp_rows.append(("ALL features", auc_all))

contrib = model_all.get_booster().predict(
    xgb.DMatrix(test.select(FEATURES).to_numpy().astype(float)), pred_contribs=True)[:, :-1]
shap_imp = sorted(zip(FEATURES, np.abs(contrib).mean(0)), key=lambda t: -t[1])

def bucket(col, edges):
    return (df.with_columns(b=pl.col(col).cut(edges)).group_by("b")
            .agg(n=pl.len(), injury_rate=pl.mean("injured_next")).sort("b"))

pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(10); pl.Config.set_fmt_float("full")
lines = [
    f"RB-games: {df.height} (train {train.height}, test {test.height}); base rate {y_all.mean():.3%}",
    "",
    "Univariate scan (sorted by p; q_bh = Benjamini-Hochberg false-discovery-adjusted; test_auc = 2023-24 holdout):",
    str(scan.with_columns(pl.col("spearman", "test_auc", "coverage").round(3),
                          pl.col("p", "q_bh").round(4))),
    "",
    "XGBoost test AUC by feature group:",
    *[f"  {name:32s} {a:.3f}" for name, a in grp_rows],
    "",
    "Top 15 mean |SHAP| (all-feature model):",
    *[f"  {f:24s} {v:.4f}" for f, v in shap_imp[:15]],
    "",
    "Injury rate by weeks on report earlier this season:", str(bucket("inj_weeks_this_season", [0, 1, 3])),
    "Injury rate by games since last on report:", str(bucket("games_since_inj", [1, 3, 8, 98])),
    "Injury rate by age:", str(bucket("age", [24, 26, 28])),
    "Injury rate by surface (turf=1):",
    str(df.group_by("turf").agg(n=pl.len(), injury_rate=pl.mean("injured_next")).sort("turf")),
    "Injury rate by short week next (<=5 days):",
    str(df.group_by("short_week_next").agg(n=pl.len(), injury_rate=pl.mean("injured_next")).sort("short_week_next")),
]
report = "\n".join(lines)
print(report)
open(f"{OUT}/correlations.txt", "w").write(report)

# ---------- figure ----------
fig, ax = plt.subplots(1, 2, figsize=(15, 8))
s = scan.sort("spearman")
colors = ["#C44E52" if qq < 0.05 else "#B0B0B0" for qq in s["q_bh"]]
ax[0].barh(s["feature"], s["spearman"], color=colors)
ax[0].axvline(0, color="k", lw=0.8)
ax[0].set_title("Spearman correlation with new injury next week\n(red = significant after FDR correction, q<0.05)")
names, aucs = zip(*grp_rows)
ax[1].barh(names[::-1], aucs[::-1], color="#4C72B0"); ax[1].axvline(0.5, color="k", ls="--", lw=0.8)
ax[1].set_xlim(0.45, max(aucs) + 0.03); ax[1].set_title("XGBoost holdout AUC (2023-24) by feature group")
plt.tight_layout(); plt.savefig(f"{OUT}/correlations.png", dpi=130)
