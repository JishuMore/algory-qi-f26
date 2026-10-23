"""Does running back workload (carries, hits taken) predict injury the next week?

Data: nflreadpy play-by-play + weekly injury reports, regular season 2016-2024.
Unit: one row per RB per game played.
Label: RB appears on the NEXT week's injury report as Out/Doubtful/Questionable
       when he was not on the report (with one of those statuses) this week.
Model: XGBoost, trained on 2016-2022, tested out-of-sample on 2023-2024.
"""
import os

import matplotlib.pyplot as plt
import nflreadpy as nfl
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

SEASONS = list(range(2016, 2025))
TEST_SEASONS = [2023, 2024]
OUT = "rb_injury_output"
os.makedirs(OUT, exist_ok=True)

# ---------- play-by-play -> RB game logs ----------
pbp = nfl.load_pbp(SEASONS).filter(pl.col("season_type") == "REG")

tackler_cols = ["solo_tackle_1_player_id", "solo_tackle_2_player_id",
                "assist_tackle_1_player_id", "assist_tackle_2_player_id",
                "assist_tackle_3_player_id", "assist_tackle_4_player_id",
                "tackle_with_assist_1_player_id", "tackle_with_assist_2_player_id"]
pbp = pbp.with_columns(
    n_tacklers=pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in tackler_cols]),
)

# ball carrier on runs and completed receptions
carries = pbp.filter((pl.col("rush_attempt") == 1) & pl.col("rusher_player_id").is_not_null()).select(
    "season", "week", "game_id", player_id="rusher_player_id",
    carry=pl.lit(1), reception=pl.lit(0), target=pl.lit(0),
    hits="n_tacklers", tfl=pl.col("tackled_for_loss").fill_null(0),
    yards=pl.col("rushing_yards").fill_null(0),
    big_hit_play=(pl.col("n_tacklers") >= 2).cast(pl.Int32),
)
targets = pbp.filter((pl.col("pass_attempt") == 1) & pl.col("receiver_player_id").is_not_null()).select(
    "season", "week", "game_id", player_id="receiver_player_id",
    carry=pl.lit(0), reception=pl.col("complete_pass").fill_null(0).cast(pl.Int32), target=pl.lit(1),
    hits=pl.when(pl.col("complete_pass") == 1).then(pl.col("n_tacklers")).otherwise(0),
    tfl=pl.lit(0), yards=pl.col("receiving_yards").fill_null(0),
    big_hit_play=((pl.col("complete_pass") == 1) & (pl.col("n_tacklers") >= 2)).cast(pl.Int32),
)
touches = pl.concat([carries, targets], how="vertical_relaxed")

touches = touches.with_columns(pl.col("season", "week").cast(pl.Int32))
games = touches.group_by("season", "week", "game_id", "player_id").agg(
    pl.sum("carry").alias("carries"), pl.sum("target").alias("targets"),
    pl.sum("reception").alias("receptions"), pl.sum("hits").alias("hits"),
    pl.sum("tfl").alias("tfl"), pl.sum("yards").alias("yards"),
    pl.sum("big_hit_play").alias("gang_tackles"),
)

players = nfl.load_players().select(
    player_id="gsis_id", position="position", birth_date="birth_date",
    weight=pl.col("weight").cast(pl.Float64),
)
games = games.join(players, on="player_id").filter(pl.col("position") == "RB")

# ---------- workload features (all known at end of week w) ----------
games = games.sort("player_id", "season", "week").with_columns(
    touches=pl.col("carries") + pl.col("receptions"),
    hits_per_touch=pl.col("hits") / (pl.col("carries") + pl.col("receptions")).clip(1),
    age=(pl.date(pl.col("season"), 9, 1) - pl.col("birth_date").str.to_date(strict=False)).dt.total_days() / 365.25,
)
over = ["player_id", "season"]
games = games.with_columns(
    season_carries=pl.col("carries").cum_sum().over(over),
    season_hits=pl.col("hits").cum_sum().over(over),
    season_games=pl.col("carries").cum_count().over(over),
    carries_l3=pl.col("carries").rolling_mean(3, min_samples=1).over(over),
    hits_l3=pl.col("hits").rolling_mean(3, min_samples=1).over(over),
    carries_spike=pl.col("carries") - pl.col("carries").shift(1).rolling_mean(4, min_samples=1).over(over),
    days_rest_proxy=pl.col("week") - pl.col("week").shift(1).over(over),
)
# prior-season workload
prior = games.group_by("player_id", "season").agg(
    pl.sum("carries").alias("prev_season_carries"), pl.sum("hits").alias("prev_season_hits"))
prior = prior.with_columns(pl.col("season") + 1)
games = games.join(prior, on=["player_id", "season"], how="left").with_columns(
    pl.col("prev_season_carries", "prev_season_hits").fill_null(0))

# ---------- injury label ----------
inj = nfl.load_injuries(SEASONS).filter(
    (pl.col("game_type") == "REG") & pl.col("report_status").is_in(["Out", "Doubtful", "Questionable"])
).select(pl.col("season", "week").cast(pl.Int32), player_id="gsis_id", status="report_status").unique(["season", "week", "player_id"])

on_report = inj.select("season", "week", "player_id", on_report_now=pl.lit(1))
nxt = inj.with_columns(week=pl.col("week") - 1).select(
    "season", "week", "player_id", injured_next=pl.lit(1), next_status="status")
df = (games
      .join(on_report, on=["season", "week", "player_id"], how="left")
      .join(nxt, on=["season", "week", "player_id"], how="left")
      .with_columns(pl.col("on_report_now", "injured_next").fill_null(0))
      # new injuries only; drop each season's final week (no "next week" report)
      .filter((pl.col("on_report_now") == 0) & (pl.col("week") < pl.col("week").max().over("season")))
      .filter(pl.col("carries") >= 3))

FEATURES = ["carries", "targets", "receptions", "hits", "hits_per_touch", "gang_tackles", "tfl",
            "yards", "season_carries", "season_hits", "season_games", "carries_l3", "hits_l3",
            "carries_spike", "days_rest_proxy", "prev_season_carries", "prev_season_hits",
            "age", "weight", "week"]

train = df.filter(~pl.col("season").is_in(TEST_SEASONS))
test = df.filter(pl.col("season").is_in(TEST_SEASONS))
Xtr, ytr = train.select(FEATURES).to_numpy().astype(float), train["injured_next"].to_numpy()
Xte, yte = test.select(FEATURES).to_numpy().astype(float), test["injured_next"].to_numpy()

model = xgb.XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=20, eval_metric="logloss")
model.fit(Xtr, ytr)
p = model.predict_proba(Xte)[:, 1]

# baselines: carries alone, hits alone (logistic)
def single(col):
    i = FEATURES.index(col)
    lr = LogisticRegression().fit(np.nan_to_num(Xtr[:, [i]]), ytr)
    return roc_auc_score(yte, lr.predict_proba(np.nan_to_num(Xte[:, [i]]))[:, 1])

# SHAP-style contributions from xgboost
contrib = model.get_booster().predict(xgb.DMatrix(Xte), pred_contribs=True)[:, :-1]
shap_imp = sorted(zip(FEATURES, np.abs(contrib).mean(0)), key=lambda t: -t[1])

# bucket tables (raw rates, full sample)
def bucket(col, edges):
    return (df.with_columns(b=pl.col(col).cut(edges)).group_by("b")
            .agg(n=pl.len(), injury_rate=pl.mean("injured_next")).sort("b"))

lines = [
    f"RB-games: {df.height}  (train {train.height}, test {test.height})",
    f"Base rate of new injury designation next week: {df['injured_next'].mean():.3%}",
    "",
    f"XGBoost test AUC: {roc_auc_score(yte, p):.3f}   PR-AUC: {average_precision_score(yte, p):.3f} (base {yte.mean():.3f})",
    f"Logistic, carries only AUC: {single('carries'):.3f}",
    f"Logistic, hits only    AUC: {single('hits'):.3f}",
    f"Logistic, season_hits  AUC: {single('season_hits'):.3f}",
    "",
    "Mean |SHAP| (test set):",
    *[f"  {f:22s} {v:.4f}" for f, v in shap_imp],
    "",
    "Injury rate by carries this game:", str(bucket("carries", [5, 10, 15, 20, 25])),
    "Injury rate by hits taken this game:", str(bucket("hits", [5, 10, 15, 20, 25, 30])),
    "Injury rate by season-to-date hits:", str(bucket("season_hits", [25, 50, 100, 150, 200])),
]
pl.Config.set_tbl_rows(20)
report = "\n".join(lines)
print(report)
open(f"{OUT}/results.txt", "w").write(report)

# test-set decile calibration
d = np.searchsorted(np.quantile(p, np.linspace(0, 1, 11)[1:-1]), p, side="right") + 1
dec = pl.DataFrame({"p": p, "y": yte, "d": d}).group_by("d").agg(pl.mean("p"), pl.mean("y")).sort("d")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
f, v = zip(*shap_imp[:12])
ax[0].barh(f[::-1], v[::-1], color="#4C72B0"); ax[0].set_title("XGBoost feature impact (mean |SHAP|)")
for col, a in [("carries", ax[1]), ("hits", ax[1])]:
    t = df.group_by(col).agg(r=pl.mean("injured_next"), n=pl.len()).filter(pl.col("n") >= 100).sort(col)
    a.plot(t[col], t["r"], marker="o", label=col)
ax[1].set_xlabel("count this game"); ax[1].set_ylabel("P(new injury next week)")
ax[1].set_title("Raw injury rate vs. carries / hits"); ax[1].legend()
ax[2].bar(dec["d"], dec["y"], color="#DD8452", label="actual")
ax[2].plot(dec["d"], dec["p"], "k--o", label="predicted")
ax[2].set_title("Test 2023-24: injury rate by predicted decile"); ax[2].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/rb_injury.png", dpi=130)
