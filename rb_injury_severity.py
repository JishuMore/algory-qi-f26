"""Injury SEVERITY for running backs: does workload/contact predict how bad the injury is?

Builds on the RB-game table cached by rb_injury_corr.py (run that first).

An injury event after game w = the RB is newly listed Out/Doubtful/Questionable for the next
game OR is moved to reserve (IR) within the next 3 weeks. Severity is graded by what actually
happened afterwards:

  games_missed  = consecutive team games after w in which he logged no snaps (snap counts)
  season_ending = he never played again that season
  ir            = roster status RES (reserve/IR) in weeks w+1..w+3

  Category      Rule                                              severity_score
  None          no injury event                                   0
  Minor         listed, missed 0 games                            1
  Moderate      missed 1-2 games                                  2-3
  Major         missed 3+ games, or placed on IR, or season over  4-10

  severity_score = 1 + min(games_missed, 8) for injured RBs, 10 if season-ending, and at
  least 5 if placed on IR. It is a 0-10 scale of time lost.

  Only games with >= 4 team games left in the season are used, so every injury has room
  to be graded Major.
"""
import json
import os

import matplotlib.pyplot as plt
import nflreadpy as nfl
import numpy as np
import polars as pl
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

SEASONS = list(range(2016, 2025))
TEST_SEASONS = [2023, 2024]
OUT = "rb_injury_output"
CATS = ["None", "Minor", "Moderate", "Major"]

df = pl.read_parquet(f"{OUT}/rb_games.parquet")
GROUPS = json.load(open(f"{OUT}/feature_groups.json"))

# ---------- outcome data ----------
sched = nfl.load_schedules(SEASONS).filter(pl.col("game_type") == "REG")
team_games = pl.concat([sched.select("season", "week", team="home_team"),
                        sched.select("season", "week", team="away_team")]).with_columns(
    pl.col("season", "week").cast(pl.Int32))

snaps = nfl.load_snap_counts(SEASONS).filter(pl.col("game_type") == "REG").filter(
    (pl.col("offense_snaps").fill_null(0) + pl.col("st_snaps").fill_null(0)) > 0).select(
    pl.col("season", "week").cast(pl.Int32), pfr_id="pfr_player_id", played=pl.lit(1)).unique()

ir = nfl.load_rosters_weekly(SEASONS).filter(pl.col("status") == "RES").select(
    pl.col("season", "week").cast(pl.Int32), player_id="gsis_id").unique()

inj = nfl.load_injuries(SEASONS).filter(pl.col("game_type") == "REG").select(
    pl.col("season", "week").cast(pl.Int32), player_id="gsis_id",
    status_pts=pl.col("report_status").replace_strict({"Questionable": 1, "Doubtful": 2, "Out": 3},
                                                      default=0, return_dtype=pl.Int32),
    body=pl.col("report_primary_injury").fill_null("unknown").str.to_lowercase(),
).filter(pl.col("status_pts") > 0).sort("status_pts", descending=True).unique(["season", "week", "player_id"], keep="first")

# ---------- severity per RB-game ----------
base = df.filter(pl.col("pfr_id").is_not_null()).select("player_id", "pfr_id", "team", "season", "week")

# every later team game in the season, flag whether the RB played it
later = (base.join(team_games.rename({"week": "g_week"}), on=["team", "season"])
         .filter(pl.col("g_week") > pl.col("week"))
         .join(snaps.rename({"week": "g_week"}), on=["season", "g_week", "pfr_id"], how="left")
         .with_columns(pl.col("played").fill_null(0))
         .sort("player_id", "season", "week", "g_week"))
missed = later.group_by("player_id", "season", "week").agg(
    remaining=pl.len(),
    # games before first appearance
    games_missed=pl.col("played").cum_sum().eq(0).sum(),
)

ir_next = (base.join(ir.rename({"week": "ir_week"}), on=["player_id", "season"])
           .filter((pl.col("ir_week") > pl.col("week")) & (pl.col("ir_week") <= pl.col("week") + 3))
           .select("player_id", "season", "week", ir=pl.lit(1)).unique())
next_rep = inj.with_columns(week=pl.col("week") - 1)

sev = (base.join(missed, on=["player_id", "season", "week"], how="left")
       .join(ir_next, on=["player_id", "season", "week"], how="left")
       .join(next_rep, on=["player_id", "season", "week"], how="left")
       .with_columns(pl.col("ir", "status_pts").fill_null(0), pl.col("games_missed", "remaining").fill_null(0),
                     pl.col("body").fill_null("(IR move, not on report)"))
       .with_columns(event=((pl.col("status_pts") > 0) | (pl.col("ir") == 1)).cast(pl.Int32),
                     season_ending=((pl.col("games_missed") == pl.col("remaining")) & (pl.col("remaining") > 0)).cast(pl.Int32))
       # a missed stretch with no injury listing or IR is not counted as an injury (benching, inactive)
       .with_columns(games_missed=pl.when(pl.col("event") == 1).then(pl.col("games_missed")).otherwise(0),
                     season_ending=pl.col("season_ending") * pl.col("event"))
       .with_columns(
           severity_score=pl.when(pl.col("event") == 0).then(0)
           .when(pl.col("season_ending") == 1).then(10)
           .otherwise(pl.max_horizontal(1 + pl.col("games_missed").clip(0, 8), pl.col("ir") * 5)),
       ).with_columns(
           severity_cat=pl.when(pl.col("severity_score") == 0).then(0)
           .when(pl.col("severity_score") == 1).then(1)
           .when(pl.col("severity_score") <= 3).then(2).otherwise(3),
       ).select("player_id", "season", "week", "status_pts", "body", "ir", "games_missed",
                "season_ending", "event", "severity_score", "severity_cat", "remaining"))

# need >= 4 games of runway, otherwise a late-season injury can never be graded Major
# (and missing the final game would look "season-ending")
df = df.join(sev, on=["player_id", "season", "week"], how="inner").filter(pl.col("remaining") >= 4).with_columns(
    major=(pl.col("severity_cat") == 3).cast(pl.Int32))

ALL = [f for g in GROUPS.values() for f in g]
PREGAME = [f for f in ALL if f not in GROUPS["in-game exit"] and f != "q4_carries"]  # drop in-game-injury tells
train = df.filter(~pl.col("season").is_in(TEST_SEASONS))
test = df.filter(pl.col("season").is_in(TEST_SEASONS))
X = lambda d, f: d.select(f).to_numpy().astype(float)

# ---------- descriptive ----------
dist = df.group_by("severity_cat").agg(
    n=pl.len(), share=pl.len() / df.height, mean_games_missed=pl.mean("games_missed"),
    ir_rate=pl.mean("ir"), season_ending=pl.mean("season_ending")).sort("severity_cat").with_columns(
    category=pl.col("severity_cat").map_elements(lambda i: CATS[i], return_dtype=pl.Utf8))
body = (df.filter(pl.col("event") == 1).group_by("body").agg(
    n=pl.len(), mean_score=pl.mean("severity_score"), major_share=pl.mean("major"))
        .filter(pl.col("n") >= 15).sort("mean_score", descending=True))

def cat_rates(col, edges):
    return (df.with_columns(b=pl.col(col).cut(edges)).group_by("b").agg(
        n=pl.len(),
        minor=(pl.col("severity_cat") == 1).mean(), moderate=(pl.col("severity_cat") == 2).mean(),
        major=(pl.col("severity_cat") == 3).mean(), mean_score=pl.mean("severity_score"))
            .sort("b"))

# ---------- univariate: severity score (all games) and severity GIVEN injury ----------
inj_only = df.filter(pl.col("event") == 1)
rows = []
for f in ALL:
    x, s = df[f].cast(pl.Float64).to_numpy(), df["severity_score"].to_numpy()
    m = ~np.isnan(x)
    r_all, p_all = spearmanr(x[m], s[m])
    xi, si = inj_only[f].cast(pl.Float64).to_numpy(), inj_only["severity_score"].to_numpy()
    mi = ~np.isnan(xi)
    r_inj, p_inj = spearmanr(xi[mi], si[mi])
    xm = df["major"].to_numpy()
    r_maj, p_maj = spearmanr(x[m], xm[m])
    rows.append((f, r_all, p_all, r_maj, p_maj, r_inj, p_inj))
scan = pl.DataFrame(rows, schema=["feature", "rho_score", "p_score", "rho_major", "p_major",
                                  "rho_if_injured", "p_if_injured"], orient="row")

def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)

scan = scan.with_columns(q_score=pl.Series(bh(scan["p_score"])), q_major=pl.Series(bh(scan["p_major"])),
                         q_if_injured=pl.Series(bh(scan["p_if_injured"]))).sort("p_score")

# ---------- models ----------
params = dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=20)

def models(feats):
    out = {}
    clf = xgb.XGBClassifier(objective="multi:softprob", num_class=4, eval_metric="mlogloss", **params)
    clf.fit(X(train, feats), train["severity_cat"].to_numpy())
    P = clf.predict_proba(X(test, feats))
    yc = test["severity_cat"].to_numpy()
    for k in range(1, 4):
        out[f"{CATS[k]} vs rest AUC"] = roc_auc_score(yc == k, P[:, k])
    out["any injury AUC (1-P(None))"] = roc_auc_score(yc > 0, 1 - P[:, 0])
    # expected severity from class probabilities, ranked against actual score
    expected = P @ np.array([0, 1, 2.5, 6.5])
    out["Spearman(expected sev, actual score)"] = spearmanr(expected, test["severity_score"])[0]

    maj = xgb.XGBClassifier(eval_metric="logloss", **params).fit(X(train, feats), train["major"].to_numpy())
    pm = maj.predict_proba(X(test, feats))[:, 1]
    out["Major (binary model) AUC"] = roc_auc_score(test["major"], pm)

    # conditional: given an injury, how bad? (regression on injured only)
    tr_i, te_i = train.filter(pl.col("event") == 1), test.filter(pl.col("event") == 1)
    reg = xgb.XGBRegressor(objective="reg:squarederror", **{**params, "min_child_weight": 10})
    reg.fit(X(tr_i, feats), tr_i["severity_score"].to_numpy())
    out["IF injured: Spearman(pred, actual score)"] = spearmanr(reg.predict(X(te_i, feats)), te_i["severity_score"])[0]
    out["IF injured: Major AUC"] = roc_auc_score(te_i["major"], reg.predict(X(te_i, feats)))
    return out, maj, reg

res_pre, maj_pre, reg_pre = models(PREGAME)
res_all, _, _ = models(ALL)
res_wc, _, _ = models(GROUPS["workload"] + GROUPS["contact"])

def shap(model, d, feats, k=12):
    c = model.get_booster().predict(xgb.DMatrix(X(d, feats)), pred_contribs=True)[:, :-1]
    return sorted(zip(feats, np.abs(c).mean(0)), key=lambda t: -t[1])[:k]

shap_major = shap(maj_pre, test, PREGAME)
shap_cond = shap(reg_pre, test.filter(pl.col("event") == 1), PREGAME)

# ---------- report ----------
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(12); pl.Config.set_tbl_width_chars(160)
fmt = lambda d: str(d.with_columns(pl.selectors.float().round(3)))
lines = [
    f"RB-games: {df.height} (train {train.height}, test {test.height}); injury events: {int(df['event'].sum())}",
    f"  events from injury report: {int((df['status_pts'] > 0).sum())}, IR moves with no report listing: "
    f"{int(((df['status_pts'] == 0) & (df['ir'] == 1)).sum())}",
    "",
    "Severity distribution:", fmt(dist.select("category", "n", "share", "mean_games_missed", "ir_rate", "season_ending")),
    "",
    "Severity score histogram (injured only):",
    fmt(inj_only.group_by("severity_score").len().sort("severity_score")),
    "",
    "Body part (injured, n>=15) by mean severity:", fmt(body),
    "",
    "Category rates by season-to-date hits:", fmt(cat_rates("season_hits", [50, 100, 150, 200])),
    "Category rates by avg hits last 3 games:", fmt(cat_rates("hits_l3", [10, 15, 20, 25])),
    "Category rates by carries this game:", fmt(cat_rates("carries", [10, 15, 20, 25])),
    "Category rates by career carries:", fmt(cat_rates("career_carries", [200, 500, 1000, 1500])),
    "Category rates by weeks on report earlier this season:", fmt(cat_rates("inj_weeks_this_season", [0, 1, 3])),
    "Category rates by age:", fmt(cat_rates("age", [24, 26, 28])),
    "",
    "Univariate Spearman (sorted by p_score). rho_score = vs 0-10 severity over all games;",
    "rho_major = vs Major injury flag; rho_if_injured = vs severity among injured RBs only. q = BH-adjusted.",
    fmt(scan.select("feature", "rho_score", "q_score", "rho_major", "q_major", "rho_if_injured", "q_if_injured")),
    "",
    "Holdout 2023-24 model results:",
    f"  {'metric':44s} {'pre-game':>9s} {'all feats':>9s} {'wkld+contact':>12s}",
    *[f"  {k:44s} {res_pre[k]:9.3f} {res_all[k]:9.3f} {res_wc[k]:12.3f}" for k in res_pre],
    "  (pre-game drops q4_share / q4_carries, which mostly flag an injury that already happened in-game)",
    "",
    "Top |SHAP|, Major-injury model (pre-game features):", *[f"  {f:24s} {v:.4f}" for f, v in shap_major],
    "",
    "Top |SHAP|, severity-given-injury model (pre-game features):", *[f"  {f:24s} {v:.4f}" for f, v in shap_cond],
]
report = "\n".join(lines)
print(report)
open(f"{OUT}/severity.txt", "w").write(report)

# ---------- figure ----------
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
t = cat_rates("season_hits", [50, 100, 150, 200])
xs = [str(b) for b in t["b"]]
bottom = np.zeros(len(xs))
for k, col, c in [(1, "minor", "#F2C14E"), (2, "moderate", "#F78154"), (3, "major", "#B33F40")]:
    ax[0].bar(xs, t[col], bottom=bottom, color=c, label=CATS[k]); bottom += t[col].to_numpy()
ax[0].set_title("Injury rate next week by severity\nvs. season-to-date hits taken"); ax[0].legend()
ax[0].set_xlabel("season hits bucket"); ax[0].tick_params(axis="x", rotation=20)
h = inj_only.group_by("severity_score").len().sort("severity_score")
ax[1].bar(h["severity_score"], h["len"], color="#4C72B0")
ax[1].set_title("Severity score of RB injuries (0-10 scale)"); ax[1].set_xlabel("severity score")
f, v = zip(*shap_major)
ax[2].barh(f[::-1], v[::-1], color="#B33F40"); ax[2].set_title("Major-injury model: feature impact (mean |SHAP|)")
plt.tight_layout(); plt.savefig(f"{OUT}/severity.png", dpi=130)
