"""Test new candidate variables for RB injury risk.

Builds on rb_games_severity.parquet (run rb_injury_corr.py then rb_injury_severity.py).
Labels: event (any new injury listing next week or IR) and major (3+ games / IR / season over).

New variable groups:
  tacklers     who made the hits: total/avg tackler weight, hits by DL/LB/DB, heavy hits (>=250 lb)
  contact      open-field hits (hit after a 10+ yd gain), out-of-bounds share, hits on 3rd/4th down
  team/opp     team carry share, opponent run D (hits per carry allowed, prior weeks), O-line stuff rate
               (prior weeks), team plays/pace, no-huddle share
  schedule     time zones crossed, night game, Thu/Mon game, Denver altitude, rain/snow,
               Vegas spread and total (pre-game expected script)
  team history team's RB injury rate in prior seasons (medical/strength staff proxy)
  player       contract year, combine 40 time, speed score

Tests: univariate Spearman + BH, and season-by-season (walk-forward) XGBoost: train on all
earlier seasons, test on one season, for 2018-2025. Reports mean AUC change from adding each group
to the existing pre-game model.
"""
import json

import matplotlib.pyplot as plt
import nflreadpy as nfl
import numpy as np
import polars as pl
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

OUT = "rb_injury_output"
df = pl.read_parquet(f"{OUT}/rb_games_severity.parquet")
GROUPS = json.load(open(f"{OUT}/feature_groups.json"))
SEASONS = sorted(df["season"].unique().to_list())
WF_SEASONS = [s for s in SEASONS if s >= 2018]

# ---------- play-level: tackler identity, contact type ----------
pbp = nfl.load_pbp(SEASONS).filter(pl.col("season_type") == "REG").with_columns(
    pl.col("season", "week").cast(pl.Int32))
players = nfl.load_players().select(
    tid="gsis_id", tw=pl.col("weight").cast(pl.Float64), tpos="position_group")
tackler_cols = ["solo_tackle_1_player_id", "solo_tackle_2_player_id",
                "assist_tackle_1_player_id", "assist_tackle_2_player_id",
                "assist_tackle_3_player_id", "assist_tackle_4_player_id",
                "tackle_with_assist_1_player_id", "tackle_with_assist_2_player_id"]
touch = pbp.filter(
    ((pl.col("rush_attempt") == 1) & pl.col("rusher_player_id").is_not_null())
    | ((pl.col("complete_pass") == 1) & pl.col("receiver_player_id").is_not_null())
).with_columns(
    player_id=pl.when(pl.col("rush_attempt") == 1).then(pl.col("rusher_player_id")).otherwise(pl.col("receiver_player_id")),
    gain=pl.col("yards_gained").fill_null(0),
    play_key=pl.concat_str("game_id", pl.col("play_id").cast(pl.Utf8)),
)
hits_long = (touch.select("play_key", "game_id", "player_id", "gain", *tackler_cols)
             .unpivot(index=["play_key", "game_id", "player_id", "gain"], value_name="tid")
             .filter(pl.col("tid").is_not_null())
             .join(players, on="tid", how="left"))
tack = hits_long.group_by("game_id", "player_id").agg(
    tackler_weight_sum=pl.sum("tw"),
    avg_tackler_weight=pl.mean("tw"),
    heavy_hits=(pl.col("tw") >= 250).sum(),
    hits_by_dl=(pl.col("tpos") == "DL").sum(),
    hits_by_lb=(pl.col("tpos") == "LB").sum(),
    hits_by_db=(pl.col("tpos") == "DB").sum(),
    open_field_hits=(pl.col("gain") >= 10).sum(),
)
cont = touch.group_by("game_id", "player_id").agg(
    oob_share=pl.col("out_of_bounds").fill_null(0).mean(),
    late_down_touches=(pl.col("down") >= 3).sum(),
    no_huddle_share=pl.col("no_huddle").fill_null(0).mean(),
)

# ---------- team / opponent context ----------
runs = pbp.filter(pl.col("rush_attempt") == 1).with_columns(
    n_tacklers=pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in tackler_cols]))
team_g = runs.group_by("season", "week", "game_id", team="posteam").agg(
    team_rushes=pl.len(), team_stuff=(pl.col("rushing_yards").fill_null(0) <= 0).mean(),
    opp=pl.col("defteam").first())
team_plays = pbp.filter(pl.col("play_type").is_in(["run", "pass"])).group_by("game_id", team="posteam").agg(
    team_plays=pl.len())
def_g = runs.group_by("season", "week", team="defteam").agg(
    d_hits=pl.sum("n_tacklers"), d_rushes=pl.len())
# prior-weeks (season-to-date, excluding this game) rates
team_g = team_g.sort("team", "season", "week").with_columns(
    oline_stuff_prior=(pl.col("team_stuff").shift(1).cum_sum() / pl.col("team_stuff").shift(1).cum_count())
    .over("team", "season"))
def_g = def_g.sort("team", "season", "week").with_columns(
    opp_hits_per_carry_prior=((pl.col("d_hits").cum_sum() - pl.col("d_hits"))
                              / (pl.col("d_rushes").cum_sum() - pl.col("d_rushes")).clip(1)).over("team", "season")
).select("season", "week", opp="team", opp_hits_per_carry_prior="opp_hits_per_carry_prior")

# ---------- schedule / environment ----------
TZ = {"ARI": -7, "ATL": -5, "BAL": -5, "BUF": -5, "CAR": -5, "CHI": -6, "CIN": -5, "CLE": -5, "DAL": -6,
      "DEN": -7, "DET": -5, "GB": -6, "HOU": -6, "IND": -5, "JAX": -5, "KC": -6, "LA": -8, "LAC": -8,
      "LV": -8, "OAK": -8, "SD": -8, "STL": -6, "MIA": -5, "MIN": -6, "NE": -5, "NO": -6, "NYG": -5,
      "NYJ": -5, "PHI": -5, "PIT": -5, "SEA": -8, "SF": -8, "TB": -5, "TEN": -6, "WAS": -5}
sched = nfl.load_schedules(SEASONS).filter(pl.col("game_type") == "REG")
weather = pbp.group_by("game_id").agg(wx=pl.col("weather").first().fill_null("").str.to_lowercase())
sched = sched.join(weather, on="game_id", how="left").select(
    "game_id", "home_team", "away_team", "weekday", "spread_line", "total_line",
    night=(pl.col("gametime").str.slice(0, 2).cast(pl.Int32, strict=False) >= 19).cast(pl.Int32),
    thu_mon=pl.col("weekday").is_in(["Thursday", "Monday"]).cast(pl.Int32),
    altitude=(pl.col("home_team") == "DEN").cast(pl.Int32),
    precip=pl.col("wx").str.contains("rain|snow|shower|drizzle").cast(pl.Int32),
    neutral=(pl.col("location") == "Neutral").cast(pl.Int32),
)

# ---------- team history: share of the team's RB-games followed by an injury, prior seasons ----------
team_hist = df.group_by("team", "season").agg(r=pl.mean("event"), n=pl.len()).sort("team", "season").with_columns(
    team_rb_inj_rate_prior=((pl.col("r") * pl.col("n")).cum_sum() - pl.col("r") * pl.col("n"))
    .over("team") / (pl.col("n").cum_sum() - pl.col("n")).over("team").clip(1)
).with_columns(pl.when(pl.col("season") == pl.col("season").min().over("team")).then(None)
               .otherwise(pl.col("team_rb_inj_rate_prior")).alias("team_rb_inj_rate_prior")).select(
    "team", "season", "team_rb_inj_rate_prior")

# ---------- player: contract year, combine ----------
con = nfl.load_contracts().filter(pl.col("gsis_id").is_not_null() & (pl.col("position") == "RB")).select(
    player_id="gsis_id", start=pl.col("year_signed").cast(pl.Int32), years=pl.col("years").cast(pl.Int32)
).filter(pl.col("start") > 0).with_columns(end=pl.col("start") + pl.col("years") - 1)
cyear = (df.select("player_id", "season").unique().join(con, on="player_id")
         .filter((pl.col("start") <= pl.col("season")) & (pl.col("end") >= pl.col("season")))
         .sort("start", descending=True).unique(["player_id", "season"], keep="first")
         .select("player_id", "season", contract_year=(pl.col("season") == pl.col("end")).cast(pl.Int32)))
comb = nfl.load_combine().filter(pl.col("pfr_id").is_not_null()).select(
    pfr_id="pfr_id", forty=pl.col("forty").cast(pl.Float64), cwt=pl.col("wt").cast(pl.Float64)
).with_columns(speed_score=pl.col("cwt") * 200 / pl.col("forty") ** 4).unique("pfr_id")

# ---------- assemble ----------
df = (df.join(tack, on=["game_id", "player_id"], how="left")
        .join(cont, on=["game_id", "player_id"], how="left")
        .join(team_g.select("game_id", "team", "team_rushes", "opp", "oline_stuff_prior"), on=["game_id", "team"], how="left")
        .join(team_plays, on=["game_id", "team"], how="left")
        .join(def_g, on=["season", "week", "opp"], how="left")
        .join(sched, on="game_id", how="left")
        .join(team_hist, on=["team", "season"], how="left")
        .join(cyear, on=["player_id", "season"], how="left")
        .join(comb, on="pfr_id", how="left"))
df = df.with_columns(
    carry_share=pl.col("carries") / pl.col("team_rushes"),
    heavy_hit_share=pl.col("heavy_hits") / pl.col("hits").clip(1),
    tz_shift=pl.when(pl.col("team") == pl.col("home_team")).then(0).otherwise(
        (pl.col("team").replace_strict(TZ, default=None) - pl.col("home_team").replace_strict(TZ, default=None)).abs()),
    team_spread=pl.when(pl.col("team") == pl.col("home_team")).then(pl.col("spread_line")).otherwise(-pl.col("spread_line")),
    contract_year=pl.col("contract_year").fill_null(0),
)

NEW = {
    "tacklers": ["tackler_weight_sum", "avg_tackler_weight", "heavy_hits", "heavy_hit_share",
                 "hits_by_dl", "hits_by_lb", "hits_by_db"],
    "contact type": ["open_field_hits", "oob_share", "late_down_touches"],
    "team/opp": ["carry_share", "opp_hits_per_carry_prior", "oline_stuff_prior", "team_plays", "no_huddle_share"],
    "schedule/env": ["tz_shift", "night", "thu_mon", "altitude", "precip", "neutral", "team_spread", "total_line"],
    "team history": ["team_rb_inj_rate_prior"],
    "player": ["contract_year", "forty", "speed_score"],
}
NEWF = [f for g in NEW.values() for f in g]
group_of = {f: g for g, fs in NEW.items() for f in fs}
PREGAME = [f for g, fs in GROUPS.items() if g != "in-game exit" for f in fs if f != "q4_carries"]

# ---------- univariate ----------
def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)

rows = []
for f in NEWF:
    x = df[f].cast(pl.Float64).to_numpy(); m = ~np.isnan(x)
    r_e, p_e = spearmanr(x[m], df["event"].to_numpy()[m])
    r_m, p_m = spearmanr(x[m], df["major"].to_numpy()[m])
    hi, lo = x[m] >= np.nanmedian(x), x[m] < np.nanmedian(x)
    ev = df["event"].to_numpy()[m]
    rows.append((f, group_of[f], m.mean(), r_e, p_e, r_m, p_m,
                 ev[hi].mean() if hi.any() else np.nan, ev[lo].mean() if lo.any() else np.nan))
scan = pl.DataFrame(rows, schema=["feature", "group", "coverage", "rho_event", "p_event", "rho_major", "p_major",
                                  "rate_top_half", "rate_bottom_half"], orient="row")
scan = scan.with_columns(q_event=pl.Series(bh(scan["p_event"])), q_major=pl.Series(bh(scan["p_major"]))).sort("p_event")

# binary features: rate when 1 vs 0
binrows = []
for f in ["night", "thu_mon", "altitude", "precip", "neutral", "contract_year"]:
    t = df.group_by(f).agg(n=pl.len(), event=pl.mean("event"), major=pl.mean("major")).sort(f)
    binrows.append(f"  {f:14s} " + "  ".join(
        f"[{r[f]}] n={r['n']:5d} any={r['event']:.3f} major={r['major']:.3f}" for r in t.iter_rows(named=True)))

# ---------- walk-forward XGBoost ----------
params = dict(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=20, eval_metric="logloss")

def walk_forward(feats, label):
    aucs = []
    for s in WF_SEASONS:
        tr, te = df.filter(pl.col("season") < s), df.filter(pl.col("season") == s)
        m = xgb.XGBClassifier(**params).fit(tr.select(feats).to_numpy().astype(float), tr[label].to_numpy())
        aucs.append(roc_auc_score(te[label], m.predict_proba(te.select(feats).to_numpy().astype(float))[:, 1]))
    return np.array(aucs)

wf = []
for label in ["event", "major"]:
    base = walk_forward(PREGAME, label)
    wf.append((label, "pre-game baseline", base.mean(), 0.0, 0.0, 0))
    for g, fs in NEW.items():
        a = walk_forward(PREGAME + fs, label)
        d = a - base
        wf.append((label, f"+ {g}", a.mean(), d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum())))
    allnew = walk_forward(PREGAME + NEWF, label)
    d = allnew - base
    wf.append((label, "+ ALL new", allnew.mean(), d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum())))
    only = walk_forward(NEWF, label)
    wf.append((label, "new variables ONLY", only.mean(), (only - base).mean(), np.nan, int((only > base).sum())))

# feature impact in the final all-in model
tr = df.filter(pl.col("season") < SEASONS[-1])
m = xgb.XGBClassifier(**params).fit(tr.select(PREGAME + NEWF).to_numpy().astype(float), tr["event"].to_numpy())
te = df.filter(pl.col("season") >= SEASONS[-2])
c = m.get_booster().predict(xgb.DMatrix(te.select(PREGAME + NEWF).to_numpy().astype(float)), pred_contribs=True)[:, :-1]
imp = sorted(zip(PREGAME + NEWF, np.abs(c).mean(0)), key=lambda t: -t[1])[:20]

# ---------- report ----------
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(14); pl.Config.set_tbl_width_chars(170)
fmt = lambda d: str(d.with_columns(pl.selectors.float().round(4)))
lines = [
    f"RB-games {df.height}, seasons {SEASONS[0]}-{SEASONS[-1]}; any-injury rate {df['event'].mean():.3%}, major {df['major'].mean():.3%}",
    "",
    "Univariate (sorted by p_event; q = Benjamini-Hochberg across the new variables):",
    fmt(scan),
    "",
    "Binary variables:", *binrows,
    "",
    f"Walk-forward XGBoost: train on seasons < s, test on s, s in {WF_SEASONS[0]}-{WF_SEASONS[-1]} ({len(WF_SEASONS)} folds).",
    "dAUC = mean change vs pre-game baseline, se = std error across folds, wins = folds improved.",
    *[f"  {lab:6s} {name:24s} AUC={a:.3f}  dAUC={d:+.4f}" + ("" if np.isnan(se) else f" (se {se:.4f})")
      + f"  wins {w}/{len(WF_SEASONS)}" for lab, name, a, d, se, w in wf],
    "",
    "Top 20 mean |SHAP|, any-injury model with all variables:",
    *[f"  {f:26s} {v:.4f}{'   <- new' if f in NEWF else ''}" for f, v in imp],
]
report = "\n".join(lines)
print(report)
open(f"{OUT}/new_variables.txt", "w").write(report)

# ---------- figure ----------
fig, ax = plt.subplots(1, 2, figsize=(16, 7))
s = scan.sort("rho_event")
ax[0].barh(s["feature"], s["rho_event"], color=["#C44E52" if q < 0.05 else "#B0B0B0" for q in s["q_event"]])
ax[0].axvline(0, color="k", lw=0.8)
ax[0].set_title("New variables: Spearman with any injury next week\n(red = significant after FDR, q<0.05)")
w = [r for r in wf if r[0] == "event" and r[1].startswith("+")]
ax[1].barh([r[1] for r in w][::-1], [r[3] for r in w][::-1], xerr=[1.96 * r[4] for r in w][::-1],
           color="#4C72B0", capsize=4)
ax[1].axvline(0, color="k", lw=0.8)
ax[1].set_title(f"Walk-forward AUC change from adding each group\n(any injury, {len(WF_SEASONS)} seasons, 95% CI)")
plt.tight_layout(); plt.savefig(f"{OUT}/new_variables.png", dpi=130)
