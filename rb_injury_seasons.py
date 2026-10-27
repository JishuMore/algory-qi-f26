"""Season-to-season: does last season's workload or injury predict next season's injuries?
Also tests published theories:

  T1 "Curse of 370" (Football Outsiders): 370+ carries -> injury/decline the next year.
  T2 Orthopaedic J. Sports Med. study: 300+ carry backs missed FEWER games next season than
     150-250 carry backs (66% vs 49% missed >= 1 game).
  T3 Re-injury: prior injury to the same body part predicts the next one (Fulton et al. 2014).
  T4 Age cliff: RBs 29+ miss 5+ games far more often than 27-28 (43.5% vs 20.5%).
  T5 Yards after contact / broken tackles raise injury risk (Keefer & Kniesner, IZA DP 12556).
  T6 Career "mileage": cumulative carries, separate from age.

Unit: RB-season with 50+ carries, seasons 2013-2024, outcome measured in season+1.
"Missed due to injury" = a week his team played, he logged no snaps, and he was listed
Out/Doubtful or on reserve/IR. Healthy scratches and benchings do not count.
"""
import os

import matplotlib.pyplot as plt
import nflreadpy as nfl
import numpy as np
import polars as pl
import xgboost as xgb
from scipy.stats import fisher_exact, spearmanr, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEASONS = list(range(2013, 2026))
OUT = "rb_injury_output"
os.makedirs(OUT, exist_ok=True)

# ---------- weekly backbone: roster status, team games, snaps, injury listings ----------
sched = nfl.load_schedules(SEASONS).filter(pl.col("game_type") == "REG")
team_weeks = pl.concat([sched.select("season", "week", team="home_team"),
                        sched.select("season", "week", team="away_team")]).with_columns(
    pl.col("season", "week").cast(pl.Int32)).unique()
team_games = team_weeks.group_by("team", "season").agg(team_games=pl.len())

ros = nfl.load_rosters_weekly(SEASONS).filter((pl.col("game_type") == "REG") & (pl.col("position") == "RB")).select(
    pl.col("season", "week").cast(pl.Int32), "team", player_id="gsis_id", pfr_id="pfr_id", status="status",
    birth_date="birth_date").filter(pl.col("player_id").is_not_null()).unique(["player_id", "season", "week"])
ros = ros.join(team_weeks, on=["season", "week", "team"], how="inner")  # drop bye weeks

snaps = nfl.load_snap_counts(SEASONS).filter(pl.col("game_type") == "REG").select(
    pl.col("season", "week").cast(pl.Int32), pfr_id="pfr_player_id",
    snaps=(pl.col("offense_snaps").fill_null(0) + pl.col("st_snaps").fill_null(0)).cast(pl.Float64)).unique(
    ["pfr_id", "season", "week"])
inj = nfl.load_injuries(SEASONS).filter(pl.col("game_type") == "REG").select(
    pl.col("season", "week").cast(pl.Int32), player_id="gsis_id", rstatus="report_status",
    body=pl.col("report_primary_injury").fill_null("").str.to_lowercase()).unique(["player_id", "season", "week"])

wk = (ros.join(snaps, on=["pfr_id", "season", "week"], how="left").join(inj, on=["player_id", "season", "week"], how="left")
      .with_columns(played=(pl.col("snaps").fill_null(0) > 0),
                    listed=pl.col("rstatus").is_in(["Out", "Doubtful", "Questionable"]),
                    on_ir=(pl.col("status") == "RES")))
wk = wk.with_columns(inj_missed=(~pl.col("played") & (pl.col("rstatus").is_in(["Out", "Doubtful"]) | pl.col("on_ir"))))

BODY = {"knee": "knee", "ankle": "ankle", "hamstring": "hamstring", "foot": "foot", "shoulder": "shoulder",
        "concussion": "concussion", "groin": "groin", "calf": "calf", "back": "back", "hip": "hip"}
def body_flags(prefix):
    return [pl.col("body").str.contains(k).any().cast(pl.Int32).alias(f"{prefix}{v}") for k, v in BODY.items()]

season_inj = wk.group_by("player_id", "season").agg(
    team=pl.col("team").mode().first(), pfr_id=pl.col("pfr_id").drop_nulls().first(),
    birth_date=pl.col("birth_date").drop_nulls().first(),
    games_played=pl.col("played").sum(), inj_games_missed=pl.col("inj_missed").sum(),
    weeks_listed=pl.col("listed").sum(), on_ir=pl.col("on_ir").any().cast(pl.Int32),
    *body_flags("inj_"),
)

# ---------- season production + advanced (YAC, broken tackles) ----------
ps = nfl.load_player_stats(SEASONS).filter(pl.col("season_type") == "REG").group_by("player_id", "season").agg(
    carries=pl.col("carries").fill_null(0).sum(), receptions=pl.col("receptions").fill_null(0).sum(),
    rush_yards=pl.col("rushing_yards").fill_null(0).sum(),
).with_columns(pl.col("season").cast(pl.Int32), touches=pl.col("carries") + pl.col("receptions"),
               ypc=pl.col("rush_yards") / pl.col("carries").clip(1))
adv = nfl.load_pfr_advstats([s for s in SEASONS if s >= 2018], stat_type="rush", summary_level="season").select(
    pl.col("season").cast(pl.Int32), pfr_id="pfr_id",
    yac_att=pl.col("yac_att").cast(pl.Float64), ybc_att=pl.col("ybc_att").cast(pl.Float64),
    brk_tkl_rate=(pl.col("brk_tkl").cast(pl.Float64) / pl.col("att").cast(pl.Float64).clip(1)),
).unique(["pfr_id", "season"])
post = nfl.load_player_stats(SEASONS).filter(pl.col("season_type") == "POST").group_by("player_id", "season").agg(
    playoff_carries=pl.col("carries").fill_null(0).sum()).with_columns(pl.col("season").cast(pl.Int32))

S = (season_inj.join(ps, on=["player_id", "season"], how="left")
     .join(adv, on=["pfr_id", "season"], how="left").join(post, on=["player_id", "season"], how="left")
     .join(team_games, on=["team", "season"], how="left")
     .with_columns(pl.col("carries", "receptions", "touches", "rush_yards", "playoff_carries").fill_null(0),
                   age=(pl.date(pl.col("season"), 9, 1) - pl.col("birth_date").cast(pl.Date)).dt.total_days() / 365.25)
     .sort("player_id", "season"))
S = S.with_columns(
    career_carries_before=(pl.col("carries").cum_sum().over("player_id") - pl.col("carries")),
    prev_inj_games_missed=pl.col("inj_games_missed").shift(1).over("player_id"),
    prev_carries=pl.col("carries").shift(1).over("player_id"),
    prev_on_ir=pl.col("on_ir").shift(1).over("player_id"),
    prev_season_present=(pl.col("season").shift(1).over("player_id") == pl.col("season") - 1),
)
S = S.with_columns(pl.when(pl.col("prev_season_present")).then(pl.col(c)).otherwise(None).alias(c)
                   for c in ["prev_inj_games_missed", "prev_carries", "prev_on_ir"])

# next-season outcome (only if he is on an RB roster next season)
nxt_cols = ["games_played", "inj_games_missed", "weeks_listed", "on_ir", "ypc", "carries", "team_games",
            *[f"inj_{v}" for v in BODY.values()]]
NX = S.select("player_id", (pl.col("season") - 1).alias("season"), *[pl.col(c).alias(f"next_{c}") for c in nxt_cols])
base = S.filter((pl.col("carries") >= 50) & (pl.col("season") <= SEASONS[-2]))
active_next = base.join(NX.select("player_id", "season", flag=pl.lit(1)), on=["player_id", "season"], how="left")
D = base.join(NX, on=["player_id", "season"], how="inner").with_columns(
    next_missed_any=(pl.col("next_inj_games_missed") >= 1).cast(pl.Int32),
    next_missed5=(pl.col("next_inj_games_missed") >= 5).cast(pl.Int32),
    next_ypc_change=pl.col("next_ypc") - pl.col("ypc"),
    carry_bucket=pl.col("carries").cut([149, 249, 299, 369], labels=["50-149", "150-249", "250-299", "300-369", "370+"]),
    age_bucket=pl.col("age").cut([24.99, 26.99, 28.99], labels=["<=24", "25-26", "27-28", "29+"]),
    prev_inj_bucket=pl.col("inj_games_missed").cut([0, 2, 5], labels=["0", "1-2", "3-5", "6+"]),
)

def summarize(by):
    return D.group_by(by).agg(
        n=pl.len(), missed_any=pl.mean("next_missed_any"), missed5=pl.mean("next_missed5"),
        ir=pl.mean("next_on_ir"), mean_missed=pl.mean("next_inj_games_missed"),
        ypc_change=pl.mean("next_ypc_change"), age=pl.mean("age")).sort(by)

def logit_or(y, Xcols, d=None):
    d = (D if d is None else d).drop_nulls(Xcols)
    X = d.select(Xcols).to_numpy().astype(float); mu, sd = X.mean(0), X.std(0)
    m = LogisticRegression(C=1e6, max_iter=1000).fit((X - mu) / sd, d[y].to_numpy())
    # bootstrap CIs on odds ratio per 1 SD
    rng = np.random.default_rng(0); bs = []
    yv = d[y].to_numpy(); Xs = (X - mu) / sd
    for _ in range(300):
        i = rng.integers(0, len(yv), len(yv))
        bs.append(LogisticRegression(C=1e6, max_iter=1000).fit(Xs[i], yv[i]).coef_[0])
    bs = np.array(bs)
    return pl.DataFrame({"var": Xcols, "sd": sd, "OR_per_sd": np.exp(m.coef_[0]),
                         "lo": np.exp(np.quantile(bs, 0.025, axis=0)), "hi": np.exp(np.quantile(bs, 0.975, axis=0)),
                         "n": [len(yv)] * len(Xcols)})

# ---------- T1 / T2 ----------
t1 = summarize("carry_bucket")
surv = active_next.with_columns(carry_bucket=pl.col("carries").cut(
    [149, 249, 299, 369], labels=["50-149", "150-249", "250-299", "300-369", "370+"])).group_by("carry_bucket").agg(
    n=pl.len(), still_on_rb_roster_next_year=pl.col("flag").is_not_null().mean()).sort("carry_bucket")
heavy_list = D.filter(pl.col("carries") >= 330).select(
    "season", "player_id", "carries", "age", "next_inj_games_missed", "next_on_ir", "ypc", "next_ypc").sort("carries", descending=True)
names = nfl.load_players().select(player_id="gsis_id", name="display_name")
heavy_list = heavy_list.join(names, on="player_id", how="left").drop("player_id")
a = D.filter(pl.col("carries") >= 300)["next_missed_any"].to_numpy()
b = D.filter((pl.col("carries") >= 150) & (pl.col("carries") < 250))["next_missed_any"].to_numpy()
t2_p = fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])[1]

# ---------- T3 re-injury ----------
reinj = []
for v in BODY.values():
    had = D.filter(pl.col(f"inj_{v}") == 1); no = D.filter(pl.col(f"inj_{v}") == 0)
    r1, r0 = had[f"next_inj_{v}"].mean(), no[f"next_inj_{v}"].mean()
    p = fisher_exact([[had[f"next_inj_{v}"].sum(), had.height - had[f"next_inj_{v}"].sum()],
                      [no[f"next_inj_{v}"].sum(), no.height - no[f"next_inj_{v}"].sum()]])[1]
    reinj.append((v, had.height, r1, r0, r1 / r0 if r0 else np.nan, p))
reinj = pl.DataFrame(reinj, schema=["body_part", "n_had_it", "next_rate_if_had", "next_rate_if_not", "risk_ratio", "p"], orient="row")
t3 = summarize("prev_inj_bucket")

# ---------- T4 age ----------
t4 = summarize("age_bucket")

# ---------- adjusted models ----------
adj_any = logit_or("next_missed_any", ["carries", "age", "career_carries_before", "inj_games_missed", "receptions", "playoff_carries"])
adj_5 = logit_or("next_missed5", ["carries", "age", "career_carries_before", "inj_games_missed", "receptions", "playoff_carries"])
adj_yac = logit_or("next_missed_any", ["carries", "age", "inj_games_missed", "yac_att", "ybc_att", "brk_tkl_rate"])
adj_2yr = logit_or("next_missed_any", ["carries", "prev_carries", "age", "inj_games_missed", "prev_inj_games_missed"],
                   D.filter(pl.col("prev_carries").is_not_null()))

# ---------- T5 game-level YAC -> next-week injury ----------
gl = pl.read_parquet(f"{OUT}/rb_games_severity.parquet") if os.path.exists(f"{OUT}/rb_games_severity.parquet") else None
yac_game = None
if gl is not None:
    aw = nfl.load_pfr_advstats([s for s in SEASONS if s >= 2018], stat_type="rush", summary_level="week").filter(
        pl.col("game_type") == "REG").select(pl.col("season", "week").cast(pl.Int32), pfr_id="pfr_player_id",
        g_yac=pl.col("rushing_yards_after_contact").cast(pl.Float64),
        g_yac_att=pl.col("rushing_yards_after_contact_avg").cast(pl.Float64),
        g_brk=(pl.col("rushing_broken_tackles").fill_null(0) + pl.col("receiving_broken_tackles").fill_null(0)).cast(pl.Float64))
    g = gl.join(aw, on=["pfr_id", "season", "week"], how="inner")
    rows = []
    for f in ["g_yac", "g_yac_att", "g_brk"]:
        x = g[f].to_numpy(); m = ~np.isnan(x)
        for lab in ["event", "major"]:
            r, p = spearmanr(x[m], g[lab].to_numpy()[m]); rows.append((f, lab, int(m.sum()), r, p))
    yac_game = pl.DataFrame(rows, schema=["feature", "label", "n", "rho", "p"], orient="row")

# ---------- T6 hinge on season carries + season-level prediction ----------
def hinge_search(x, y, grid):
    s = x.std(); ll0 = None; out = []
    for k in grid:
        X = np.column_stack([x / s, np.maximum(0, x - k) / s])
        m = LogisticRegression(C=1e6, max_iter=500).fit(X, y)
        p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        out.append((k, np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)), m.coef_[0][0] / s, (m.coef_[0][0] + m.coef_[0][1]) / s))
    return max(out, key=lambda t: t[1])
hx, hy = D["carries"].cast(pl.Float64).to_numpy(), D["next_missed_any"].to_numpy()
hk = hinge_search(hx, hy, np.arange(100, 340, 10))

FEATS = ["carries", "receptions", "touches", "ypc", "age", "career_carries_before", "inj_games_missed", "weeks_listed",
         "on_ir", "games_played", "playoff_carries", "prev_carries", "prev_inj_games_missed", "prev_on_ir",
         *[f"inj_{v}" for v in BODY.values()]]
params = dict(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=10, eval_metric="logloss")
wf = {}
for label in ["next_missed_any", "next_missed5", "next_on_ir"]:
    preds, ys, preds_age = [], [], []
    for s in range(2017, SEASONS[-2] + 1):
        tr, te = D.filter(pl.col("season") < s), D.filter(pl.col("season") == s)
        m = xgb.XGBClassifier(**params).fit(tr.select(FEATS).to_numpy().astype(float), tr[label].to_numpy())
        preds.append(m.predict_proba(te.select(FEATS).to_numpy().astype(float))[:, 1])
        ma = LogisticRegression().fit(tr.select("age").fill_null(26).to_numpy(), tr[label].to_numpy())
        preds_age.append(ma.predict_proba(te.select("age").fill_null(26).to_numpy())[:, 1])
        ys.append(te[label].to_numpy())
    y, p, pa = np.concatenate(ys), np.concatenate(preds), np.concatenate(preds_age)
    wf[label] = (len(y), y.mean(), roc_auc_score(y, p), roc_auc_score(y, pa))
m_all = xgb.XGBClassifier(**params).fit(D.select(FEATS).to_numpy().astype(float), D["next_missed_any"].to_numpy())
c = m_all.get_booster().predict(xgb.DMatrix(D.select(FEATS).to_numpy().astype(float)), pred_contribs=True)[:, :-1]
imp = sorted(zip(FEATS, np.abs(c).mean(0)), key=lambda t: -t[1])[:12]

# ---------- report ----------
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(14); pl.Config.set_tbl_width_chars(170)
fmt = lambda d: str(d.with_columns(pl.selectors.float().round(3)))
lines = [
    f"RB-seasons with 50+ carries, {SEASONS[0]}-{SEASONS[-2]}, still on an RB roster next season: {D.height}",
    f"Next season: {D['next_missed_any'].mean():.1%} missed >=1 game to injury, {D['next_missed5'].mean():.1%} missed 5+, "
    f"{D['next_on_ir'].mean():.1%} went on IR",
    "",
    "T1/T2  Carries this season -> next season", fmt(t1),
    "Survivorship (share still on an RB roster next season):", fmt(surv),
    f"T2 test: 300+ carries missed>=1 next year {a.mean():.1%} (n={len(a)}) vs 150-249 {b.mean():.1%} (n={len(b)}), Fisher p={t2_p:.3f}",
    "Every 330+ carry season and what happened next:", fmt(heavy_list),
    "",
    "T3  Injury games missed this season -> next season", fmt(t3),
    "T3  Same-body-part re-injury next season", fmt(reinj.sort("p")),
    "",
    "T4  Age -> next season", fmt(t4),
    "",
    "Adjusted logistic (odds ratio per 1 SD, 95% bootstrap CI). next season missed >= 1 game:", fmt(adj_any),
    "next season missed 5+ games:", fmt(adj_5),
    "T5  with yards before/after contact and broken tackles (2018+ seasons):", fmt(adj_yac),
    "Two seasons back (carries and injuries in each of the last two seasons):", fmt(adj_2yr),
    "",
    "T5  Game level: yards after contact / broken tackles this game -> injury next week (Spearman):",
    fmt(yac_game) if yac_game is not None else "  (skipped: run rb_injury_severity.py first)",
    "",
    f"T6  Hinge search on season carries for next-season missed>=1: best k={hk[0]} carries, "
    f"slope below {hk[2]*100:+.3f} / above {hk[3]*100:+.3f} log-odds per 100 carries",
    "",
    "Walk-forward season-level XGBoost (train on earlier seasons, test 2017-2024):",
    *[f"  {k:18s} n={n} rate={r:.3f}  XGB AUC={au:.3f}   age-only AUC={aa:.3f}" for k, (n, r, au, aa) in wf.items()],
    "Top drivers (mean |SHAP|, next_missed_any):", *[f"  {f:24s} {v:.4f}" for f, v in imp],
]
report = "\n".join(lines)
print(report)
open(f"{OUT}/seasons.txt", "w").write(report)

# ---------- figure ----------
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for axi, (t, col, title) in zip(ax, [(t1, "carry_bucket", "Carries this season"), (t3, "prev_inj_bucket", "Games missed to injury this season"),
                                     (t4, "age_bucket", "Age this season")]):
    x = np.arange(t.height)
    axi.bar(x - 0.2, t["missed_any"], 0.4, label="missed >=1 next season", color="#4C72B0")
    axi.bar(x + 0.2, t["missed5"], 0.4, label="missed 5+ next season", color="#B33F40")
    axi.set_xticks(x, [f"{v}\n(n={n})" for v, n in zip(t[col], t["n"])]); axi.set_title(title)
ax[0].legend(); plt.tight_layout(); plt.savefig(f"{OUT}/seasons.png", dpi=130)
