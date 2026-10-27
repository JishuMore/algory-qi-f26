"""Combine the validated findings with yardage / carries: does running volume add anything?

Needs the caches written by rb_injury_sensitivity.py, rb_injury_distance.py, rb_injury_seasons.py.

Feature sets
  FINDINGS  what held up earlier: injury history (prior contact / soft-tissue listings, games since
            last listing, weeks listed this season, last season's games missed to injury),
            plus role context (snap share, touches per snap, receptions, last season's playoff
            carries, age)
  YARDAGE   how much he ran: carries and scrimmage / rush yards this game, last 3 games,
            season to date, last season, career carries (and NGS distance on the NGS subset)

Questions
  1. Walk-forward AUC: FINDINGS vs YARDAGE vs FINDINGS + YARDAGE (game and season level)
  2. Interaction: does yardage matter MORE for backs with an injury history?
  3. Risk tiers: injury history x recent workload
"""
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

OUT = "rb_injury_output"
rng = np.random.default_rng(0)

g = pl.read_parquet(f"{OUT}/rb_games_sensitivity.parquet")
dist = pl.read_parquet(f"{OUT}/rb_games_distance.parquet").select(
    "player_id", "season", "week", "game_yards", "wk_rush_yards", "prior_season_carries", "prior_season_yards",
    "prior_l3_yards", "prior_l3_carries", "ngs_dist", "prior_season_ngs_dist")
S = pl.read_parquet(f"{OUT}/rb_seasons.parquet")
prev = S.select("player_id", (pl.col("season") + 1).alias("season"),
                prev_ssn_inj_missed="inj_games_missed", prev_ssn_receptions="receptions",
                prev_ssn_playoff_carries="playoff_carries", prev_ssn_carries="carries",
                prev_ssn_rush_yards="rush_yards").unique(["player_id", "season"])
g = (g.join(dist, on=["player_id", "season", "week"], how="left")
      .join(prev, on=["player_id", "season"], how="left")
      .with_columns(history=((pl.col("prior_contact_inj") + pl.col("prior_soft_inj")) > 0).cast(pl.Int32)))

FINDINGS = ["prior_contact_inj", "prior_soft_inj", "games_since_inj", "inj_weeks_this_season",
            "prev_season_inj_weeks", "prev_ssn_inj_missed", "snap_pct", "touches_per_snap", "targets",
            "prev_ssn_receptions", "prev_ssn_playoff_carries", "age"]
YARDAGE = ["carries", "game_yards", "wk_rush_yards", "carries_l3", "hits_l3", "prior_l3_carries", "prior_l3_yards",
           "prior_season_carries", "prior_season_yards", "career_carries", "prev_ssn_carries", "prev_ssn_rush_yards"]
NGS = ["ngs_dist", "prior_season_ngs_dist"]
LABELS = {"any_1wk": "any injury next week", "any_4wk": "any injury within 4 wks",
          "contact_4wk": "contact injury within 4 wks", "major": "major injury"}

params = dict(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=20, eval_metric="logloss")

def walk_forward(d, feats, label, first, kind="xgb"):
    out = []
    for s in sorted(d["season"].unique().to_list()):
        if s < first:
            continue
        tr, te = d.filter(pl.col("season") < s), d.filter(pl.col("season") == s)
        if te[label].sum() < 3:
            continue
        Xtr, Xte = tr.select(feats).to_numpy().astype(float), te.select(feats).to_numpy().astype(float)
        if kind == "xgb":
            m = xgb.XGBClassifier(**params).fit(Xtr, tr[label].to_numpy())
        else:
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=0.5, max_iter=2000)).fit(Xtr, tr[label].to_numpy())
        out.append((te[label].to_numpy(), m.predict_proba(Xte)[:, 1], te["player_id"].to_numpy()))
    y = np.concatenate([o[0] for o in out]); p = np.concatenate([o[1] for o in out])
    pid = np.concatenate([o[2] for o in out])
    return y, p, pid

def compare(d, sets, label, first, kind="xgb", n_boot=300):
    preds = {k: walk_forward(d, f, label, first, kind) for k, f in sets.items()}
    y, pid = preds[next(iter(sets))][0], preds[next(iter(sets))][2]
    players = np.unique(pid); idx = {p: np.where(pid == p)[0] for p in players}
    res = {k: roc_auc_score(y, v[1]) for k, v in preds.items()}
    # player-clustered bootstrap CI for (combined - findings)
    keys = list(sets)
    diffs = []
    for _ in range(n_boot):
        i = np.concatenate([idx[p] for p in rng.choice(players, len(players))])
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], preds[keys[2]][1][i]) - roc_auc_score(y[i], preds[keys[0]][1][i]))
    return res, np.quantile(diffs, [0.025, 0.975]), len(y), y.mean()

game_rows = []
for lab in LABELS:
    for kind in ["xgb", "logit"]:
        res, ci, n, rate = compare(g, {"findings": FINDINGS, "yardage": YARDAGE, "findings+yardage": FINDINGS + YARDAGE},
                                   lab, 2018, kind)
        game_rows.append((lab, kind, n, rate, res["findings"], res["yardage"], res["findings+yardage"],
                          res["findings+yardage"] - res["findings"], ci[0], ci[1]))
ngs_rows = g.filter(pl.col("ngs_dist").is_not_null())
for lab in ["any_1wk", "any_4wk", "major"]:
    res, ci, n, rate = compare(ngs_rows, {"findings": FINDINGS, "yardage+NGS": YARDAGE + NGS,
                                          "findings+yardage+NGS": FINDINGS + YARDAGE + NGS}, lab, 2018, "logit")
    game_rows.append((lab + " [NGS games]", "logit", n, rate, res["findings"], res["yardage+NGS"],
                      res["findings+yardage+NGS"], res["findings+yardage+NGS"] - res["findings"], ci[0], ci[1]))
game_tab = pl.DataFrame(game_rows, schema=["label", "model", "n_test", "rate", "AUC_findings", "AUC_yardage",
                                           "AUC_combined", "dAUC_from_yardage", "ci_lo", "ci_hi"], orient="row")

# ---------- interaction: does yardage matter more for previously injured backs? ----------
def interaction(d, x, lab, hist="history", n_boot=300):
    d = d.drop_nulls([x])
    xs = (d[x].to_numpy() - d[x].mean()) / d[x].std()
    h = d[hist].to_numpy().astype(float); y = d[lab].to_numpy()
    X = np.column_stack([xs, h, xs * h])
    fit = lambda X, y: LogisticRegression(C=1e6, max_iter=1000).fit(X, y).coef_[0]
    b = fit(X, y)
    bs = np.array([fit(X[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))])
    lo, hi = np.quantile(bs[:, 2], [0.025, 0.975])
    return (x, lab, len(y), np.exp(b[0]), np.exp(b[0] + b[2]), np.exp(b[2]), np.exp(lo), np.exp(hi))

inter = pl.DataFrame([interaction(g, x, lab) for x in ["prior_l3_yards", "prior_season_yards", "prior_season_carries",
                                                     "hits_l3", "career_carries"]
                      for lab in ["any_4wk", "contact_4wk", "major"]],
                     schema=["yardage_var", "label", "n", "OR_per_sd_no_history", "OR_per_sd_with_history",
                             "interaction_OR", "int_lo", "int_hi"], orient="row")

# ---------- risk tiers ----------
g = g.with_columns(workload_tercile=pl.col("prior_l3_yards").qcut(3, labels=["low", "mid", "high"]),
                   hist_label=pl.when(pl.col("history") == 1).then(pl.lit("injury history")).otherwise(pl.lit("clean")))
tiers = g.drop_nulls("workload_tercile").group_by("hist_label", "workload_tercile").agg(
    n=pl.len(), any_1wk=pl.mean("any_1wk"), any_4wk=pl.mean("any_4wk"),
    contact_4wk=pl.mean("contact_4wk"), major=pl.mean("major")).sort("hist_label", "workload_tercile")

# ---------- season level ----------
S_FIND = ["inj_games_missed", "weeks_listed", "on_ir", "receptions", "playoff_carries", "prev_inj_games_missed",
          "age", "inj_knee", "inj_ankle", "inj_hamstring", "inj_calf", "inj_foot"]
S_YARD = ["carries", "rush_yards", "touches", "ypc", "career_carries_before", "prev_carries"]
S = S.with_columns(history=(pl.col("inj_games_missed") > 0).cast(pl.Int32))
season_rows = []
for lab in ["next_missed_any", "next_missed5", "next_on_ir"]:
    for kind in ["logit", "xgb"]:
        res, ci, n, rate = compare(S, {"findings": S_FIND, "yardage": S_YARD, "findings+yardage": S_FIND + S_YARD},
                                   lab, 2017, kind)
        season_rows.append((lab, kind, n, rate, res["findings"], res["yardage"], res["findings+yardage"],
                            res["findings+yardage"] - res["findings"], ci[0], ci[1]))
season_tab = pl.DataFrame(season_rows, schema=["label", "model", "n_test", "rate", "AUC_findings", "AUC_yardage",
                                               "AUC_combined", "dAUC_from_yardage", "ci_lo", "ci_hi"], orient="row")
s_inter = pl.DataFrame([interaction(S, x, lab) for x in ["carries", "rush_yards", "touches"]
                        for lab in ["next_missed_any", "next_missed5"]],
                       schema=["yardage_var", "label", "n", "OR_per_sd_no_history", "OR_per_sd_with_history",
                               "interaction_OR", "int_lo", "int_hi"], orient="row")
s_tiers = S.with_columns(carry_grp=pl.col("carries").cut([149, 249], labels=["50-149", "150-249", "250+"]),
                         hist_label=pl.when(pl.col("history") == 1).then(pl.lit("missed games this season"))
                         .otherwise(pl.lit("no missed games"))).group_by("hist_label", "carry_grp").agg(
    n=pl.len(), next_missed_any=pl.mean("next_missed_any"), next_missed5=pl.mean("next_missed5"),
    next_on_ir=pl.mean("next_on_ir")).sort("hist_label", "carry_grp")

# ---------- report ----------
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(14); pl.Config.set_tbl_width_chars(175)
fmt = lambda d: str(d.with_columns(pl.selectors.float().round(3)))
lines = [
    f"GAME LEVEL: {g.height} RB-games, walk-forward test seasons 2018+",
    "dAUC_from_yardage = AUC(findings+yardage) - AUC(findings); CI = 95% player-clustered bootstrap",
    fmt(game_tab), "",
    "Interaction: odds ratio per 1 SD of yardage, for backs without vs with prior injury listings.",
    "interaction_OR > 1 (CI excluding 1) would mean volume is riskier for previously injured backs.",
    fmt(inter), "",
    "Risk tiers (injury history x yards over previous 3 games):", fmt(tiers), "",
    f"SEASON LEVEL: {S.height} RB-seasons, walk-forward test seasons 2017+ (outcome = next season)",
    fmt(season_tab), "",
    "Season interaction (history = missed games to injury this season):", fmt(s_inter), "",
    "Season risk tiers:", fmt(s_tiers),
]
report = "\n".join(lines)
print(report)
open(f"{OUT}/combined.txt", "w").write(report)

# ---------- figure ----------
fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
gt = game_tab.filter(pl.col("model") == "logit")
x = np.arange(gt.height)
for i, (col, c, name) in enumerate([("AUC_findings", "#4C72B0", "findings"), ("AUC_yardage", "#DD8452", "yardage/carries"),
                                    ("AUC_combined", "#55A868", "combined")]):
    ax[0].bar(x + (i - 1) * 0.27, gt[col], 0.27, color=c, label=name)
ax[0].axhline(0.5, color="k", ls="--", lw=0.8); ax[0].set_ylim(0.45, 0.7)
ax[0].set_xticks(x, gt["label"], rotation=30, ha="right"); ax[0].legend(); ax[0].set_title("Game level: walk-forward AUC")
for hl, c in [("clean", "#4C72B0"), ("injury history", "#B33F40")]:
    t = tiers.filter(pl.col("hist_label") == hl)
    ax[1].plot(["low", "mid", "high"], t["any_4wk"], marker="o", color=c, label=f"{hl}: any 4 wks")
    ax[1].plot(["low", "mid", "high"], t["major"], marker="s", ls="--", color=c, label=f"{hl}: major")
ax[1].set_xlabel("yards over previous 3 games (tercile)"); ax[1].set_ylabel("rate"); ax[1].legend(fontsize=8)
ax[1].set_title("Injury history x recent workload")
st = season_tab.filter(pl.col("model") == "logit"); x = np.arange(st.height)
for i, (col, c) in enumerate([("AUC_findings", "#4C72B0"), ("AUC_yardage", "#DD8452"), ("AUC_combined", "#55A868")]):
    ax[2].bar(x + (i - 1) * 0.27, st[col], 0.27, color=c)
ax[2].axhline(0.5, color="k", ls="--", lw=0.8); ax[2].set_ylim(0.4, 0.7)
ax[2].set_xticks(x, st["label"]); ax[2].set_title("Season level (next season): walk-forward AUC")
plt.tight_layout(); plt.savefig(f"{OUT}/combined.png", dpi=130)
