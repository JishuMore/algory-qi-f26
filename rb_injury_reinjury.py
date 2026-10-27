"""Previous injury x distance run -> new injury (re-injury). Non-linear.

Question: after an RB comes back from an injury, does the chance he gets hurt again (especially the
SAME body part) depend on how far he has run since returning? Is there a distance where it spikes?

Distance
  est_dist  NGS tracked distance (efficiency x rush yards) when NGS has the game (10+ carries,
            2016+); otherwise carries x the player's median NGS yards-per-carry (league median
            if he has none). Robustness version: scrimmage yards.
  dist_since_return   cumulative est_dist from the first game after his last injury listing
                      through this game (never-injured backs: since his first game in the data)
  return_load_ratio   distance per game since return / his distance per game before the injury
                      (ramp-up theory: coming back too hot)

Labels (from rb_games_sensitivity.parquet rows: RB-games, 3+ carries, not on report this week)
  any_4wk        any new listing within 4 weeks
  same_part_4wk  new listing within 4 weeks for the SAME body part as his last injury
  soft_4wk / contact_4wk as before

Methods
  A binned rates with Wilson CIs, B spline logistic (likelihood-ratio vs linear vs none, with
  controls), C re-injury-free "survival" curve over distance since return (discrete hazard),
  D hinge/breakpoint search + clustered bootstrap, E walk-forward XGBoost ΔAUC + 2-D partial
  dependence (distance x prior injury type).
"""
import matplotlib.pyplot as plt
import nflreadpy as nfl
import numpy as np
import polars as pl
import xgboost as xgb
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import SplineTransformer

OUT = "rb_injury_output"
rng = np.random.default_rng(0)
g = pl.read_parquet(f"{OUT}/rb_games_sensitivity.parquet")
SEASONS = sorted(g["season"].unique().to_list())
key = lambda: pl.col("season") * 100 + pl.col("week")

# ---------- weekly distance series (every RB game, not just modeled rows) ----------
ws = nfl.load_player_stats(SEASONS).filter((pl.col("season_type") == "REG") & (pl.col("position") == "RB")).select(
    "player_id", pl.col("season", "week").cast(pl.Int32),
    carries=pl.col("carries").fill_null(0).cast(pl.Float64),
    yards=(pl.col("rushing_yards").fill_null(0) + pl.col("receiving_yards").fill_null(0)).cast(pl.Float64))
ngs = nfl.load_nextgen_stats([s for s in SEASONS if s >= 2016], stat_type="rushing").filter(
    (pl.col("season_type") == "REG") & (pl.col("week") > 0) & (pl.col("rush_yards") > 0)).select(
    player_id="player_gsis_id", season=pl.col("season").cast(pl.Int32), week=pl.col("week").cast(pl.Int32),
    ngs_dist=pl.col("efficiency") * pl.col("rush_yards"),
    dpc=pl.col("efficiency") * pl.col("rush_yards") / pl.col("rush_attempts")).unique(["player_id", "season", "week"])
league_dpc = ngs["dpc"].median()
ws = (ws.join(ngs, on=["player_id", "season", "week"], how="left")
        .with_columns(player_dpc=pl.col("dpc").median().over("player_id"))
        .with_columns(est_dist=pl.coalesce("ngs_dist", pl.col("carries") * pl.col("player_dpc").fill_null(league_dpc)),
                      k=key()).sort("player_id", "k")
        .with_columns(cum_dist=pl.col("est_dist").cum_sum().over("player_id"),
                      cum_yards=pl.col("yards").cum_sum().over("player_id"),
                      cum_games=pl.col("k").cum_count().over("player_id")))

# ---------- injury listings with body part ----------
PARTS = ["knee", "ankle", "hamstring", "foot", "shoulder", "concussion", "groin", "calf", "back", "hip", "toe", "quad"]
SOFT = {"hamstring", "groin", "calf", "quad", "back", "hip"}
inj = nfl.load_injuries(SEASONS).filter(
    (pl.col("game_type") == "REG") & pl.col("report_status").is_in(["Out", "Doubtful", "Questionable"])).select(
    player_id="gsis_id", season=pl.col("season").cast(pl.Int32), week=pl.col("week").cast(pl.Int32),
    body=pl.col("report_primary_injury").fill_null("").str.to_lowercase()).unique(["player_id", "season", "week"])
part_expr = pl.lit("other")
for p in reversed(PARTS):
    part_expr = pl.when(pl.col("body").str.contains(p)).then(pl.lit(p)).otherwise(part_expr)
inj = inj.with_columns(part=part_expr, k=key()).sort("k")

# cumulative distance at the moment of each listing (asof: last game on or before the listing week)
inj_at = inj.join_asof(ws.select("player_id", "k", "cum_dist", "cum_yards", "cum_games").sort("k"),
                       on="k", by="player_id", strategy="backward")
# pre-injury baseline: distance per game over his last 4 games before the listing
ws = ws.with_columns(dist_l4=pl.col("est_dist").rolling_mean(4, min_samples=1).over("player_id"))
inj_at = inj_at.join_asof(ws.select("player_id", "k", pre_injury_dpg="dist_l4").sort("k"),
                          on="k", by="player_id", strategy="backward")

# ---------- attach "last injury" to each modeled game ----------
g = g.with_columns(k=key()).sort("k")
g = g.join(ws.select("player_id", "season", "week", "est_dist", "cum_dist", "cum_yards", "cum_games", "dist_l4"),
           on=["player_id", "season", "week"], how="left")
last = inj_at.select("player_id", "k", last_k="k", last_part="part", inj_cum_dist="cum_dist",
                     inj_cum_yards="cum_yards", inj_cum_games="cum_games", pre_injury_dpg="pre_injury_dpg").sort("k")
g = g.sort("k").join_asof(last, on="k", by="player_id", strategy="backward", allow_exact_matches=False)
first_game = ws.group_by("player_id").agg(first_cum=pl.col("cum_dist").first())
g = g.join(first_game, on="player_id", how="left").with_columns(
    prior_injured=pl.col("last_part").is_not_null().cast(pl.Int32),
    last_type=pl.when(pl.col("last_part").is_null()).then(pl.lit("none"))
    .when(pl.col("last_part").is_in(list(SOFT))).then(pl.lit("soft")).otherwise(pl.lit("contact")),
)
g = g.with_columns(
    dist_since_return=pl.when(pl.col("prior_injured") == 1)
    .then(pl.col("cum_dist") - pl.col("inj_cum_dist").fill_null(0)).otherwise(pl.col("cum_dist")),
    yards_since_return=pl.when(pl.col("prior_injured") == 1)
    .then(pl.col("cum_yards") - pl.col("inj_cum_yards").fill_null(0)).otherwise(pl.col("cum_yards")),
    games_since_return=pl.when(pl.col("prior_injured") == 1)
    .then(pl.col("cum_games") - pl.col("inj_cum_games").fill_null(0)).otherwise(pl.col("cum_games")),
).with_columns(
    return_load_ratio=pl.when(pl.col("prior_injured") == 1).then(
        (pl.col("dist_since_return") / pl.col("games_since_return").clip(1)) / pl.col("pre_injury_dpg").clip(1)),
)

# same-part new listing within 4 weeks
nxt = (g.select("player_id", "season", "week", "last_part")
       .join(inj.select("player_id", "season", lw="week", npart="part"), on=["player_id", "season"])
       .filter((pl.col("lw") > pl.col("week")) & (pl.col("lw") <= pl.col("week") + 4) & (pl.col("npart") == pl.col("last_part")))
       .select("player_id", "season", "week").unique().with_columns(same_part_4wk=pl.lit(1)))
g = g.join(nxt, on=["player_id", "season", "week"], how="left").with_columns(pl.col("same_part_4wk").fill_null(0))
g = g.filter(pl.col("dist_since_return").is_not_null() & (pl.col("dist_since_return") >= 0))
inj_rows = g.filter(pl.col("prior_injured") == 1)

# ---------- A. binned rates ----------
def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h

EDGES = [0, 250, 500, 1000, 1500, 2500, 4000, 6000, 1e9]
def binned(d, col, lab, edges=EDGES):
    t = (d.with_columns(b=pl.col(col).cut(edges[1:-1])).group_by("b")
         .agg(n=pl.len(), x=pl.col(col).median(), k=pl.sum(lab)).sort("x"))
    lo, hi = wilson(t["k"].to_numpy(), t["n"].to_numpy())
    return t.with_columns(rate=pl.col("k") / pl.col("n"), lo=pl.Series(lo), hi=pl.Series(hi))

# ---------- B. spline logistic vs linear vs none ----------
CONTROLS = ["games_since_return", "age", "snap_pct"]
def ll(m, X, y):
    p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

def spline_test(d, xcol, lab, controls=CONTROLS, knots=5):
    d = d.drop_nulls([xcol, *controls]).drop_nans([xcol]).with_columns(pl.col(xcol).clip(0))
    x = np.log1p(d[xcol].to_numpy())[:, None]  # log scale: distance is heavy-tailed
    C = d.select(controls).to_numpy().astype(float); C = (C - C.mean(0)) / C.std(0)
    y = d[lab].to_numpy()
    sp = SplineTransformer(n_knots=knots, degree=3, include_bias=False).fit(x)
    fit = lambda X: LogisticRegression(C=1e4, max_iter=3000).fit(X, y)
    X0, X1, X2 = C, np.column_stack([C, (x - x.mean()) / x.std()]), np.column_stack([C, sp.transform(x)])
    m0, m1, m2 = fit(X0), fit(X1), fit(X2)
    l0, l1, l2 = ll(m0, X0, y), ll(m1, X1, y), ll(m2, X2, y)
    df_lin, df_sp = 1, X2.shape[1] - X0.shape[1]
    # fitted curve at control means (0 after standardizing)
    grid = np.linspace(x.min(), np.quantile(x, 0.99), 60)[:, None]
    Xg = np.column_stack([np.zeros((60, C.shape[1])), sp.transform(grid)])
    curve = m2.predict_proba(Xg)[:, 1]
    boots = []
    for _ in range(150):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() < 5:
            continue
        mb = LogisticRegression(C=1e4, max_iter=3000).fit(X2[i], y[i]); boots.append(mb.predict_proba(Xg)[:, 1])
    lo, hi = np.quantile(np.array(boots), [0.05, 0.95], axis=0)
    return dict(n=len(y), events=int(y.sum()),
                p_linear=chi2.sf(2 * (l1 - l0), df_lin),
                p_spline_vs_none=chi2.sf(2 * (l2 - l0), df_sp),
                p_nonlinear=chi2.sf(2 * (l2 - l1), df_sp - df_lin),
                grid=np.expm1(grid[:, 0]), curve=curve, lo=lo, hi=hi)

spl = {}
for name, d, lab in [("same-part re-injury | prior injury", inj_rows, "same_part_4wk"),
                     ("any new injury | prior injury", inj_rows, "any_4wk"),
                     ("any new injury | never injured", g.filter(pl.col("prior_injured") == 0), "any_4wk"),
                     ("soft-tissue | prior soft-tissue", inj_rows.filter(pl.col("last_type") == "soft"), "soft_4wk"),
                     ("contact | prior contact", inj_rows.filter(pl.col("last_type") == "contact"), "contact_4wk")]:
    spl[name] = spline_test(d, "dist_since_return", lab)
spl_y = spline_test(inj_rows, "yards_since_return", "same_part_4wk")
spl_ramp = spline_test(inj_rows.filter(pl.col("return_load_ratio").is_not_null()), "return_load_ratio", "same_part_4wk")

# ---------- C. re-injury-free curve over distance since return (discrete hazard) ----------
# per-row 1-week same-part hazard: same part listed NEXT week
nx1 = (g.select("player_id", "season", "week", "last_part")
       .join(inj.select("player_id", "season", lw="week", npart="part"), on=["player_id", "season"])
       .filter((pl.col("lw") == pl.col("week") + 1) & (pl.col("npart") == pl.col("last_part")))
       .select("player_id", "season", "week").unique().with_columns(same_part_1wk=pl.lit(1)))
inj_rows = inj_rows.join(nx1, on=["player_id", "season", "week"], how="left").with_columns(pl.col("same_part_1wk").fill_null(0))
HZ = [0, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 9000, 1e9]
def hazard(d):
    t = (d.with_columns(b=pl.col("dist_since_return").cut(HZ[1:-1])).group_by("b")
         .agg(at_risk=pl.len(), events=pl.sum("same_part_1wk"), x=pl.col("dist_since_return").median()).sort("x"))
    h = (t["events"] / t["at_risk"]).to_numpy()
    return t.with_columns(hazard=pl.Series(h), survival=pl.Series(np.cumprod(1 - h)))
hz_all, hz_soft, hz_contact = hazard(inj_rows), hazard(inj_rows.filter(pl.col("last_type") == "soft")), \
    hazard(inj_rows.filter(pl.col("last_type") == "contact"))
hz_part = []
for p in ["knee", "ankle", "hamstring", "foot", "shoulder", "concussion"]:
    d = inj_rows.filter(pl.col("last_part") == p)
    for lo_, hi_ in [(0, 500), (500, 1500), (1500, 4000), (4000, 1e9)]:
        s = d.filter((pl.col("dist_since_return") >= lo_) & (pl.col("dist_since_return") < hi_))
        hz_part.append((p, f"{int(lo_)}-{'inf' if hi_ > 1e8 else int(hi_)}", s.height,
                        s["same_part_4wk"].mean() if s.height else np.nan))
hz_part = pl.DataFrame(hz_part, schema=["prior_part", "dist_since_return", "n", "same_part_4wk_rate"], orient="row")

# ---------- D. breakpoint search ----------
def hinge(x, y, grid):
    s = x.std(); best = None
    for k in grid:
        X = np.column_stack([x / s, np.maximum(0, x - k) / s])
        m = LogisticRegression(C=1e6, max_iter=1000).fit(X, y); l = ll(m, X, y)
        if best is None or l > best[1]:
            best = (k, l, m.coef_[0][0] / s, (m.coef_[0][0] + m.coef_[0][1]) / s)
    return best
x = inj_rows["dist_since_return"].to_numpy(); y = inj_rows["same_part_4wk"].to_numpy(); pid = inj_rows["player_id"].to_numpy()
grid = np.quantile(x, np.linspace(0.05, 0.9, 18))
hk = hinge(x, y, grid)
players = np.unique(pid); idx = {p: np.where(pid == p)[0] for p in players}
bk = []
for _ in range(80):
    i = np.concatenate([idx[p] for p in rng.choice(players, len(players))])
    if y[i].sum() >= 5:
        bk.append(hinge(x[i], y[i], grid)[0])
bk = np.array(bk)

# ---------- E. walk-forward XGBoost: does distance-since-return add to history? ----------
HIST = ["prior_contact_inj", "prior_soft_inj", "games_since_inj", "inj_weeks_this_season", "prev_season_inj_weeks",
        "games_since_return", "age", "snap_pct"]
DIST = ["dist_since_return", "yards_since_return", "return_load_ratio", "est_dist", "dist_l4"]
params = dict(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=20, eval_metric="logloss")
def wf(d, feats, lab):
    ys, ps = [], []
    for s in range(2018, SEASONS[-1] + 1):
        tr, te = d.filter(pl.col("season") < s), d.filter(pl.col("season") == s)
        if te[lab].sum() < 2:
            continue
        m = xgb.XGBClassifier(**params).fit(tr.select(feats).to_numpy().astype(float), tr[lab].to_numpy())
        ys.append(te[lab].to_numpy()); ps.append(m.predict_proba(te.select(feats).to_numpy().astype(float))[:, 1])
    return roc_auc_score(np.concatenate(ys), np.concatenate(ps))
wf_rows = [(lab, wf(inj_rows, HIST, lab), wf(inj_rows, DIST, lab), wf(inj_rows, HIST + DIST, lab))
           for lab in ["same_part_4wk", "any_4wk", "soft_4wk", "contact_4wk"]]

# 2-D partial dependence: distance since return x prior injury type (same-part model)
pd_rows = inj_rows.with_columns(is_soft=(pl.col("last_type") == "soft").cast(pl.Int32))
F2 = HIST + DIST + ["is_soft"]
m2 = xgb.XGBClassifier(**params).fit(pd_rows.select(F2).to_numpy().astype(float), pd_rows["same_part_4wk"].to_numpy())
Xb = pd_rows.select(F2).to_numpy().astype(float)
dgrid = np.quantile(x, np.linspace(0.02, 0.97, 25))
pdp = {}
for soft in [0, 1]:
    vals = []
    for dv in dgrid:
        Xc = Xb.copy(); Xc[:, F2.index("dist_since_return")] = dv; Xc[:, F2.index("is_soft")] = soft
        vals.append(m2.predict_proba(Xc)[:, 1].mean())
    pdp[soft] = np.array(vals)

# ---------- report ----------
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(12); pl.Config.set_tbl_width_chars(170)
fmt = lambda d: str(d.with_columns(pl.selectors.float().round(4)))
def sline(name, r):
    return (f"  {name:38s} n={r['n']:5d} events={r['events']:4d}  linear p={r['p_linear']:.3f}  "
            f"spline vs none p={r['p_spline_vs_none']:.3f}  non-linearity p={r['p_nonlinear']:.3f}")
lines = [
    f"RB-games {g.height}; with a prior injury listing {inj_rows.height} "
    f"(soft {inj_rows.filter(pl.col('last_type') == 'soft').height}, contact {inj_rows.filter(pl.col('last_type') == 'contact').height})",
    f"Median distance per carry (NGS) {league_dpc:.1f} yds; median distance since return {inj_rows['dist_since_return'].median():.0f} yds",
    f"Same-part re-injury within 4 wks: {inj_rows['same_part_4wk'].mean():.3%} of games after a prior injury",
    "",
    "A. Binned rates: same-part re-injury within 4 wks by distance run since return (prior-injured backs)",
    fmt(binned(inj_rows, "dist_since_return", "same_part_4wk")),
    "   any new injury within 4 wks, prior-injured backs:", fmt(binned(inj_rows, "dist_since_return", "any_4wk")),
    "   any new injury within 4 wks, never-injured backs (distance = career in data):",
    fmt(binned(g.filter(pl.col("prior_injured") == 0), "dist_since_return", "any_4wk")),
    "",
    "B. Spline logistic on log(distance since return), controls: games since return, age, snap %",
    "   (non-linearity p tests spline vs straight line; spline vs none tests ANY relationship)",
    *[sline(k, v) for k, v in spl.items()],
    sline("same-part | yards since return", spl_y),
    sline("same-part | return load ratio (ramp-up)", spl_ramp),
    "",
    "C. Same-part re-injury hazard per game, by distance since return",
    "   all prior injuries:", fmt(hz_all.select("b", "at_risk", "events", "hazard", "survival")),
    "   prior soft-tissue:", fmt(hz_soft.select("b", "at_risk", "events", "hazard", "survival")),
    "   prior contact:", fmt(hz_contact.select("b", "at_risk", "events", "hazard", "survival")),
    "   by body part (same-part re-injury within 4 wks):", fmt(hz_part),
    "",
    f"D. Breakpoint (same-part, prior-injured): best k = {hk[0]:.0f} yds since return; slope below "
    f"{hk[2]*1000:+.3f}, above {hk[3]*1000:+.3f} log-odds per 1000 yds; bootstrap 90% range "
    f"{np.quantile(bk, 0.05):.0f}-{np.quantile(bk, 0.95):.0f} yds (wide range = no stable threshold)",
    "",
    "E. Walk-forward AUC (test 2018+), prior-injured backs only:",
    *[f"  {lab:14s} history={a:.3f}  distance-only={b:.3f}  history+distance={c:.3f}  d={c - a:+.3f}"
      for lab, a, b, c in wf_rows],
]
report = "\n".join(lines)
print(report)
open(f"{OUT}/reinjury.txt", "w").write(report)

# ---------- figure ----------
fig, ax = plt.subplots(2, 3, figsize=(19, 10))
for a, (name, c) in zip([ax[0, 0], ax[0, 0], ax[0, 1], ax[0, 1], ax[0, 1]],
                        [("same-part re-injury | prior injury", "#B33F40"), ("any new injury | prior injury", "#4C72B0"),
                         ("soft-tissue | prior soft-tissue", "#DD8452"), ("contact | prior contact", "#55A868"),
                         ("any new injury | never injured", "#8C8C8C")]):
    r = spl[name]
    a.plot(r["grid"], r["curve"], color=c, label=name); a.fill_between(r["grid"], r["lo"], r["hi"], color=c, alpha=0.15)
for a, t in [(ax[0, 0], "Spline fit vs distance since return"), (ax[0, 1], "By injury type (spline fit)")]:
    a.set_xscale("symlog", linthresh=100); a.set_xlabel("distance run since return (yds)"); a.legend(fontsize=8); a.set_title(t)
for t, c, lab in [(hz_all, "#333333", "all"), (hz_soft, "#DD8452", "prior soft"), (hz_contact, "#55A868", "prior contact")]:
    ax[0, 2].step(t["x"], t["survival"], where="post", color=c, label=lab)
ax[0, 2].set_xscale("symlog", linthresh=100); ax[0, 2].set_title("Same-part re-injury-free curve over distance")
ax[0, 2].set_xlabel("distance since return (yds)"); ax[0, 2].legend()
b = binned(inj_rows, "dist_since_return", "same_part_4wk")
ax[1, 0].errorbar(b["x"], b["rate"], yerr=[b["rate"] - b["lo"], b["hi"] - b["rate"]], marker="o", capsize=3, color="#B33F40")
ax[1, 0].axvline(hk[0], ls=":", color="k"); ax[1, 0].set_xscale("symlog", linthresh=100)
ax[1, 0].set_title("Same-part re-injury within 4 wks (binned, 95% CI)\ndotted = best breakpoint")
for soft, c, lab in [(0, "#55A868", "prior contact injury"), (1, "#DD8452", "prior soft-tissue injury")]:
    ax[1, 1].plot(dgrid, pdp[soft], color=c, marker=".", label=lab)
ax[1, 1].set_xscale("symlog", linthresh=100); ax[1, 1].legend(); ax[1, 1].set_title("XGBoost partial dependence (same-part re-injury)")
r = spl_ramp
ax[1, 2].plot(r["grid"], r["curve"], color="#8172B3"); ax[1, 2].fill_between(r["grid"], r["lo"], r["hi"], color="#8172B3", alpha=0.15)
ax[1, 2].axvline(1, ls=":", color="k"); ax[1, 2].set_xlim(0, 3)
ax[1, 2].set_title("Ramp-up: load per game since return / pre-injury load"); ax[1, 2].set_xlabel("ratio (1 = same as before)")
plt.tight_layout(); plt.savefig(f"{OUT}/reinjury.png", dpi=130)
