"""Benchmark tweaking: does a workload-injury correlation appear under ANY reasonable definition?

Builds on rb_games_severity.parquet (run rb_injury_corr.py then rb_injury_severity.py).

Varies three things:
  labels    any listing within next 1 / 2 / 4 weeks, Out-only next week, soft-tissue injury within
            4 weeks, contact injury within 4 weeks, Major (3+ games / IR / season over)
  windows   carries / hits / touches / snaps over the last 1, 2, 3, 4, 6, 8 games, season to date,
            and acute:chronic workload ratios (ACWR = this game / mean of previous 4)
  filters   minimum carries this game: 3 (all), 10 (regular), 15 (lead back)

Untested-until-now variables:
  ACWR (sports-science spike metric), practice participation this week (limited / DNP while
  still playing), prior soft-tissue injuries, last season's playoff workload.

Multiple testing guard: every (label, filter, feature) correlation is estimated on DISCOVERY
seasons 2013-2020, BH-corrected there, and only counted if it replicates (same sign, p < 0.05)
on CONFIRMATION seasons 2021-2025.
"""
import matplotlib.pyplot as plt
import nflreadpy as nfl
import numpy as np
import polars as pl
from scipy.stats import spearmanr

OUT = "rb_injury_output"
DISCOVERY = list(range(2013, 2021))
df = pl.read_parquet(f"{OUT}/rb_games_severity.parquet")
SEASONS = sorted(df["season"].unique().to_list())

# ---------- full per-game workload series (all games, not only modeled rows) ----------
pbp = nfl.load_pbp(SEASONS).filter(pl.col("season_type") == "REG").with_columns(
    pl.col("season", "week").cast(pl.Int32))
tackler_cols = ["solo_tackle_1_player_id", "solo_tackle_2_player_id",
                "assist_tackle_1_player_id", "assist_tackle_2_player_id",
                "assist_tackle_3_player_id", "assist_tackle_4_player_id",
                "tackle_with_assist_1_player_id", "tackle_with_assist_2_player_id"]
pbp = pbp.with_columns(n_tacklers=pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in tackler_cols]))
touch = pl.concat([
    pbp.filter((pl.col("rush_attempt") == 1) & pl.col("rusher_player_id").is_not_null()).select(
        "season", "week", player_id="rusher_player_id", c=pl.lit(1), t=pl.lit(1), h="n_tacklers"),
    pbp.filter((pl.col("complete_pass") == 1) & pl.col("receiver_player_id").is_not_null()).select(
        "season", "week", player_id="receiver_player_id", c=pl.lit(0), t=pl.lit(1), h="n_tacklers"),
])
series = touch.group_by("player_id", "season", "week").agg(
    g_carries=pl.sum("c").cast(pl.Float64), g_touches=pl.sum("t").cast(pl.Float64),
    g_hits=pl.sum("h").cast(pl.Float64))
snaps = nfl.load_snap_counts(SEASONS).filter(pl.col("game_type") == "REG").select(
    pl.col("season", "week").cast(pl.Int32), pfr_id="pfr_player_id", g_snaps=pl.col("offense_snaps").cast(pl.Float64))
ids = df.select("player_id", "pfr_id").unique("player_id")
series = (series.join(ids, on="player_id").join(snaps, on=["season", "week", "pfr_id"], how="left")
          .with_columns(pl.col("g_snaps").fill_null(0)).sort("player_id", "season", "week"))

ov = ["player_id", "season"]
win_cols = []
for v in ["carries", "hits", "touches", "snaps"]:
    g = f"g_{v}"
    for n in [1, 2, 3, 4, 6, 8]:
        name = f"{v}_l{n}"
        series = series.with_columns(pl.col(g).rolling_sum(n, min_samples=1).over(ov).alias(name))
        win_cols.append(name)
    series = series.with_columns(pl.col(g).cum_sum().over(ov).alias(f"{v}_season"))
    win_cols.append(f"{v}_season")
    # acute:chronic = this game vs mean of previous 4 games; 3-game acute vs previous 6
    chronic4 = pl.col(g).shift(1).rolling_mean(4, min_samples=2).over(ov)
    chronic6 = pl.col(g).shift(3).rolling_mean(6, min_samples=3).over(ov)
    series = series.with_columns(
        (pl.col(g) / chronic4.clip(1)).alias(f"acwr_{v}_1v4"),
        (pl.col(g).rolling_mean(3, min_samples=3).over(ov) / chronic6.clip(1)).alias(f"acwr_{v}_3v6"))
    win_cols += [f"acwr_{v}_1v4", f"acwr_{v}_3v6"]

# ---------- injuries: labels at several horizons, practice, history ----------
SOFT = "hamstring|groin|calf|quad|thigh|achilles|hip flexor|abdom|oblique|pectoral|back"
CONTACT = "knee|ankle|shoulder|concussion|rib|chest|hand|wrist|finger|thumb|elbow|neck|foot|toe"
inj = nfl.load_injuries(SEASONS).filter(pl.col("game_type") == "REG").select(
    pl.col("season", "week").cast(pl.Int32), player_id="gsis_id",
    status=pl.col("report_status"), practice=pl.col("practice_status").fill_null(""),
    body=pl.col("report_primary_injury").fill_null(pl.col("practice_primary_injury")).fill_null("").str.to_lowercase())
listed = inj.filter(pl.col("status").is_in(["Out", "Doubtful", "Questionable"])).with_columns(
    out=(pl.col("status") == "Out").cast(pl.Int32),
    soft=pl.col("body").str.contains(SOFT).cast(pl.Int32),
    contact=pl.col("body").str.contains(CONTACT).cast(pl.Int32),
).group_by("player_id", "season", "week").agg(pl.max("out"), pl.max("soft"), pl.max("contact"))

base = df.select("player_id", "season", "week")

def horizon_label(h, cond, name):
    lab = (base.join(listed.rename({"week": "lw"}), on=["player_id", "season"])
           .filter((pl.col("lw") > pl.col("week")) & (pl.col("lw") <= pl.col("week") + h)).filter(cond)
           .select("player_id", "season", "week").unique().with_columns(pl.lit(1).alias(name)))
    return lab

labels = [horizon_label(1, pl.lit(True), "any_1wk"), horizon_label(2, pl.lit(True), "any_2wk"),
          horizon_label(4, pl.lit(True), "any_4wk"), horizon_label(1, pl.col("out") == 1, "out_1wk"),
          horizon_label(4, pl.col("soft") == 1, "soft_4wk"), horizon_label(4, pl.col("contact") == 1, "contact_4wk")]
for lab in labels:
    df = df.join(lab, on=["player_id", "season", "week"], how="left")
LABELS = ["any_1wk", "any_2wk", "any_4wk", "out_1wk", "soft_4wk", "contact_4wk", "major"]
df = df.with_columns(pl.col(l).fill_null(0) for l in LABELS[:-1])

# practice participation in the week leading INTO this game (player not given a game status)
prac = inj.with_columns(
    practice_level=pl.when(pl.col("practice").str.contains("Did Not")).then(2)
    .when(pl.col("practice").str.contains("Limited")).then(1).otherwise(0)
).group_by("player_id", "season", "week").agg(pl.max("practice_level"))
# prior soft-tissue / contact listings (career, before this week)
hist = listed.sort("player_id", "season", "week").with_columns(
    prior_soft=pl.col("soft").cum_sum().over("player_id") - pl.col("soft"),
    prior_contact=pl.col("contact").cum_sum().over("player_id") - pl.col("contact"))
hk = hist.with_columns(key=pl.col("season") * 100 + pl.col("week")).select("player_id", "key", "prior_soft", "prior_contact",
                                                                          s=pl.col("soft"), c=pl.col("contact"))
df = df.with_columns(key=pl.col("season") * 100 + pl.col("week")).sort("key")
df = df.join_asof(hk.with_columns(ps=pl.col("prior_soft") + pl.col("s"), pc=pl.col("prior_contact") + pl.col("c"))
                  .select("player_id", "key", "ps", "pc").sort("key"),
                  on="key", by="player_id", strategy="backward", allow_exact_matches=False)
df = (df.rename({"ps": "prior_soft_inj", "pc": "prior_contact_inj"})
        .with_columns(pl.col("prior_soft_inj", "prior_contact_inj").fill_null(0))
        .join(prac, on=["player_id", "season", "week"], how="left")
        .with_columns(pl.col("practice_level").fill_null(0)))

# last season's playoff workload
post = nfl.load_player_stats([SEASONS[0] - 1] + SEASONS).filter(pl.col("season_type") == "POST").group_by(
    "player_id", "season").agg(prev_playoff_carries=pl.col("carries").fill_null(0).sum().cast(pl.Float64),
                               prev_playoff_games=pl.len().cast(pl.Float64)).with_columns(
    season=(pl.col("season") + 1).cast(pl.Int32))
df = (df.join(series.select("player_id", "season", "week", *win_cols), on=["player_id", "season", "week"], how="left")
        .join(post, on=["player_id", "season"], how="left")
        .with_columns(pl.col("prev_playoff_carries", "prev_playoff_games").fill_null(0)))

NEW_VARS = ["practice_level", "prior_soft_inj", "prior_contact_inj", "prev_playoff_carries", "prev_playoff_games"]
FEATS = win_cols + NEW_VARS
FILTERS = {"all (3+ carries)": 3, "regular (10+)": 10, "lead back (15+)": 15}

# ---------- grid: discovery -> confirmation ----------
def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)

def rho(d, f, lab):
    x, y = d[f].cast(pl.Float64).to_numpy(), d[lab].to_numpy()
    m = ~np.isnan(x)
    if m.sum() < 50 or y[m].sum() < 10 or np.nanstd(x[m]) == 0:
        return np.nan, np.nan
    return spearmanr(x[m], y[m])

rows = []
for fname, mc in FILTERS.items():
    d = df.filter(pl.col("carries") >= mc)
    disc, conf = d.filter(pl.col("season").is_in(DISCOVERY)), d.filter(~pl.col("season").is_in(DISCOVERY))
    for lab in LABELS:
        for f in FEATS:
            r1, p1 = rho(disc, f, lab)
            r2, p2 = rho(conf, f, lab)
            rows.append((fname, lab, f, disc.height, disc[lab].mean(), r1, p1, r2, p2))
grid = pl.DataFrame(rows, schema=["filter", "label", "feature", "n_disc", "rate_disc", "rho_disc", "p_disc",
                                  "rho_conf", "p_conf"], orient="row").drop_nans(["p_disc"])
grid = grid.with_columns(q_disc=pl.Series(bh(grid["p_disc"])))
grid = grid.with_columns(replicated=((pl.col("q_disc") < 0.10) & (pl.col("p_conf") < 0.05)
                                     & (np.sign(pl.col("rho_disc")) == np.sign(pl.col("rho_conf")))))
hits = grid.filter(pl.col("q_disc") < 0.10).sort("p_disc")
repl = grid.filter(pl.col("replicated")).sort("p_disc")

# ---------- ACWR danger-zone table ----------
def acwr_table(col, lab):
    return (df.filter(pl.col(col).is_not_null()).with_columns(
        zone=pl.col(col).cut([0.8, 1.3, 1.5], labels=["<0.8 (under)", "0.8-1.3 (sweet spot)", "1.3-1.5", ">1.5 (spike)"]))
        .group_by("zone").agg(n=pl.len(), rate=pl.mean(lab)).sort("zone"))

def lvl_table(col, lab_list):
    return df.group_by(col).agg(n=pl.len(), *[pl.mean(l).alias(l) for l in lab_list]).sort(col)

# window-length sweep: rho by window, per label (all filter, full sample)
sweep = []
for v in ["carries", "hits", "touches", "snaps"]:
    for n in [1, 2, 3, 4, 6, 8]:
        f = f"{v}_l{n}"
        sweep.append((v, n, *[rho(df, f, lab)[0] for lab in LABELS]))
sweep = pl.DataFrame(sweep, schema=["var", "window", *LABELS], orient="row")

# ---------- report ----------
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_cols(14); pl.Config.set_tbl_width_chars(180)
fmt = lambda d: str(d.with_columns(pl.selectors.float().round(4)))
lines = [
    f"RB-games {df.height}; discovery {DISCOVERY[0]}-{DISCOVERY[-1]}, confirmation {DISCOVERY[-1] + 1}-{SEASONS[-1]}",
    "Label base rates (all rows): " + ", ".join(f"{l}={df[l].mean():.3f}" for l in LABELS),
    f"Tests run: {grid.height} (filters x labels x features)",
    "",
    f"Discovery hits (BH q < 0.10): {hits.height}.  Replicated in confirmation (same sign, p < 0.05): {repl.height}",
    fmt(hits.select("filter", "label", "feature", "n_disc", "rho_disc", "q_disc", "rho_conf", "p_conf", "replicated")),
    "",
    "Window-length sweep (Spearman, all games, full sample):", fmt(sweep),
    "",
    "ACWR (carries, this game / previous 4) zones:",
    *[f"  {lab}:\n" + fmt(acwr_table("acwr_carries_1v4", lab)) for lab in ["any_1wk", "soft_4wk", "major"]],
    "ACWR (hits, 3-game vs previous 6) zones:",
    *[f"  {lab}:\n" + fmt(acwr_table("acwr_hits_3v6", lab)) for lab in ["any_1wk", "soft_4wk", "major"]],
    "",
    "Practice participation this week (0 full/none, 1 limited, 2 DNP) - played anyway:",
    fmt(lvl_table("practice_level", LABELS)),
    "Prior soft-tissue listings (career):",
    fmt(df.with_columns(b=pl.col("prior_soft_inj").clip(0, 3)).group_by("b").agg(
        n=pl.len(), soft_4wk=pl.mean("soft_4wk"), any_4wk=pl.mean("any_4wk"), major=pl.mean("major")).sort("b")),
    "Last season's playoff games:",
    fmt(df.with_columns(b=pl.col("prev_playoff_games").clip(0, 3)).group_by("b").agg(
        n=pl.len(), any_1wk=pl.mean("any_1wk"), any_4wk=pl.mean("any_4wk"), major=pl.mean("major")).sort("b")),
]
report = "\n".join(lines)
print(report)
open(f"{OUT}/sensitivity.txt", "w").write(report)

# ---------- figure ----------
fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
for v, c in [("carries", "#4C72B0"), ("hits", "#DD8452"), ("touches", "#55A868"), ("snaps", "#8172B3")]:
    s = sweep.filter(pl.col("var") == v)
    ax[0].plot(s["window"], s["any_4wk"], marker="o", color=c, label=v)
ax[0].axhline(0, color="k", lw=0.8); ax[0].set_xlabel("window (games)"); ax[0].legend()
ax[0].set_title("Correlation with injury in next 4 weeks\nby workload window length")
heat = grid.filter(pl.col("filter") == "all (3+ carries)").pivot(on="label", index="feature", values="rho_disc")
keep = [f for f in ["carries_l1", "carries_l3", "carries_l8", "carries_season", "hits_l1", "hits_l3", "hits_l8",
                    "hits_season", "snaps_l3", "acwr_carries_1v4", "acwr_hits_3v6", *NEW_VARS] if f in heat["feature"]]
heat = heat.filter(pl.col("feature").is_in(keep))
im = ax[1].imshow(heat.select(LABELS).to_numpy(), cmap="RdBu_r", vmin=-0.08, vmax=0.08, aspect="auto")
ax[1].set_xticks(range(len(LABELS)), LABELS, rotation=40); ax[1].set_yticks(range(heat.height), heat["feature"])
ax[1].set_title("Spearman rho, discovery seasons (all RBs)"); plt.colorbar(im, ax=ax[1])
t = lvl_table("practice_level", LABELS)
for i, lab in enumerate(["any_1wk", "any_4wk", "soft_4wk", "major"]):
    ax[2].bar(np.arange(t.height) + i * 0.2, t[lab], width=0.2, label=lab)
ax[2].set_xticks(np.arange(t.height) + 0.3, ["full / not listed", "limited", "DNP"])
ax[2].set_title("Practice participation the week he still played"); ax[2].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/sensitivity.png", dpi=130)
