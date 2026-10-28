"""Re-test three injury findings for running backs with nflreadpy + XGBoost.

H1  Injury history predicts new injury (weeks already on the report this season;
    prior contact injury -> another contact injury within 4 weeks).
H2  Same-part re-injury risk is highest in the first few hundred scrimmage yards
    after a back returns from an injury.
H3  History x heavy recent workload is the best simple screen (injury within 4 weeks).
Then: XGBoost with / without history features, trained 2013-2020, tested 2021-2025.

Unit: one row per RB per regular-season game played.
"Listed" = Out / Doubtful / Questionable on that week's official injury report.
"New injury" next week = listed next week and not listed this week (or listed for a
different body part).
"Contact" is a body-part proxy (the report doesn't say how an injury happened).
"""
import os

import nflreadpy as nfl
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

SEASONS = list(range(2013, 2026))
TRAIN_END = 2020
OUT = "rb_injury_output"
os.makedirs(OUT, exist_ok=True)

CONTACT = {"knee", "ankle", "shoulder", "concussion", "ribs", "chest", "hand", "wrist",
           "thumb", "finger", "elbow", "neck", "foot", "toe", "head", "collarbone"}

# ---------- injury reports ----------
inj = (nfl.load_injuries(SEASONS)
       .filter((pl.col("game_type") == "REG")
               & pl.col("report_status").is_in(["Out", "Doubtful", "Questionable"]))
       .with_columns(season=pl.col("season").cast(pl.Int32), week=pl.col("week").cast(pl.Int32),
                     part=pl.coalesce("report_primary_injury", "practice_primary_injury")
                     .str.to_lowercase().str.split(",").list.first().str.strip_chars()
                     .str.replace(r"^(right|left) ", "").str.replace(r"s$", "")
                     .fill_null("unknown"))
       .group_by("gsis_id", "season", "week").agg(pl.first("part"))
       .rename({"gsis_id": "player_id"}))

# ---------- weekly RB workload ----------
st = (nfl.load_player_stats(SEASONS)
      .filter((pl.col("season_type") == "REG") & (pl.col("position") == "RB"))
      .select("player_id", "season", "week",
              touches=pl.col("carries").fill_null(0) + pl.col("receptions").fill_null(0),
              yards=pl.col("rushing_yards").fill_null(0) + pl.col("receiving_yards").fill_null(0))
      .filter(pl.col("touches") > 0))

# full player-week grid (game weeks + injury weeks) so history/returns are tracked
# through weeks a back didn't play
grid = (pl.concat([st.select("player_id", "season", "week"), inj.select("player_id", "season", "week")])
        .unique()
        .filter(pl.col("player_id").is_in(st["player_id"].unique().implode())))
g = (grid.join(st, on=["player_id", "season", "week"], how="left")
     .join(inj, on=["player_id", "season", "week"], how="left")
     .with_columns(played=pl.col("touches").is_not_null(),
                   touches=pl.col("touches").fill_null(0), yards=pl.col("yards").fill_null(0),
                   listed=pl.col("part").is_not_null())
     .sort("player_id", "season", "week"))

# walk each player's history in python (clear stateful logic: return clocks etc.)
rows = []
for (pid,), d in g.group_by(["player_id"], maintain_order=True):
    d = d.to_dicts()
    career_parts = {}          # part -> times listed (prior)
    contact_wks = []           # (season, week) of contact listings
    cur_season, wks_listed_season = None, 0
    since_return = {}          # part -> scrimmage yards since that part came off the report
    for i, r in enumerate(d):
        if r["season"] != cur_season:
            cur_season, wks_listed_season = r["season"], 0
        nxt = d[i + 1] if i + 1 < len(d) and d[i + 1]["season"] == r["season"] \
            and d[i + 1]["week"] == r["week"] + 1 else None
        fut = [x for x in d[i + 1:i + 5] if x["season"] == r["season"] and x["week"] <= r["week"] + 4]
        if r["played"]:
            new_next = nxt is not None and nxt["listed"] and (not r["listed"] or nxt["part"] != r["part"])
            new_4 = [x for x in fut if x["listed"] and x["part"] != r["part"]]
            last3 = [x["touches"] for x in d[max(0, i - 2):i + 1] if x["season"] == r["season"]]
            # most recent returned-from part and yards since return
            ret_part = min(since_return, key=since_return.get) if since_return and not r["listed"] else None
            rows.append(dict(
                player_id=pid, season=r["season"], week=r["week"], touches=r["touches"],
                touches_l3=np.mean(last3), listed_now=r["listed"],
                wks_listed_season=wks_listed_season,
                career_listings=sum(career_parts.values()),
                any_history=sum(career_parts.values()) > 0,
                prior_contact_4w=any((s == r["season"] and r["week"] - w <= 4) for s, w in contact_wks),
                ret_part=ret_part,
                yds_since_return=max(since_return.get(ret_part), 0) if ret_part else None,
                new_next=new_next,
                new_4w=len(new_4) > 0,
                new_contact_4w=any(x["part"] in CONTACT for x in new_4),
                same_part_next=bool(nxt and nxt["listed"] and not r["listed"] and nxt["part"] == ret_part),
                same_part_4w=bool(ret_part) and any(x["listed"] and x["part"] == ret_part for x in fut),
            ))
        # update state after the row is recorded (features use only the past)
        if r["listed"]:
            wks_listed_season += 1
            prev_listed = i > 0 and d[i - 1]["listed"] and d[i - 1]["part"] == r["part"]
            if not prev_listed:
                career_parts[r["part"]] = career_parts.get(r["part"], 0) + 1
                if r["part"] in CONTACT:
                    contact_wks.append((r["season"], r["week"]))
            since_return[r["part"]] = 0  # clock restarts when he comes off the report
        else:
            # yards this game count toward every open return clock; close clocks past 2,500
            since_return = {p: y + r["yards"] for p, y in since_return.items() if y + r["yards"] <= 2500}

df = pl.DataFrame(rows)
df.write_parquet(f"{OUT}/rb_history_panel.parquet")

out = []
def say(s=""):
    print(s); out.append(s)

def rate_tbl(frame, by, y):
    return frame.group_by(by).agg(n=pl.len(), rate=pl.mean(y)).sort(by)

def logit(formula, data):
    m = smf.logit(formula, data=data.to_pandas()).fit(disp=0, cov_type="cluster",
                                                      cov_kwds={"groups": data["player_id"].to_pandas().astype("category").cat.codes})
    return m

say(f"RB-games: {df.height}   seasons {SEASONS[0]}-{SEASONS[-1]}")
say(f"Base rate, new listing next week: {df['new_next'].mean():.2%}   within 4 wks: {df['new_4w'].mean():.2%}")

# ---------- H1 ----------
say("\n=== H1: weeks already on report this season -> new injury next week ===")
h1 = df.with_columns(b=pl.col("wks_listed_season").cut([0, 1, 3], labels=["0", "1", "2-3", "4+"]))
for era, f in [("all", h1), ("2013-20", h1.filter(pl.col("season") <= 2020)), ("2021-25", h1.filter(pl.col("season") > 2020))]:
    t = rate_tbl(f, "b", "new_next")
    say(f"{era:8s} " + "  ".join(f"{r['b']}: {r['rate']:.1%} (n={r['n']})" for r in t.iter_rows(named=True)))
m = logit("new_next ~ wks_listed_season + touches_l3 + C(season)", df.with_columns(pl.col("new_next").cast(int)))
say(f"logit (controls: recent touches, season FE; player-clustered SE): "
    f"wks_listed OR={np.exp(m.params['wks_listed_season']):.3f}  p={m.pvalues['wks_listed_season']:.2g}")

say("\n--- prior contact injury (<=4 wks ago) -> new contact injury within 4 wks ---")
t = rate_tbl(df, "prior_contact_4w", "new_contact_4w")
say("  ".join(f"prior={r['prior_contact_4w']}: {r['rate']:.1%} (n={r['n']})" for r in t.iter_rows(named=True)))
m = logit("new_contact_4w ~ prior_contact_4w + touches_l3 + C(season)",
          df.with_columns(pl.col("new_contact_4w").cast(int), pl.col("prior_contact_4w").cast(int)))
say(f"logit OR={np.exp(m.params['prior_contact_4w']):.2f}  p={m.pvalues['prior_contact_4w']:.2g}")

# ---------- H2 ----------
say("\n=== H2: same-part re-injury next week by scrimmage yards since return ===")
h2 = df.filter(pl.col("yds_since_return").is_not_null() & ~pl.col("listed_now")
               & ~pl.col("ret_part").is_in(["illnes", "unknown", "not injury related"])).with_columns(
    b=pl.col("yds_since_return").cut([250, 500, 1000, 1500], labels=["0-250", "250-500", "500-1000", "1000-1500", "1500-2500"]))
for r in h2.group_by("b").agg(n=pl.len(), nxt=pl.mean("same_part_next"), w4=pl.mean("same_part_4w")).sort("b").iter_rows(named=True):
    say(f"  {r['b']:10s} next wk {r['nxt']:.1%}   within 4 wks {r['w4']:.1%}  (n={r['n']})")
say("  first 250 yds by part (same-part within 4 wks):")
for r in (h2.filter(pl.col("yds_since_return") < 250).group_by("ret_part")
          .agg(n=pl.len(), rate=pl.mean("same_part_4w")).filter(pl.col("n") >= 60)
          .sort("rate", descending=True).iter_rows(named=True)):
    say(f"    {r['ret_part']:12s} {r['rate']:.1%} (n={r['n']})")
m = logit("same_part_4w ~ np.log1p(yds_since_return) + touches_l3 + C(season)",
          h2.with_columns(pl.col("same_part_4w").cast(int)))
say(f"logit log(1+yds since return) OR={np.exp(m.params['np.log1p(yds_since_return)']):.3f}  "
    f"p={m.pvalues['np.log1p(yds_since_return)']:.2g}")

# ---------- H3 ----------
say("\n=== H3: history x heavy recent workload -> new injury within 4 wks ===")
q75 = df["touches_l3"].quantile(0.75)
h3 = df.with_columns(heavy=pl.col("touches_l3") >= q75)
say(f"heavy = avg touches last 3 games >= {q75:.1f} (top quartile)")
for r in rate_tbl(h3, ["any_history", "heavy"], "new_4w").iter_rows(named=True):
    say(f"  history={r['any_history']!s:5s} heavy={r['heavy']!s:5s}  {r['rate']:.1%} (n={r['n']})")
m = logit("new_4w ~ any_history * heavy + C(season)",
          h3.with_columns(*(pl.col(c).cast(int) for c in ["new_4w", "any_history", "heavy"])))
for k in ["any_history", "heavy", "any_history:heavy"]:
    say(f"  {k:18s} OR={np.exp(m.params[k]):.2f}  p={m.pvalues[k]:.2g}")

# ---------- XGBoost ----------
say("\n=== XGBoost: new listing next week, train <=2020, test 2021-25 ===")
X_all = ["touches", "touches_l3", "listed_now", "wks_listed_season", "career_listings",
         "prior_contact_4w", "yds_since_return", "week"]
X_work = ["touches", "touches_l3", "week"]
tr, te = df.filter(pl.col("season") <= TRAIN_END), df.filter(pl.col("season") > TRAIN_END)
def fit(cols, y="new_next"):
    m = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=30, eval_metric="logloss")
    Xtr = tr.select(cols).cast(pl.Float64).to_numpy(); Xte = te.select(cols).cast(pl.Float64).to_numpy()
    m.fit(Xtr, tr[y].to_numpy())
    p = m.predict_proba(Xte)[:, 1]
    return m, p, Xte
for name, cols in [("workload only", X_work), ("history only", [c for c in X_all if c not in X_work] + ["week"]), ("all", X_all)]:
    for y in ["new_next", "new_4w"]:
        m, p, Xte = fit(cols, y)
        yt = te[y].to_numpy()
        # bootstrap CI on AUC
        rng = np.random.default_rng(0); aucs = []
        for _ in range(300):
            idx = rng.integers(0, len(yt), len(yt)); aucs.append(roc_auc_score(yt[idx], p[idx]))
        say(f"  {name:14s} {y:8s} AUC {roc_auc_score(yt, p):.3f} [{np.percentile(aucs, 2.5):.3f}, {np.percentile(aucs, 97.5):.3f}]"
            f"  PR-AUC {average_precision_score(yt, p):.3f} (base {yt.mean():.3f})")
m, p, Xte = fit(X_all)
contrib = m.get_booster().predict(xgb.DMatrix(Xte), pred_contribs=True)[:, :-1]
say("  mean |SHAP| (all-features, next week):")
for f, v in sorted(zip(X_all, np.abs(contrib).mean(0)), key=lambda t: -t[1]):
    say(f"    {f:20s} {v:.3f}")
dec = te.with_columns(p=p, dec=pl.Series(p).qcut(10, labels=[str(i) for i in range(1, 11)], allow_duplicates=True))
say("  test injury rate by predicted decile: " + " ".join(
    f"{r['rate']:.1%}" for r in rate_tbl(dec, "dec", "new_next").iter_rows(named=True)))

with open(f"{OUT}/history_tests_results.txt", "w") as fh:
    fh.write("\n".join(out) + "\n")
