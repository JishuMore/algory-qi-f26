# Running Back Injuries: Does Workload Predict Them?

**Data:** nflreadpy (nflverse): play-by-play, injury reports, weekly rosters, snap counts, Next Gen Stats, PFR advanced stats, contracts, combine.
**Scope:** NFL regular seasons 2013–2025 (2016–2024 in the first rounds). Unit = one RB game (3+ carries) or one RB season (50+ carries).
**Method:** XGBoost and logistic regression, always scored on seasons the model did not train on (holdout or walk-forward), plus correlation scans corrected for multiple testing.

## Bottom line

1. **Workload is a weak predictor of RB injuries.** The best models reach an AUC of about 0.58–0.63 for "any new injury next week" (0.50 = coin flip). That's a real but modest edge.
2. **Injury history is the most reliable signal.** It beats carries, yards, distance run, and hits taken.
3. **No workload threshold makes injuries more likely.** Risk rises as a back goes from backup to regular, then flattens or falls for the heaviest workloads.
4. **Nothing in public data predicts serious injuries** (3+ games missed, IR, or season over). Those models score about 0.50–0.53 AUC.
5. **Workload adds little on top of injury history.** It works at most as an amplifier: a previously injured back on a heavy recent load is the highest-risk group, about 1.5× the lowest.

## What is actually worth looking at

Ranked by how well each finding holds up. "Replicated" means found in 2013–2020 and confirmed in 2021–2025, or significant when predicting seasons the model never trained on.

### Worth pursuing (held up)

| Signal | Evidence | Size of effect | Why it matters |
|---|---|---|---|
| **1. Injury history, especially to the same body part** | Replicated in both eras, at both game and season level, and after controls | Prior contact injury → contact injury within 4 wks: ρ ≈ +0.10 in both eras. 2–3 weeks on the report already this season: 12.2% vs 6.8% injury next week. Last season's games missed: OR 1.29 per +2.3 games. | The single most consistent predictor across 9 rounds. |
| **2. The window right after return** | Clear, monotonic pattern in 5,171 games; spline p = 0.04 | Same-part re-injury 7.5% in the first 250 yds back vs 3.1% at 1,500–2,500 yds. Knee 10.4% and ankle 8.4% in the first 500 yds. | Actionable: it says *when* a back is most vulnerable (the first 1–3 games back), which is more useful than *how much* he runs. |
| **3. History × recent workload (risk tiers)** | Consistent direction; interaction not significant | Injury history + top-third recent yards: 28.9% injury within 4 wks vs 16.6–18.8% for clean backs. | The most practical screening rule found, though workload acts as an amplifier, not a cause. |

### Possibly worth a second look (suggestive, not proven)

- **Receptions and last season's playoff carries → next season's injuries** (OR 1.26 and 1.24, CIs exclude 1). Each is a single test and hasn't been replicated in a second era.
- **Team pace, team RB injury history, Denver altitude** (q ≈ 0.07, small samples). A team-level (medical / strength-staff) effect would be interesting if it held up with more data.
- **Calf / hamstring / foot recurrence season to season** (risk ratios 1.8–4.9×). Only calf is significant, and the samples are small.

### Not worth more time (tested repeatedly, nothing there)

- Carries, yards, hits or NGS distance **in a single game**, season to date, or over a career, used to predict injury.
- **A distance or carry threshold** for injury or re-injury. None exists; risk plateaus or falls.
- **The acute:chronic workload ratio (ACWR)**, the "curse of 370", the age-29 cliff, yards after contact, broken tackles, tackler size, turf, weather, travel, short weeks, contract year.
- **Predicting serious (Major) injuries** from any public variable: AUC about 0.50–0.53 in every round.

### If you take one thing forward

Build a **re-injury watch-list**, not a workload model: flag backs in their first ~500 yds (1–3 games) back from a knee or ankle injury, and backs with 2+ prior listings this season who are on a heavy recent load. These are the only groups whose injury rates are clearly above baseline (roughly 1.5–2× the base rate). Better data could make this predictive: tracking-based load, pass-blocking reps, and injury detail such as severity grade, surgery, or which leg.

## What we tested, and what happened

| # | Question | Script | Result |
|---|---|---|---|
| 1 | Do carries and hits in a game predict next-week injury? | `rb_injury.py` | Single-game carries and hits show **no effect** (AUC 0.50). Contact built up over several games shows a little. |
| 2 | Wider scan: surface, weather, snaps, game script, injury history | `rb_injury_corr.py` | Injury history and accumulated hits are significant. Turf, weather, short weeks and running style are not. Model AUC ≈ 0.63. |
| 3 | Injury severity index (Minor / Moderate / Major, 0–10 time lost) | `rb_injury_severity.py` | Major injuries ≈ 6–7% of games. Knee (41% Major) and foot (38%) are the worst injuries. Workload does **not** predict severity. |
| 4 | Running distance (NGS tracking) and thresholds | `rb_injury_distance.py` | **No breakpoint where risk accelerates.** Risk plateaus after about 13 carries or 75 yards into a season. NGS distance adds nothing. |
| 5 | New variables: tacklers, opponent, schedule, team history, contract | `rb_injury_newvars.py` | 27 variables, none significant after correction. Combined, they add ≈ 0 AUC. Weak leads: team pace, team RB injury history, Denver altitude. |
| 6 | Sensitivity grid: 857 definitions, found on 2013–20, checked on 2021–25 | `rb_injury_sensitivity.py` | Replicated: **prior contact injuries → contact injury within 4 wks** and **prior soft-tissue → soft-tissue**. The workload correlations that replicate reflect playing time, not fatigue. The acute:chronic workload ratio (ACWR, a sports-science spike metric) fails. |
| 7 | Season → next season, published theories | `rb_injury_seasons.py` | Last season's games missed to injury predict next season's (OR 1.29). Receptions and playoff carries add risk. Carries do not. |
| 8 | Findings + yardage/carries combined | `rb_injury_combined.py` | Adding yardage changes AUC by −0.01 to +0.03, every CI includes 0. History × workload interaction points the expected way but isn't significant. |

| 9 | Previous injury × distance run since return → re-injury (non-linear) | `rb_injury_reinjury.py` | **No distance that triggers re-injury.** Same-part re-injury risk is highest in the **first ~500 yds after returning** (7.5% within 4 wks) and falls to ~3% by 1,500–2,500 yds. Knee (10.4%) and ankle (8.4%) are most at risk early. Returning on a heavier-than-usual load shows no effect. Distance adds ≈ 0 AUC. |

## Re-injury and distance since return

Estimated distance = NGS tracked distance where available, otherwise carries × the player's yards run per carry (median 16.3).

| Distance run since returning | 0–250 | 250–500 | 500–1k | 1k–1.5k | 1.5k–2.5k | 2.5k–4k | 4k+ |
|---|---|---|---|---|---|---|---|
| Same-part re-injury within 4 wks | 7.5% | 7.2% | 6.4% | 4.7% | **3.1%** | 4.6% | 2.5–4.8% |

- The relationship is non-linear and *decreasing*: a vulnerable window right after return that fades with distance. A spline fit with controls for time since return, age and snap share is borderline (p = 0.04). The best breakpoint, about 2,000 yds, is where the *decline* levels off, and the bootstrap range is unstable (770–4,600 yds).
- Re-injury-free curve: about 95% of backs avoid re-injuring the same part through their first 500 yds back and about 85% through 9,000 yds. Most re-injuries happen early.
- For backs with no injury history, risk rises with career distance up to about 3,000 yds and then drops (non-linearity p = 0.009). That's the same backup-to-regular exposure effect and survivor bias seen earlier, not wear and tear.

## Key numbers

**Injury history (game level).** Rate of a new injury listing the next week:

| Weeks already on the report this season | 0 | 1 | 2–3 |
|---|---|---|---|
| Injury rate next week | 6.8% | 10.0% | 12.2% |

**Recent workload × injury history.** Injury within the next 4 weeks:

| Yards over previous 3 games | Low | Mid | High |
|---|---|---|---|
| No prior injury listings | 18.8% | 16.6% | 21.8% |
| Prior injury listings | 21.8% | 25.3% | **28.9%** |

**Season to next season.** Share who missed 1+ games to injury the next season:

| Carries this season | 50–149 | 150–249 | 250–299 | 300+ |
|---|---|---|---|---|
| Missed 1+ games next season | 44.9% | 53.5% | 48.5% | 30.4% |

## Published theories checked

| Theory | Verdict |
|---|---|
| **"Curse of 370"** (Football Outsiders): 370+ carries leads to injury or decline the next year | Untestable. Only 2 such seasons since 2013. All 6 seasons of 330+ carries were followed by a drop in yards per carry, which is consistent with regression to the mean. |
| **Heavy workload is protective** (*Orthopaedic J. Sports Med.*) | Consistent: 300+ carry backs missed time 30% of the time vs 54% for 150–249 (p = 0.048). Probably survivor bias, since only durable backs get 300 carries. |
| **Same-body-part re-injury** (Fulton et al.) | Supported in direction: calf 4.9× (p = 0.03); knee 1.5×, hamstring 1.8×, foot 2.3× (not significant). |
| **Age cliff at 29** | Not supported. Missing 5+ games barely rises with age (16% → 18%) once survivors are accounted for. |
| **Yards after contact raises injury risk** (Keefer & Kniesner) | Not supported, either by season or by game (ρ ≈ 0.01). |
| **ACWR spike > 1.5** | Not supported for RBs. The ratio runs backwards because an in-game injury shortens that game's workload. |

## Pitfalls we found and corrected (worth knowing for any follow-up)

- **Bye weeks:** there is no injury report the week a team is on bye, so the label was 0 by construction. These games are now excluded (fixed from `rb_injury_corr.py` on; the original `rb_injury.py` still has this issue).
- **In-game injury leakage:** a back hurt during a game stops carrying, so low same-game or 4th-quarter volume reflects an injury that has *already happened*. Pre-game models drop these variables.
- **End-of-season censoring:** a late-season injury can't produce many missed games. The severity analysis requires 4+ games left in the season.
- **IR moves without an injury report listing** (about 40% of Major injuries) were missing from the early label. They are now included.
- **Survivor bias:** heavily used backs and older backs are the durable ones. This flattens or reverses most "more work = more injury" curves.
- **Multiple testing:** hundreds of comparisons were run. Results are Benjamini–Hochberg corrected, and the sensitivity grid uses a discovery/confirmation split.

## Limitations

- **Injury reports are noisy and strategic.** "Questionable" designations in particular can be gamesmanship.
- **Contact is incomplete.** "Hits" counts only tacklers credited in the play-by-play. Pass-blocking contact is invisible.
- **NGS distance covers only games with 10+ carries.** Real player-tracking data (Big Data Bowl) is not in nflreadpy.
- **Season-level samples are small:** 773 RB-seasons, only 23 with 300+ carries.

## Suggested next steps

1. Player-tracking data (speed, acceleration, collision intensity) from the Big Data Bowl.
2. Pass-blocking snaps (PFF) as a hidden contact load.
3. Finer injury history: same leg, ACL vs sprain, offseason surgery.
4. FTN charting (2022+) for play-level contact detail.

## Reproducing

Run in this order (each step caches a parquet the next ones use):
`rb_injury_corr.py` → `rb_injury_severity.py` → `rb_injury_distance.py`, `rb_injury_newvars.py`, `rb_injury_sensitivity.py`, `rb_injury_seasons.py` → `rb_injury_combined.py`, `rb_injury_reinjury.py`.
Full outputs: the `*.txt` and `*.png` files in `rb_injury_output/`.
