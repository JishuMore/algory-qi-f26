"""
Ascend Algo - Round 1 trader (CARROT, POTATO, BROCCOLI, VEGGIE_BASKET, SUNLIGHT_INDEX).

Self-contained: only the standard library + `datamodel`. All rolling state (EWMAs, short
price histories, previous book) lives in `traderData` as JSON and is updated O(1) per tick.

The feature engine (`update_features`) is the single source of truth: `fit_ridge.py`
replays the historical CSVs through this exact function, fits Ridge per product/horizon,
and rewrites the FITTED COEFFICIENTS block below. Do not hand-edit features in one place
only - train/live parity depends on both sides calling the same code.

Until the block is populated, the trader falls back to fair = Deep-MM mid
(mid + outer_dev), the strongest single signal found in the EDA.
"""
import json
import math

try:  # The fit script imports this module offline, where datamodel may be absent.
    from datamodel import Order, TradingState  # noqa: F401
except ImportError:  # pragma: no cover
    Order = None
    TradingState = None


PRODUCTS = ["CARROT", "POTATO", "BROCCOLI", "VEGGIE_BASKET", "SUNLIGHT_INDEX"]
WEATHER_NUM = ["sunshine", "rainfall", "temperature", "humidity", "wind_speed", "soil_moisture"]

# ----------------------------------------------------------------------------------------
# Execution parameters - UNTUNED defaults. Set LIMITS from the round rules and tune the
# rest in the backtester. Exceeding a position limit makes the exchange reject all orders.
# ----------------------------------------------------------------------------------------
LIMITS = {"CARROT": 50, "POTATO": 50, "BROCCOLI": 50, "VEGGIE_BASKET": 50, "SUNLIGHT_INDEX": 50}
TAKE_EDGE = {"CARROT": 1.0, "POTATO": 1.0, "BROCCOLI": 1.0, "VEGGIE_BASKET": 2.0, "SUNLIGHT_INDEX": 1.5}
MAKE_EDGE = {"CARROT": 1.0, "POTATO": 1.0, "BROCCOLI": 1.0, "VEGGIE_BASKET": 2.0, "SUNLIGHT_INDEX": 1.5}
GAMMA = {"CARROT": 0.05, "POTATO": 0.05, "BROCCOLI": 0.05, "VEGGIE_BASKET": 0.08, "SUNLIGHT_INDEX": 0.05}
MAKE_SIZE = {"CARROT": 10, "POTATO": 10, "BROCCOLI": 10, "VEGGIE_BASKET": 8, "SUNLIGHT_INDEX": 10}
HORIZON_WEIGHTS = {"h10": 0.5, "h50": 0.5}  # fair = mid + sum_h w_h * predicted move at h

# ----------------------------------------------------------------------------------------
# Feature sets per product (collinearity-pruned). BASKET_SPREAD_adj was dropped: it is an
# exact linear combination of BASKET_SPREAD, is_rain, is_storm and soil_moisture_ema20,
# which are all already in the VEGGIE_BASKET set.
# ----------------------------------------------------------------------------------------
_REGIME = ["is_rain", "is_storm", "is_sunny", "is_pest_high",
           "trans_cloudy_to_rain", "trans_rain_to_cloudy", "trans_pest_up", "trans_pest_down"]


def _base(p):
    return [f"{p}_outer_dev", f"{p}_obi1_ema10", f"{p}_mm_obi_ema50", f"{p}_wall_imb", f"{p}_inv_imb",
            f"{p}_past_1", f"{p}_past_5", f"{p}_past_10",
            f"{p}_t_sign_lag1", f"{p}_t_L2_lag1", f"{p}_t_small_ema10", f"{p}_t_sqty_ema50"] + _REGIME


FEATURES = {
    "CARROT": _base("CARROT") + ["BASKET_SPREAD", "BASKET_SPREAD_past_10", "CP_SPREAD", "CP_SPREAD_past_10",
                                 "temperature_ema20", "soil_moisture_ema20", "wind_speed_ema20"],
    "POTATO": _base("POTATO") + ["BASKET_SPREAD", "BASKET_SPREAD_past_10", "CP_SPREAD", "CP_SPREAD_past_10",
                                 "CARROT_past_10", "CARROT_mm_obi_ema50", "POTATO_t_med_ema10",
                                 "temperature_ema20", "soil_moisture_ema20", "soil_moisture_d50",
                                 "wind_speed_ema20"],
    "BROCCOLI": _base("BROCCOLI") + ["BASKET_SPREAD", "BASKET_SPREAD_past_10", "BROCCOLI_t_large_ema10",
                                     "temperature_ema20", "soil_moisture_ema20", "wind_speed_ema20"],
    "VEGGIE_BASKET": _base("VEGGIE_BASKET") + ["BASKET_SPREAD", "BASKET_SPREAD_past_10", "CP_SPREAD_past_10",
                                               "SYNTH_mm_obi_ema50", "CARROT_outer_dev", "POTATO_outer_dev",
                                               "BROCCOLI_outer_dev", "VEGGIE_BASKET_t_large_ema10",
                                               "temperature_ema20", "soil_moisture_ema20", "soil_moisture_d50",
                                               "sunshine_ema20"],
    "SUNLIGHT_INDEX": _base("SUNLIGHT_INDEX") + ["SUNLIGHT_INDEX_t_contrarian_ema10",
                                                 "sunshine_ema20", "temperature_ema20", "wind_speed_ema20"],
}

# === BEGIN FITTED COEFFICIENTS (written by fit_ridge.py; raw feature space) ===
COEFS = {}
# === END FITTED COEFFICIENTS ===


# ----------------------------------------------------------------------------------------
# Incremental EWMA identical to pandas `.ewm(span=s, adjust=True).mean()`:
#   num_t = x_t + (1-a) num_{t-1},  den_t = 1 + (1-a) den_{t-1},  ema_t = num_t / den_t
# ----------------------------------------------------------------------------------------
def _ewm(ew, key, x, span):
    a = 2.0 / (span + 1.0)
    num, den = ew.get(key, (0.0, 0.0))
    num = x + (1.0 - a) * num
    den = 1.0 + (1.0 - a) * den
    ew[key] = (num, den)
    return num / den


def _push(hist, x, maxlen):
    hist.append(x)
    if len(hist) > maxlen:
        del hist[0]


def _diff(hist, h):
    return hist[-1] - hist[-1 - h] if len(hist) > h else 0.0


def _sign(x):
    return (x > 0) - (x < 0)


def update_features(S, ts, books, trades, obs):
    """
    Advance state S by one tick and return the feature dict for this tick.

    S      : mutable dict (persisted as traderData).
    ts     : current timestamp.
    books  : {product: (bids, asks)}; bids = [(price, vol)] best-first, asks likewise, vol > 0.
    trades : {product: [(price, qty)]} market trades that printed during the PREVIOUS tick
             (what state.market_trades holds at `ts`). Signed against the previous tick's book.
    obs    : dict with 'weather', 'pest_alert' and WEATHER_NUM keys, or None if unavailable.
    """
    ew = S.setdefault("ew", {})
    mids = S.setdefault("mids", {})
    prev_book = S.get("book_prev", {})
    last = S.setdefault("last", {})  # last valid per-product features (used if a side is empty)
    f = {}

    # ---- Weather / regime block -------------------------------------------------------
    if obs is not None:
        w, pa = obs.get("weather"), obs.get("pest_alert")
        pw, pp = S.get("w_prev", w), S.get("p_prev", pa)
        S["w_prev"], S["p_prev"] = w, pa
        for k in WEATHER_NUM:
            v = obs.get(k)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                v = S.get("wlast_" + k, 0.0)
            S["wlast_" + k] = float(v)
            S["w_" + k + "_ema20"] = _ewm(ew, k + "_e20", float(v), 20)
        sm = S.setdefault("sm20_hist", [])
        _push(sm, S["w_soil_moisture_ema20"], 51)
        S["sm_d50"] = _diff(sm, 50)
        S["reg"] = [float(w == "rain"), float(w == "storm"), float(w == "sunny"), float(pa == "high")]
        S["trans"] = [
            _ewm(ew, "tr_c2r", float(pw == "cloudy" and w == "rain"), 10),
            _ewm(ew, "tr_r2c", float(pw == "rain" and w == "cloudy"), 10),
            _ewm(ew, "tr_pu", float((pp == "none" and pa == "low") or (pp == "low" and pa == "high")), 20),
            _ewm(ew, "tr_pd", float((pp == "high" and pa == "low") or (pp == "low" and pa == "none")), 20),
        ]
    reg = S.get("reg", [0.0, 0.0, 0.0, 0.0])
    trans = S.get("trans", [0.0, 0.0, 0.0, 0.0])
    f["is_rain"], f["is_storm"], f["is_sunny"], f["is_pest_high"] = reg
    f["trans_cloudy_to_rain"], f["trans_rain_to_cloudy"], f["trans_pest_up"], f["trans_pest_down"] = trans
    for k in ("sunshine", "temperature", "wind_speed", "soil_moisture"):
        f[k + "_ema20"] = S.get("w_" + k + "_ema20", 0.0)
    f["soil_moisture_d50"] = S.get("sm_d50", 0.0)

    # ---- Per-product book + lagged trade-flow block -------------------------------------
    new_prev = {}
    for p in PRODUCTS:
        bids, asks = books.get(p, ([], []))
        if not bids or not asks:
            f.update(last.get(p, {}))
            if p in prev_book:
                new_prev[p] = prev_book[p]
            continue
        (b1p, b1v), (a1p, a1v) = bids[0], asks[0]
        b2 = bids[1] if len(bids) > 1 else None
        a2 = asks[1] if len(asks) > 1 else None
        mid = 0.5 * (b1p + a1p)
        mmbp, mmbv = b2 if b2 else (b1p, b1v)
        mmap, mmav = a2 if a2 else (a1p, a1v)

        pf = {}
        pf[p + "_mid"] = mid
        pf[p + "_outer_dev"] = 0.5 * (mmbp + mmap) - mid
        pf[p + "_obi1_ema10"] = _ewm(ew, p + "_obi1", (b1v - a1v) / (b1v + a1v), 10)
        pf[p + "_mm_obi_ema50"] = _ewm(ew, p + "_mmobi", (mmbv - mmav) / (mmbv + mmav), 50)
        pf[p + "_wall_imb"] = float(b2 is None) - float(a2 is None)
        pf[p + "_inv_imb"] = float(b2 is not None and b1v > b2[1]) - float(a2 is not None and a1v > a2[1])

        h = mids.setdefault(p, [])
        _push(h, mid, 51)
        for k in (1, 5, 10, 50):
            pf[f"{p}_past_{k}"] = _diff(h, k)

        # Trades from the previous tick, signed vs the previous tick's book.
        net_sign = net_sqty = sign_l2 = small = med = large = contra = 0.0
        pb = prev_book.get(p)
        if pb is not None:
            pmid, pb2, pa2 = pb
            for price, qty in trades.get(p, ()):
                s = _sign(price - pmid)
                sq = s * qty
                net_sign += s
                net_sqty += sq
                if price == pb2 or price == pa2:
                    sign_l2 += s
                if qty <= 5:
                    small += sq
                elif qty <= 10:
                    med += sq
                else:
                    large += sq
                    if qty <= 12:
                        contra -= sq
        pf[p + "_t_sign_lag1"] = net_sign
        pf[p + "_t_L2_lag1"] = sign_l2
        pf[p + "_t_small_ema10"] = _ewm(ew, p + "_tsm", small, 10)
        pf[p + "_t_med_ema10"] = _ewm(ew, p + "_tmd", med, 10)
        pf[p + "_t_large_ema10"] = _ewm(ew, p + "_tlg", large, 10)
        pf[p + "_t_sqty_ema50"] = _ewm(ew, p + "_tsq", net_sqty, 50)
        if p == "SUNLIGHT_INDEX":
            pf[p + "_t_contrarian_ema10"] = _ewm(ew, p + "_tct", contra, 10)

        new_prev[p] = [mid, b2[0] if b2 else None, a2[0] if a2 else None]
        last[p] = pf
        f.update(pf)
    S["book_prev"] = new_prev

    # ---- Cross-product block --------------------------------------------------------------
    if all(p + "_mid" in f for p in ("CARROT", "POTATO", "BROCCOLI", "VEGGIE_BASKET")):
        bs = f["VEGGIE_BASKET_mid"] - (2.0 * f["CARROT_mid"] + 3.0 * f["POTATO_mid"] + f["BROCCOLI_mid"])
        cp = f["CARROT_mid"] - (5.0 / 3.0) * f["POTATO_mid"]
        bh, ch = S.setdefault("bs_hist", []), S.setdefault("cp_hist", [])
        _push(bh, bs, 11)
        _push(ch, cp, 11)
        f["BASKET_SPREAD"], f["BASKET_SPREAD_past_10"] = bs, _diff(bh, 10)
        f["CP_SPREAD"], f["CP_SPREAD_past_10"] = cp, _diff(ch, 10)
        f["SYNTH_mm_obi_ema50"] = (2.0 * f["CARROT_mm_obi_ema50"] + 3.0 * f["POTATO_mm_obi_ema50"]
                                   + f["BROCCOLI_mm_obi_ema50"]) / 6.0
    S["ts_prev"] = ts
    return f


def predict(p, f):
    """Predicted mid move at each horizon; empty dict if no fitted model for p."""
    out = {}
    for hk, m in COEFS.get(p, {}).items():
        out[hk] = m["intercept"] + sum(w * f.get(k, 0.0) for k, w in m["w"].items())
    return out


# ----------------------------------------------------------------------------------------
# Adapters from the exchange's TradingState
# ----------------------------------------------------------------------------------------
def _books_from_state(state):
    books = {}
    for p, od in state.order_depths.items():
        bids = sorted(((pr, v) for pr, v in od.buy_orders.items() if v > 0), reverse=True)
        asks = sorted((pr, -v) for pr, v in od.sell_orders.items() if v < 0)
        books[p] = (bids, asks)
    return books


def _trades_from_state(state, prev_ts):
    out = {}
    if prev_ts is None:
        return out
    for p, lst in (getattr(state, "market_trades", None) or {}).items():
        rows = []
        for tr in lst:
            t = getattr(tr, "timestamp", None)
            if t is not None and t < prev_ts:  # stale trade already counted
                continue
            rows.append((tr.price, abs(tr.quantity)))
        out[p] = rows
    return out


def _obs_from_state(state):
    """
    Best-effort read of weather observations. CHECK THIS against the round's datamodel.py:
    if it returns None live, the weather features freeze at 0 while the model was trained
    with them, which biases predictions.
    """
    o = getattr(state, "observations", None)
    if o is None:
        return None
    src = None
    if isinstance(o, dict):
        src = o
    else:
        pv = getattr(o, "plainValueObservations", None)
        if isinstance(pv, dict) and pv:
            src = pv
        else:
            src = {k: getattr(o, k) for k in ["weather", "pest_alert"] + WEATHER_NUM if hasattr(o, k)}
    if not src or "weather" not in src:
        return None
    return src


class Trader:
    def __init__(self):
        self.last_features = None  # exposed for parity tests

    def run(self, state):
        try:
            S = json.loads(state.traderData) if state.traderData else {}
        except (ValueError, TypeError):
            S = {}
        prev_ts = S.get("ts_prev")
        if prev_ts is not None and state.timestamp <= prev_ts:  # new day/session
            S, prev_ts = {}, None

        books = _books_from_state(state)
        f = update_features(S, state.timestamp, books, _trades_from_state(state, prev_ts),
                            _obs_from_state(state))
        self.last_features = f

        result = {}
        for p in PRODUCTS:
            bids, asks = books.get(p, ([], []))
            if not bids or not asks or p + "_mid" not in f:
                continue
            result[p] = self._orders_for(p, f, bids, asks, state.position.get(p, 0))
        return result, 0, json.dumps(S, separators=(",", ":"))

    @staticmethod
    def _orders_for(p, f, bids, asks, pos):
        mid = f[p + "_mid"]
        preds = predict(p, f)
        if preds:
            fair = mid + sum(HORIZON_WEIGHTS.get(h, 0.0) * v for h, v in preds.items())
        else:
            fair = mid + f[p + "_outer_dev"]  # fallback: Deep-MM mid
        lim = LIMITS[p]
        res = fair - GAMMA[p] * pos  # inventory-skewed reservation price
        buy_cap, sell_cap = lim - pos, lim + pos
        orders = []

        # Take: cross only when the edge beats the threshold.
        for ap, av in asks:
            if ap > res - TAKE_EDGE[p] or buy_cap <= 0:
                break
            q = min(av, buy_cap)
            orders.append(Order(p, ap, q))
            buy_cap -= q
        for bp, bv in bids:
            if bp < res + TAKE_EDGE[p] or sell_cap <= 0:
                break
            q = min(bv, sell_cap)
            orders.append(Order(p, bp, -q))
            sell_cap -= q

        # Make: one tick inside the touch, never past the reservation +/- edge.
        best_bid, best_ask = bids[0][0], asks[0][0]
        bid_px = min(best_bid + 1, math.floor(res - MAKE_EDGE[p]), best_ask - 1)
        ask_px = max(best_ask - 1, math.ceil(res + MAKE_EDGE[p]), best_bid + 1)
        if bid_px < ask_px:
            if buy_cap > 0:
                orders.append(Order(p, int(bid_px), min(MAKE_SIZE[p], buy_cap)))
            if sell_cap > 0:
                orders.append(Order(p, int(ask_px), -min(MAKE_SIZE[p], sell_cap)))
        return orders
