"""Ascend Algo Round 2 trader.

Underlyings: CARROT, POTATO, BROCCOLI, VEGGIE_BASKET (= 2C + 3P + 1B + 11.10)
Options:     7 strikes x call/put on each crop (42 symbols), European, r = 0.

Pipeline per tick
  1. Parse books, rebuild the outer "anchor" bot's quote (merged-level aware).
  2. Signed trade flow / level-2 sweep EMAs from last tick's market trades.
  3. Put-call-parity implied spot per crop, basket basis, underlying fair values.
  4. Defence bookkeeping: own fills -> counterparty markouts, sniper quarantine,
     queue-starvation streaks, open-interest stampedes, per-symbol drawdown.
  5. One IV inversion per strike, EMA/smile blend, Black-Scholes fair for all 42 options.
  6. Cubic-skewed taker + passive orders with anti-sniping vetoes, delta-hedge clamp.
  7. Order sanitizer: nothing can breach a position limit.
"""
import json
import math
from collections import deque

try:
    from datamodel import Order, TradingState
except ImportError:  # local testing without the platform's datamodel
    TradingState = object

    class Order:
        def __init__(self, symbol, price, quantity):
            self.symbol = symbol
            self.price = price
            self.quantity = quantity

        def __repr__(self):
            return "(%s, %s, %s)" % (self.symbol, self.price, self.quantity)


# ----------------------------------------------------------------------------
# Static configuration
# ----------------------------------------------------------------------------
CROPS = ("CARROT", "POTATO", "BROCCOLI")
BASKET = "VEGGIE_BASKET"
UNDERLYINGS = CROPS + (BASKET,)

STRIKES = {
    "CARROT": (725, 850, 975, 1100, 1225, 1350, 1475),
    "POTATO": (450, 525, 600, 675, 750, 825, 900),
    "BROCCOLI": (2075, 2200, 2325, 2450, 2575, 2700, 2825),
}
CALL_SYMS = {u: tuple("%s_C_%d" % (u, k) for k in STRIKES[u]) for u in CROPS}
PUT_SYMS = {u: tuple("%s_P_%d" % (u, k) for k in STRIKES[u]) for u in CROPS}
OPTION_SYMS = tuple(s for u in CROPS for s in CALL_SYMS[u] + PUT_SYMS[u])
OI_KEY = {s: "oi_" + s for s in OPTION_SYMS}

LIMITS = {"CARROT": 80, "POTATO": 100, "BROCCOLI": 40, BASKET: 30}
OPT_LIMIT = 50
for _s in OPTION_SYMS:
    LIMITS[_s] = OPT_LIMIT

BASKET_W = {"CARROT": 2.0, "POTATO": 3.0, "BROCCOLI": 1.0}
BASKET_PREMIUM = 11.10

# S_fair = mid1 + w_pcp*(S_pcp-mid1) + w_m2*(true_mid2-mid1) + w_sv*sv_ema5
#          + w_l2*l2_sweep_vol + w_l2e*l2_ema5 + w_basis*basis_dev
FAIR_W = {
    "CARROT": (0.38, 0.24, 0.080, 0.00, 0.00, 0.018),
    "POTATO": (0.37, 0.22, 0.075, 0.00, 0.00, 0.020),
    "BROCCOLI": (0.36, 0.22, 0.150, 0.30, 0.25, 0.015),
}

IV_INIT = {"CARROT": 1.15, "POTATO": 2.15, "BROCCOLI": 0.62}
A20 = 2.0 / 21.0
A100 = 2.0 / 101.0
A5 = 2.0 / 6.0

# Option quoting
OPT_TAKE_EDGE = 0.65
OPT_SWEEP_EDGE = 0.90
OPT_TAKE_SIZE = 12
OPT_PASSIVE_SIZE = 10
OPT_K1 = 0.015
OPT_K3 = 0.045
OPT_DELTA_SKEW = 0.004

# Underlying quoting
UND_TAKE_EDGE = 0.75
UND_TAKE_SIZE = 15
UND_K1 = 0.025
UND_K3 = 0.060
UND_HEDGE_SKEW = 0.008
# Passive size per side (the spec leaves this open; tune from backtests).
UND_PASSIVE_SIZE = {"CARROT": 20, "POTATO": 25, "BROCCOLI": 12, BASKET: 8}

# ---- Multi-player defence ----
PASSIVE_EDGE = 0.35          # min distance of any passive quote from reservation price
VETO_ALPHA = 0.35            # |fair - mid| above which the leaning inner quote is vetoed
INV_BLOCK_RATIO = 0.75       # no same-direction passive orders beyond this |pos|/limit

MARKOUT_HORIZON = 5          # ticks
CP_ALPHA = 0.15
CP_MIN_FILLS = 3
SNIPER_MARKOUT = -0.15
SOFT_MARKOUT = 0.30
QUARANTINE_TICKS = 10
QUARANTINE_SPREAD = 0.50
QUARANTINE_SHIFT = 0.35

STARVE_STREAK = 2
STARVE_EXPIRY = 50           # ticks without new evidence before a streak is forgotten
STARVE_MIN_SPREAD = 5
STARVE_STEP_EDGE = 1.10

OI_JUMP = 12
OI_TICKS = 5
OI_SPREAD = 0.40

HEDGE_RATIO = 1.25
HEDGE_SIZE = 10

DD_LIMIT = 1500.0
DD_TICKS = 200
DD_TAKE_EDGE = 1.50

PEST_MAP = {"none": 0.0, "low": 1.0, "high": 2.0}
ME = "SUBMISSION"

_erf = math.erf
_log = math.log
_exp = math.exp
_sqrt = math.sqrt
_floor = math.floor
_ceil = math.ceil
INV_SQRT2 = 0.7071067811865475
INV_SQRT2PI = 0.3989422804014327


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def bs_call_delta(S, K, T, sigma):
    """Zero-rate Black-Scholes call price and call delta."""
    vs = sigma * _sqrt(T)
    if vs < 1e-8:
        vs = 1e-8
    d1 = _log(S / K) / vs + 0.5 * vs
    nd1 = 0.5 * (1.0 + _erf(d1 * INV_SQRT2))
    nd2 = 0.5 * (1.0 + _erf((d1 - vs) * INV_SQRT2))
    return S * nd1 - K * nd2, nd1


def implied_vol(S, K, T, target_call, sigma, iters):
    """Newton-Raphson on the call price, warm-started at `sigma`. None on failure."""
    sqrtT = _sqrt(T)
    lsk = _log(S / K)
    for _ in range(iters):
        vs = sigma * sqrtT
        d1 = lsk / vs + 0.5 * vs
        price = S * 0.5 * (1.0 + _erf(d1 * INV_SQRT2)) - K * 0.5 * (1.0 + _erf((d1 - vs) * INV_SQRT2))
        vega = S * INV_SQRT2PI * _exp(-0.5 * d1 * d1) * sqrtT
        if vega < 1e-6:
            return None
        step = (price - target_call) / vega
        lim = 0.5 * sigma
        if step > lim:
            step = lim
        elif step < -lim:
            step = -lim
        sigma -= step
        if -1e-7 < step < 1e-7:
            break
    if 0.01 < sigma < 10.0:
        return sigma
    return None


def read_book(od):
    """-> (bid1, bvol1, bid2, bvol2, ask1, avol1, ask2, avol2); None where absent."""
    bid1 = bid2 = ask1 = ask2 = None
    bv1 = bv2 = av1 = av2 = 0
    b = od.buy_orders
    if b:
        if len(b) == 1:
            for bid1, bv1 in b.items():
                pass
        else:
            ps = sorted(b, reverse=True)
            bid1 = ps[0]
            bid2 = ps[1]
            bv1 = b[bid1]
            bv2 = b[bid2]
    s = od.sell_orders
    if s:
        if len(s) == 1:
            for ask1, av1 in s.items():
                pass
        else:
            ps = sorted(s)
            ask1 = ps[0]
            ask2 = ps[1]
            av1 = s[ask1]
            av2 = s[ask2]
    return bid1, abs(bv1), bid2, abs(bv2), ask1, abs(av1), ask2, abs(av2)


def quad_fit(xs, ys):
    """Least-squares y = a + b x + c x^2. Returns (a, b, c) or None."""
    n = len(xs)
    if n < 3:
        return None
    sx = sx2 = sx3 = sx4 = sy = sxy = sx2y = 0.0
    for x, y in zip(xs, ys):
        x2 = x * x
        sx += x
        sx2 += x2
        sx3 += x2 * x
        sx4 += x2 * x2
        sy += y
        sxy += x * y
        sx2y += x2 * y
    det = n * (sx2 * sx4 - sx3 * sx3) - sx * (sx * sx4 - sx3 * sx2) + sx2 * (sx * sx3 - sx2 * sx2)
    if abs(det) < 1e-14:
        return None
    a = (sy * (sx2 * sx4 - sx3 * sx3) - sx * (sxy * sx4 - sx3 * sx2y) + sx2 * (sxy * sx3 - sx2 * sx2y)) / det
    b = (n * (sxy * sx4 - sx2y * sx3) - sy * (sx * sx4 - sx3 * sx2) + sx2 * (sx * sx2y - sxy * sx2)) / det
    c = (n * (sx2 * sx2y - sx3 * sxy) - sx * (sx * sx2y - sxy * sx2) + sy * (sx * sx3 - sx2 * sx2)) / det
    return a, b, c


def sanitize_orders(sym, pos, limit, cands):
    """cands: list of (priority, price, qty); priority 0 = hedge, 1 = taker, 2 = passive.

    Risk-reducing orders go first, then by priority; quantities are clamped to
    the remaining buy/sell capacity so the symbol can never be rejected.
    """
    if len(cands) > 1:
        if pos > 0:
            cands.sort(key=lambda c: (0 if c[2] < 0 else 1, c[0]))
        elif pos < 0:
            cands.sort(key=lambda c: (0 if c[2] > 0 else 1, c[0]))
        else:
            cands.sort(key=lambda c: c[0])
    rem_buy = limit - pos
    rem_sell = limit + pos
    out = []
    for _, price, q in cands:
        if q > 0:
            if q > rem_buy:
                q = rem_buy
            rem_buy -= q
        else:
            if -q > rem_sell:
                q = -rem_sell
            rem_sell += q
        if q:
            out.append(Order(sym, int(price), int(q)))
    return out


# ----------------------------------------------------------------------------
# Trader
# ----------------------------------------------------------------------------
class Trader:
    def __init__(self):
        self.loaded = False
        self.tick = 0
        self.prev_ts = None
        self.prev_und = {}      # underlying -> (mid1, bid2, ask2) from last tick
        self.prev_tob = {}      # symbol -> (bid1, ask1) from last tick
        self.sv = {u: 0.0 for u in UNDERLYINGS}
        self.l2 = {u: 0.0 for u in UNDERLYINGS}
        self.basis_ema = None
        self.iv20 = {u: [IV_INIT[u]] * 7 for u in CROPS}
        self.iv100 = {u: [IV_INIT[u]] * 7 for u in CROPS}
        self.iv_seeded = {u: [False] * 7 for u in CROPS}
        self.dte = 15.0
        self.pest = 0.0
        # --- defence state ---
        self.cp_ema = {}        # counterparty -> EMA of our 5-tick markout against them
        self.cp_fills = {}      # counterparty -> scored fills
        self.fills = deque()    # (due_tick, symbol, price, our_sign, cp)
        self.quarantine = {}    # symbol -> (until_tick, fair_shift)
        self.harvest_until = {}  # symbol -> tick (soft counterparty active)
        self.last_passive = {}  # symbol -> (bid_px | None, ask_px | None) posted last tick
        self.starve = {}        # symbol -> [streak, last_evidence_tick]
        self.prev_oi = {}       # option -> external open interest last tick
        self.oi_until = {}      # option -> tick
        self.cash = {}          # symbol -> intraday cash flow
        self.peak = {}          # symbol -> peak intraday pnl
        self.reduce_until = {}  # symbol -> tick (reduce-only mode)

    # -- persistence fallback (only used if the runner re-instantiates Trader) --
    def _load(self, data):
        self.loaded = True
        if not data:
            return
        try:
            d = json.loads(data)
            for u in CROPS:
                self.iv20[u] = list(d["i2"][u])
                self.iv100[u] = list(d["i1"][u])
                self.iv_seeded[u] = [bool(x) for x in d["is"][u]]
            self.basis_ema = d.get("be")
            self.dte = d.get("dte", self.dte)
            self.cp_ema = d.get("ce", {})
            self.cp_fills = d.get("cf", {})
            for u in UNDERLYINGS:
                self.sv[u] = d["sv"].get(u, 0.0)
                self.l2[u] = d["l2"].get(u, 0.0)
        except Exception:
            pass

    def _dump(self):
        r = round
        return json.dumps({
            "i2": {u: [r(x, 5) for x in self.iv20[u]] for u in CROPS},
            "i1": {u: [r(x, 5) for x in self.iv100[u]] for u in CROPS},
            "is": {u: [1 if x else 0 for x in self.iv_seeded[u]] for u in CROPS},
            "be": self.basis_ema,
            "dte": self.dte,
            "sv": {u: r(v, 4) for u, v in self.sv.items()},
            "l2": {u: r(v, 4) for u, v in self.l2.items()},
            "ce": self.cp_ema,
            "cf": self.cp_fills,
        }, separators=(",", ":"))

    def _new_day(self):
        self.prev_und = {}
        self.prev_tob = {}
        self.prev_ts = None
        self.basis_ema = None
        for u in UNDERLYINGS:
            self.sv[u] = 0.0
            self.l2[u] = 0.0
        self.fills.clear()
        self.quarantine = {}
        self.harvest_until = {}
        self.last_passive = {}
        self.starve = {}
        self.prev_oi = {}
        self.oi_until = {}
        self.cash = {}
        self.peak = {}
        self.reduce_until = {}

    # -- observations: expiry, pest alert, de-spoofed open interest --
    def _read_obs(self, state, position):
        try:
            pv = state.observations.plainValueObservations
        except AttributeError:
            return
        if not pv:
            return
        d = pv.get("days_to_expiry")
        if d is not None:
            try:
                self.dte = float(d)
            except (TypeError, ValueError):
                pass
        p = pv.get("pest_alert")
        if p is not None:
            if isinstance(p, str):
                self.pest = PEST_MAP.get(p.lower(), 0.0)
            else:
                try:
                    self.pest = float(p)
                except (TypeError, ValueError):
                    pass
        prev_oi = self.prev_oi
        tick = self.tick
        for sym in OPTION_SYMS:
            v = pv.get(OI_KEY[sym])
            if v is None:
                continue
            try:
                ext = float(v) - abs(position.get(sym, 0))
            except (TypeError, ValueError):
                continue
            prev = prev_oi.get(sym)
            if prev is not None and abs(ext - prev) >= OI_JUMP:
                self.oi_until[sym] = tick + OI_TICKS  # multi-player stampede
            prev_oi[sym] = ext

    # -- own fills: cash, counterparty markouts, quarantine, queue starvation, drawdown --
    def _defence(self, state, mids, position, mtrades):
        tick = self.tick
        prev_ts = self.prev_ts
        cp_ema = self.cp_ema
        cp_fills = self.cp_fills

        # score fills whose 5-tick horizon has elapsed
        fills = self.fills
        while fills and fills[0][0] <= tick:
            _, sym, price, sign, cp = fills.popleft()
            m = mids.get(sym)
            if m is None:
                continue
            mo = sign * (m - price)
            e = cp_ema.get(cp)
            cp_ema[cp] = mo if e is None else e + CP_ALPHA * (mo - e)
            cp_fills[cp] = cp_fills.get(cp, 0) + 1

        got_buy = set()
        got_sell = set()
        own = getattr(state, "own_trades", None)
        if own:
            cash = self.cash
            prev_tob = self.prev_tob
            for sym, trades in own.items():
                tob = prev_tob.get(sym)
                for t in trades:
                    ts = getattr(t, "timestamp", None)
                    if prev_ts is not None and ts is not None and ts < prev_ts:
                        continue  # stale fill already processed
                    if t.buyer == ME:
                        sign = 1.0
                        cp = t.seller
                        got_buy.add(sym)
                        passive = tob is not None and (tob[1] is None or t.price < tob[1])
                    elif t.seller == ME:
                        sign = -1.0
                        cp = t.buyer
                        got_sell.add(sym)
                        passive = tob is not None and (tob[0] is None or t.price > tob[0])
                    else:
                        continue
                    cash[sym] = cash.get(sym, 0.0) - sign * t.price * t.quantity
                    if not passive:
                        continue  # we were the aggressor: says nothing about the counterparty
                    st = self.starve.get(sym)
                    if st is not None:
                        st[0] = 0
                    if not cp or cp == ME:
                        continue
                    fills.append((tick + MARKOUT_HORIZON - 1, sym, t.price, sign, cp))
                    if cp_fills.get(cp, 0) >= CP_MIN_FILLS:
                        e = cp_ema[cp]
                        if e < SNIPER_MARKOUT:
                            # follow the sniper's direction instead of fighting it
                            self.quarantine[sym] = (tick + QUARANTINE_TICKS, -QUARANTINE_SHIFT * sign)
                        elif e > SOFT_MARKOUT:
                            self.harvest_until[sym] = tick + QUARANTINE_TICKS
            if len(fills) > 4000:
                fills.clear()

        # queue starvation: the market traded through our resting price but we got nothing
        if mtrades:
            starve = self.starve
            for sym, (lb, la) in self.last_passive.items():
                tr = mtrades.get(sym)
                if not tr:
                    continue
                hit = False
                for t in tr:
                    ts = getattr(t, "timestamp", None)
                    if prev_ts is not None and ts is not None and ts < prev_ts:
                        continue
                    p = t.price
                    if (lb is not None and p <= lb and sym not in got_buy) or \
                            (la is not None and p >= la and sym not in got_sell):
                        hit = True
                        break
                if hit:
                    st = starve.get(sym)
                    if st is None:
                        starve[sym] = [1, tick]
                    else:
                        st[0] += 1
                        st[1] = tick

        # per-symbol drawdown circuit breaker
        peak = self.peak
        for sym, c in self.cash.items():
            m = mids.get(sym)
            if m is None:
                continue
            pnl = c + position.get(sym, 0) * m
            pk = peak.get(sym)
            if pk is None or pnl > pk:
                peak[sym] = pnl
            elif pk - pnl > DD_LIMIT:
                self.reduce_until[sym] = tick + DD_TICKS
                peak[sym] = pnl  # re-arm from here

    # -- order generation for one symbol --
    def _quote(self, sym, bk, pos, limit, fair, extra_skew, is_opt, hedge):
        """Returns sanitized orders. `hedge` > 0 forces a buy at ask1, < 0 a sell at bid1."""
        bid1, bv1, bid2, bv2, ask1, av1, ask2, av2 = bk
        tick = self.tick
        if is_opt:
            k1, k3 = OPT_K1, OPT_K3
            take_edge, take_size, size = OPT_TAKE_EDGE, OPT_TAKE_SIZE, OPT_PASSIVE_SIZE
        else:
            k1, k3 = UND_K1, UND_K3
            take_edge, take_size, size = UND_TAKE_EDGE, UND_TAKE_SIZE, UND_PASSIVE_SIZE[sym]

        tox = 0.0
        q = self.quarantine.get(sym)
        if q is not None and q[0] >= tick:
            tox += QUARANTINE_SPREAD
            fair += q[1]
            size = max(1, size // 2)
        elif self.harvest_until.get(sym, -1) >= tick:
            size += size // 2
        if is_opt and self.oi_until.get(sym, -1) >= tick:
            tox += OI_SPREAD

        ratio = pos / limit
        res = fair - limit * (k1 * ratio + k3 * ratio * ratio * ratio) - extra_skew
        reduce_only = self.reduce_until.get(sym, -1) >= tick
        buy_cap = limit - pos
        sell_cap = limit + pos
        cands = []

        if hedge > 0 and ask1 is not None:
            cands.append((0, ask1, min(hedge, av1, buy_cap)))
        elif hedge < 0 and bid1 is not None:
            cands.append((0, bid1, -min(-hedge, bv1, sell_cap)))

        # ---- takers ----
        if ask1 is not None and hedge >= 0:
            edge = res - ask1
            if edge > take_edge:
                cap = take_size
                if reduce_only and edge <= DD_TAKE_EDGE:
                    cap = -pos if pos < 0 else 0
                qty = min(av1, buy_cap, cap)
                if qty > 0:
                    cands.append((1, ask1, qty))
                    buy_cap -= qty
                    if is_opt and ask2 is not None and res - ask2 > OPT_SWEEP_EDGE and not reduce_only:
                        qty = min(av2, buy_cap, take_size)
                        if qty > 0:
                            cands.append((1, ask2, qty))
                            buy_cap -= qty
        if bid1 is not None and hedge <= 0:
            edge = bid1 - res
            if edge > take_edge:
                cap = take_size
                if reduce_only and edge <= DD_TAKE_EDGE:
                    cap = pos if pos > 0 else 0
                qty = min(bv1, sell_cap, cap)
                if qty > 0:
                    cands.append((1, bid1, -qty))
                    sell_cap -= qty
                    if is_opt and bid2 is not None and bid2 - res > OPT_SWEEP_EDGE and not reduce_only:
                        qty = min(bv2, sell_cap, take_size)
                        if qty > 0:
                            cands.append((1, bid2, -qty))
                            sell_cap -= qty

        # ---- passive quotes ----
        lb = la = None
        if bid1 is not None and ask1 is not None:
            pb = bid1 + 1
            pa = ask1 - 1
            st = self.starve.get(sym)
            if st is not None and st[0] >= STARVE_STREAK:
                if tick - st[1] > STARVE_EXPIRY:
                    st[0] = 0
                else:
                    # a faster player owns the penny queue: jump it if it pays, else fall back
                    wide = ask1 - bid1 >= STARVE_MIN_SPREAD
                    pb = bid1 + 2 if (wide and res - (bid1 + 2) > STARVE_STEP_EDGE) else bid1
                    pa = ask1 - 2 if (wide and (ask1 - 2) - res > STARVE_STEP_EDGE) else ask1
            edge = PASSIVE_EDGE + tox
            pb = min(pb, _floor(res - edge))
            pa = max(pa, _ceil(res + edge))
            # anti-sniping veto: never rest inside the spread against our own alpha
            alpha = fair - 0.5 * (bid1 + ask1)
            if alpha > VETO_ALPHA and pa < ask1:
                pa = ask1
            elif alpha < -VETO_ALPHA and pb > bid1:
                pb = bid1
            if pb < pa:
                if bid1 <= pb < ask1 and buy_cap > 0 and ratio <= INV_BLOCK_RATIO and hedge >= 0:
                    qty = min(size, buy_cap)
                    if reduce_only:
                        qty = min(qty, -pos) if pos < 0 else 0
                    if qty > 0:
                        cands.append((2, pb, qty))
                        lb = pb
                if bid1 < pa <= ask1 and sell_cap > 0 and ratio >= -INV_BLOCK_RATIO and hedge <= 0:
                    qty = min(size, sell_cap)
                    if reduce_only:
                        qty = min(qty, pos) if pos > 0 else 0
                    if qty > 0:
                        cands.append((2, pa, -qty))
                        la = pa
        if lb is not None or la is not None:
            self._new_passive[sym] = (lb, la)
        if not cands:
            return None
        return sanitize_orders(sym, pos, limit, cands)

    # -- main entry point --
    def run(self, state):
        if not self.loaded:
            self._load(getattr(state, "traderData", ""))
        ts = state.timestamp
        if self.prev_ts is not None and ts < self.prev_ts:
            self._new_day()

        depths = state.order_depths
        position = state.position
        self._read_obs(state, position)
        result = {}
        mids = {}
        tob = {}
        self._new_passive = {}

        # ---------------- 1. underlying books ----------------
        ubook = {}
        mid1 = {}
        tmid2 = {}
        und_now = {}
        for u in UNDERLYINGS:
            od = depths.get(u)
            if od is None:
                continue
            bk = read_book(od)
            bid1, bv1, bid2, bv2, ask1, av1, ask2, av2 = bk
            tob[u] = (bid1, ask1)
            if bid1 is None or ask1 is None:
                continue
            ubook[u] = bk
            m = 0.5 * (bid1 + ask1)
            mid1[u] = mids[u] = m
            # Anchor bot sits at level 2, unless merged into level 1 (level 2
            # missing) or crossed over (level 1 carries the big size).
            tb = bid1 if (bid2 is None or bv1 > bv2 * 1.35) else bid2
            ta = ask1 if (ask2 is None or av1 > av2 * 1.35) else ask2
            tmid2[u] = 0.5 * (tb + ta)
            und_now[u] = (m, bid2, ask2)

        # ---------------- 2. signed flow / level-2 sweeps ----------------
        mtrades = getattr(state, "market_trades", None) or {}
        prev_ts = self.prev_ts
        l2_raw = {}
        for u in UNDERLYINGS:
            sv = 0.0
            l2 = 0.0
            pu = self.prev_und.get(u)
            tr = mtrades.get(u)
            if pu is not None and tr:
                pm, pb2, pa2 = pu
                for t in tr:
                    tts = getattr(t, "timestamp", None)
                    if prev_ts is not None and tts is not None and tts < prev_ts:
                        continue
                    q = t.quantity
                    p = t.price
                    sv += q if p >= pm else -q
                    if pa2 is not None and p >= pa2:
                        l2 += q
                    elif pb2 is not None and p <= pb2:
                        l2 -= q
            l2_raw[u] = l2
            self.sv[u] += A5 * (sv - self.sv[u])
            self.l2[u] += A5 * (l2 - self.l2[u])
        self.prev_und = und_now

        # ---------------- 3. option books, put-call-parity spot ----------------
        obook = {}
        cmid = {}
        pmid = {}
        s_pcp = {}
        for u in CROPS:
            ks = STRIKES[u]
            cs = CALL_SYMS[u]
            ps = PUT_SYMS[u]
            cm = [None] * 7
            pm_ = [None] * 7
            tot = 0.0
            n = 0
            for i in range(7):
                od = depths.get(cs[i])
                if od is not None:
                    bk = read_book(od)
                    obook[cs[i]] = bk
                    tob[cs[i]] = (bk[0], bk[4])
                    if bk[0] is not None and bk[4] is not None:
                        cm[i] = mids[cs[i]] = 0.5 * (bk[0] + bk[4])
                od = depths.get(ps[i])
                if od is not None:
                    bk = read_book(od)
                    obook[ps[i]] = bk
                    tob[ps[i]] = (bk[0], bk[4])
                    if bk[0] is not None and bk[4] is not None:
                        pm_[i] = mids[ps[i]] = 0.5 * (bk[0] + bk[4])
                if cm[i] is not None and pm_[i] is not None:
                    tot += cm[i] - pm_[i] + ks[i]
                    n += 1
            cmid[u] = cm
            pmid[u] = pm_
            if n:
                s_pcp[u] = tot / n
            elif u in mid1:
                s_pcp[u] = mid1[u]

        # ---------------- 4. basket basis & underlying fair values ----------------
        basis_dev = 0.0
        if len(mid1) == 4:
            basis = mid1[BASKET] - (2.0 * mid1["CARROT"] + 3.0 * mid1["POTATO"] + mid1["BROCCOLI"] + BASKET_PREMIUM)
            if self.basis_ema is None:
                self.basis_ema = basis
            else:
                self.basis_ema += A20 * (basis - self.basis_ema)
            basis_dev = basis - self.basis_ema

        s_fair = {}
        for u in CROPS:
            m = mid1.get(u)
            if m is None:
                if u in s_pcp:
                    s_fair[u] = s_pcp[u]  # no underlying book: still price options
                continue
            w = FAIR_W[u]
            s_fair[u] = (m + w[0] * (s_pcp[u] - m) + w[1] * (tmid2[u] - m) + w[2] * self.sv[u]
                         + w[3] * l2_raw[u] + w[4] * self.l2[u] + w[5] * basis_dev)
        if BASKET in mid1:
            m = mid1[BASKET]
            f = m + 0.26 * (tmid2[BASKET] - m) - 0.12 * basis_dev + 0.35 * self.sv[BASKET] + 0.15 * self.pest
            if len(s_fair) == 3:
                synth = 2.0 * s_fair["CARROT"] + 3.0 * s_fair["POTATO"] + s_fair["BROCCOLI"] + BASKET_PREMIUM
                f += 0.25 * (synth - m)
            s_fair[BASKET] = f

        # ---------------- 5. multi-player defence bookkeeping ----------------
        self._defence(state, mids, position, mtrades)

        # ---------------- 6. options: IV, fair values, orders ----------------
        T = self.dte / 365.0
        if T < 1e-6:
            T = 1e-6
        net_delta = {}
        quote = self._quote
        for u in CROPS:
            sp = s_pcp.get(u)
            sf = s_fair.get(u)
            if sp is None or sf is None or sp <= 0.0 or sf <= 0.0:
                net_delta[u] = 0.0
                continue
            ks = STRIKES[u]
            cm = cmid[u]
            pm_ = pmid[u]
            iv20 = self.iv20[u]
            iv100 = self.iv100[u]
            seeded = self.iv_seeded[u]

            # one inversion per strike on the parity-averaged call-equivalent price
            xs = []
            ys = []
            for i in range(7):
                K = ks[i]
                c = cm[i]
                p = pm_[i]
                if c is not None and p is not None:
                    target = 0.5 * (c + p + sp - K)
                elif c is not None and K >= sp:
                    target = c
                elif p is not None and K < sp:
                    target = p + sp - K
                else:
                    continue
                intrinsic = sp - K if sp > K else 0.0
                if target - intrinsic < 0.02:
                    continue
                iv = implied_vol(sp, K, T, target, iv20[i], 3 if seeded[i] else 15)
                if iv is None:
                    continue
                if seeded[i]:
                    iv20[i] += A20 * (iv - iv20[i])
                    iv100[i] += A100 * (iv - iv100[i])
                else:
                    iv20[i] = iv100[i] = iv
                    seeded[i] = True
                xs.append(_log(K / sp))
                ys.append(iv)
            smile = quad_fit(xs, ys)

            # fair values and deltas at the predicted spot
            cs = CALL_SYMS[u]
            ps = PUT_SYMS[u]
            fair_c = [0.0] * 7
            fair_p = [0.0] * 7
            dlt = [0.0] * 7
            nd = 0.0
            for i in range(7):
                K = ks[i]
                e20 = iv20[i]
                sm = e20
                if smile is not None:
                    x = _log(K / sp)
                    sm = smile[0] + smile[1] * x + smile[2] * x * x
                    if sm < 0.5 * e20:
                        sm = 0.5 * e20
                    elif sm > 1.5 * e20:
                        sm = 1.5 * e20
                ivt = 0.75 * e20 + 0.15 * sm + 0.10 * iv100[i]
                tc, dc = bs_call_delta(sf, K, T, ivt)
                tp = tc - sf + K
                c = cm[i]
                p = pm_[i]
                # strike-level parity snapback: lean each leg toward the other leg's parity value
                fair_c[i] = 0.85 * tc + 0.15 * (p + sf - K) if p is not None else tc
                fair_p[i] = 0.85 * tp + 0.15 * (c - sf + K) if c is not None else tp
                dlt[i] = dc
                pc = position.get(cs[i], 0)
                pp = position.get(ps[i], 0)
                if pc:
                    nd += pc * dc
                if pp:
                    nd += pp * (dc - 1.0)
            net_delta[u] = nd

            for i in range(7):
                dc = dlt[i]
                bk = obook.get(cs[i])
                if bk is not None:
                    o = quote(cs[i], bk, position.get(cs[i], 0), OPT_LIMIT, fair_c[i],
                              OPT_DELTA_SKEW * dc * nd, True, 0)
                    if o:
                        result[cs[i]] = o
                bk = obook.get(ps[i])
                if bk is not None:
                    o = quote(ps[i], bk, position.get(ps[i], 0), OPT_LIMIT, fair_p[i],
                              OPT_DELTA_SKEW * (dc - 1.0) * nd, True, 0)
                    if o:
                        result[ps[i]] = o

        # ---------------- 7. underlyings: orders ----------------
        pos_basket = position.get(BASKET, 0)
        for u in UNDERLYINGS:
            bk = ubook.get(u)
            if bk is None or u not in s_fair:
                continue
            pos = position.get(u, 0)
            limit = LIMITS[u]
            extra = 0.0
            hedge = 0
            if u != BASKET:
                nd = net_delta.get(u, 0.0)
                extra = UND_HEDGE_SKEW * nd
                tot = pos + BASKET_W[u] * pos_basket + nd
                if tot < -HEDGE_RATIO * limit and pos < limit:
                    hedge = HEDGE_SIZE
                elif tot > HEDGE_RATIO * limit and pos > -limit:
                    hedge = -HEDGE_SIZE
            o = quote(u, bk, pos, limit, s_fair[u], extra, False, hedge)
            if o:
                result[u] = o

        self.last_passive = self._new_passive
        self.prev_tob = tob
        self.prev_ts = ts
        self.tick += 1
        return result, 0, self._dump()
