"""Ascend Algo Round 2 trader.

Underlyings: CARROT, POTATO, BROCCOLI, VEGGIE_BASKET (= 2C + 3P + 1B + 11.10)
Options:     7 strikes x call/put on each crop (42 symbols), European, r = 0.

Pipeline per tick
  1. Parse books, rebuild the outer "anchor" bot's quote (merged-level aware).
  2. Signed trade flow / level-2 sweep EMAs from last tick's market trades.
  3. Put-call-parity implied spot per crop, basket basis, underlying fair values.
  4. One IV inversion per strike, EMA/smile blend, Black-Scholes fair for all 42 options.
  5. Counterparty toxicity tracking from our own fills.
  6. Inventory/delta-skewed taker + passive orders, then a hard position-limit clamp.
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

LIMITS = {"CARROT": 80, "POTATO": 100, "BROCCOLI": 40, BASKET: 30}
OPT_LIMIT = 50
for _u in CROPS:
    for _s in CALL_SYMS[_u] + PUT_SYMS[_u]:
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
OPT_PASSIVE_EDGE = 0.25
OPT_PASSIVE_SIZE = 10
OPT_INV_SKEW = 0.015
OPT_DELTA_SKEW = 0.004

# Underlying quoting
UND_TAKE_EDGE = 0.75
UND_TAKE_SIZE = 15
UND_PASSIVE_EDGE = 0.30
UND_INV_SKEW = 0.025
UND_HEDGE_SKEW = 0.008
BASKET_INV_SKEW = 0.05
# Passive size per side (the spec leaves this open; tune from backtests).
UND_PASSIVE_SIZE = {"CARROT": 20, "POTATO": 25, "BROCCOLI": 12, BASKET: 8}

# Counterparty toxicity
TOXIC_MIN_FILLS = 6
TOXIC_WINDOW = 10
TOXIC_EXTRA_EDGE = 0.30

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
        self.cp_stats = {}      # counterparty -> [n1, sum_adv1, n5, sum_adv5]
        self.pending = deque()  # (due_tick, symbol, cp, cp_sign, price, horizon)
        self.toxic_until = {}   # symbol -> tick until which quotes stay defensive

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
            self.cp_stats = d.get("cp", {})
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
            "cp": self.cp_stats,
        }, separators=(",", ":"))

    # -- observations --
    def _read_obs(self, state):
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

    # -- counterparty toxicity --
    def _update_counterparties(self, state, mids):
        tick = self.tick
        pending = self.pending
        stats = self.cp_stats
        # score fills whose horizon has elapsed
        while pending and pending[0][0] <= tick:
            _, sym, cp, sign, price, h = pending.popleft()
            m = mids.get(sym)
            if m is None:
                continue
            st = stats.get(cp)
            if st is None:
                st = stats[cp] = [0, 0.0, 0, 0.0]
            adv = (m - price) * sign
            if h == 1:
                st[0] += 1
                st[1] += adv
            else:
                st[2] += 1
                st[3] += adv
        own = getattr(state, "own_trades", None)
        if not own:
            return
        prev_ts = self.prev_ts
        prev_tob = self.prev_tob
        new = []
        for sym, trades in own.items():
            tob = prev_tob.get(sym)
            if tob is None:
                continue
            for t in trades:
                ts = getattr(t, "timestamp", None)
                if prev_ts is not None and ts is not None and ts < prev_ts:
                    continue  # stale fill already processed
                if t.buyer == ME:
                    cp = t.seller
                    sign = -1.0  # counterparty sold to us
                    passive = tob[1] is None or t.price < tob[1]
                elif t.seller == ME:
                    cp = t.buyer
                    sign = 1.0
                    passive = tob[0] is None or t.price > tob[0]
                else:
                    continue
                if not cp or cp == ME or not passive:
                    continue  # only fills where they hit our resting quote
                new.append((sym, cp, sign, t.price))
                st = stats.get(cp)
                if st is not None and st[2] >= TOXIC_MIN_FILLS and st[3] > 0.0 and st[1] > 0.0:
                    self.toxic_until[sym] = tick + TOXIC_WINDOW
        # queue horizon-1 entries first so the deque stays sorted by due tick
        for sym, cp, sign, price in new:
            pending.append((tick, sym, cp, sign, price, 1))
        for sym, cp, sign, price in new:
            pending.append((tick + 4, sym, cp, sign, price, 5))
        if len(pending) > 4000:
            pending.clear()

    # -- main entry point --
    def run(self, state):
        if not self.loaded:
            self._load(getattr(state, "traderData", ""))
        ts = state.timestamp
        if self.prev_ts is not None and ts < self.prev_ts:
            # new day inside the same process: drop intraday flow state
            self.prev_und = {}
            self.prev_tob = {}
            self.pending.clear()
            self.toxic_until = {}
            self.basis_ema = None
            for u in UNDERLYINGS:
                self.sv[u] = 0.0
                self.l2[u] = 0.0
            self.prev_ts = None

        self._read_obs(state)
        depths = state.order_depths
        position = state.position
        tick = self.tick
        result = {}
        mids = {}
        tob = {}

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

        # ---------------- 5. counterparty toxicity ----------------
        self._update_counterparties(state, mids)
        toxic_until = self.toxic_until

        # ---------------- 6. options: IV, fair values, orders ----------------
        T = self.dte / 365.0
        if T < 1e-6:
            T = 1e-6
        net_delta = {}
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
                iv = implied_vol(sp, K, T, target, iv20[i], 4 if seeded[i] else 15)
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
                for sym, fair, delta in ((cs[i], fair_c[i], dlt[i]), (ps[i], fair_p[i], dlt[i] - 1.0)):
                    bk = obook.get(sym)
                    if bk is None:
                        continue
                    bid1, bv1, bid2, bv2, ask1, av1, ask2, av2 = bk
                    pos = position.get(sym, 0)
                    res = fair - OPT_INV_SKEW * pos - OPT_DELTA_SKEW * delta * nd
                    buy_cap = OPT_LIMIT - pos
                    sell_cap = OPT_LIMIT + pos
                    orders = []
                    if ask1 is not None and buy_cap > 0 and res - ask1 > OPT_TAKE_EDGE:
                        q = min(av1, buy_cap, OPT_TAKE_SIZE)
                        if q > 0:
                            orders.append(Order(sym, int(ask1), int(q)))
                            buy_cap -= q
                        if ask2 is not None and buy_cap > 0 and res - ask2 > OPT_SWEEP_EDGE:
                            q = min(av2, buy_cap, OPT_TAKE_SIZE)
                            if q > 0:
                                orders.append(Order(sym, int(ask2), int(q)))
                                buy_cap -= q
                    if bid1 is not None and sell_cap > 0 and bid1 - res > OPT_TAKE_EDGE:
                        q = min(bv1, sell_cap, OPT_TAKE_SIZE)
                        if q > 0:
                            orders.append(Order(sym, int(bid1), -int(q)))
                            sell_cap -= q
                        if bid2 is not None and sell_cap > 0 and bid2 - res > OPT_SWEEP_EDGE:
                            q = min(bv2, sell_cap, OPT_TAKE_SIZE)
                            if q > 0:
                                orders.append(Order(sym, int(bid2), -int(q)))
                                sell_cap -= q
                    if bid1 is not None and ask1 is not None:
                        edge = OPT_PASSIVE_EDGE
                        size = OPT_PASSIVE_SIZE
                        if toxic_until.get(sym, -1) >= tick:
                            edge += TOXIC_EXTRA_EDGE
                            size //= 2
                        pb = min(bid1 + 1, _floor(res - edge))
                        pa = max(ask1 - 1, _ceil(res + edge))
                        if pb < pa:
                            if bid1 <= pb < ask1 and buy_cap > 0:
                                orders.append(Order(sym, int(pb), int(min(size, buy_cap))))
                            if bid1 < pa <= ask1 and sell_cap > 0:
                                orders.append(Order(sym, int(pa), -int(min(size, sell_cap))))
                    if orders:
                        result[sym] = orders

        # ---------------- 7. underlyings: orders ----------------
        for u in UNDERLYINGS:
            bk = ubook.get(u)
            if bk is None or u not in s_fair:
                continue
            bid1, bv1, bid2, bv2, ask1, av1, ask2, av2 = bk
            pos = position.get(u, 0)
            limit = LIMITS[u]
            if u == BASKET:
                res = s_fair[u] - BASKET_INV_SKEW * pos
            else:
                res = s_fair[u] - UND_INV_SKEW * pos - UND_HEDGE_SKEW * net_delta.get(u, 0.0)
            buy_cap = limit - pos
            sell_cap = limit + pos
            orders = []
            if buy_cap > 0 and res - ask1 > UND_TAKE_EDGE:
                q = min(av1, buy_cap, UND_TAKE_SIZE)
                if q > 0:
                    orders.append(Order(u, int(ask1), int(q)))
                    buy_cap -= q
            if sell_cap > 0 and bid1 - res > UND_TAKE_EDGE:
                q = min(bv1, sell_cap, UND_TAKE_SIZE)
                if q > 0:
                    orders.append(Order(u, int(bid1), -int(q)))
                    sell_cap -= q
            edge = UND_PASSIVE_EDGE
            size = UND_PASSIVE_SIZE[u]
            if toxic_until.get(u, -1) >= tick:
                edge += TOXIC_EXTRA_EDGE
                size //= 2
            pb = min(bid1 + 1, _floor(res - edge))
            pa = max(ask1 - 1, _ceil(res + edge))
            if pb < pa:
                if bid1 <= pb < ask1 and buy_cap > 0:
                    orders.append(Order(u, int(pb), int(min(size, buy_cap))))
                if bid1 < pa <= ask1 and sell_cap > 0:
                    orders.append(Order(u, int(pa), -int(min(size, sell_cap))))
            if orders:
                result[u] = orders

        # ---------------- 8. hard position-limit clamp ----------------
        for sym, orders in result.items():
            limit = LIMITS.get(sym)
            if limit is None:
                continue
            pos = position.get(sym, 0)
            buy_cap = limit - pos
            sell_cap = limit + pos
            buys = sells = 0
            ok = True
            for o in orders:
                if o.quantity > 0:
                    buys += o.quantity
                else:
                    sells -= o.quantity
            if buys > buy_cap or sells > sell_cap:
                ok = False
            if not ok:
                kept = []
                for o in orders:
                    q = o.quantity
                    if q > 0:
                        q = min(q, buy_cap)
                        buy_cap -= q
                    else:
                        q = -min(-q, sell_cap)
                        sell_cap += q
                    if q:
                        kept.append(Order(sym, o.price, q))
                result[sym] = kept

        self.prev_tob = tob
        self.prev_ts = ts
        self.tick = tick + 1
        return result, 0, self._dump()
