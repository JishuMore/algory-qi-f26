"""Fit per-horizon forecast coefficients for trader.py and compare to the baseline features.

Replays each day through backtest.run_day with non-trading loggers so features are built
from exactly what the live trader sees. Reports cross-day out-of-sample R^2 and writes
(--write) embeds coefficients fit on both days into trader.COEF.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import backtest  # noqa: E402
import baseline_trader as bl  # noqa: E402
import trader as tr  # noqa: E402

RIDGE = 1.0


class Logger(tr.Trader):
    def __init__(self):
        super().__init__()
        self.rows = []      # (mids, {sym: {h: feats}}, base_feats)
        self.base = bl.Trader()

    def run(self, state):
        mids, events = {}, {}
        for s in tr.LIMIT:
            d = state.order_depths.get(s)
            events[s] = tr.informed_event(state.market_trades.get(s, []), state.own_trades.get(s, []),
                                          self.prev_mid.get(s))
            mids[s] = (max(d.buy_orders) + min(d.sell_orders)) / 2.0 if d.buy_orders and d.sell_orders else None
            if mids[s] is not None:
                self.prev_mid[s] = mids[s]
        self.model.step(mids, events)
        feats = {s: ({h: self.model.features(s, h) for h in tr.HORIZONS} if mids[s] is not None and self.model.ready(s) else None)
                 for s in tr.LIMIT}
        # baseline feature engine (same bookkeeping as baseline_trader.run, no orders)
        b = self.base
        for s in bl.LIMIT:
            d = state.order_depths.get(s)
            b.events[s].append(b.informed_event(state, s))
            if d.buy_orders and d.sell_orders:
                b.mid[s] = mids[s]
            if s in b.mid:
                b.mids[s].append(b.mid[s])
            for key in ('events', 'mids'):
                lst = getattr(b, key)[s]
                if len(lst) > bl.HIST:
                    del lst[0]
        bf = {s: (b.features(s) if s in b.mid and mids[s] is not None else None) for s in bl.LIMIT}
        for s in bl.LIMIT:
            if mids[s] is not None:
                b.prev_mid[s] = mids[s]
        self.rows.append((mids, feats, bf))
        return {}, 0, ''


def collect(day):
    holder = {}

    class L(Logger):
        def __init__(self):
            super().__init__()
            holder['t'] = self

    backtest.run_day(L, day)
    return holder['t'].rows


def design(rows, s, h, base=False):
    X, y = [], []
    for i in range(len(rows) - h):
        m0, f0, b0 = rows[i]
        m1 = rows[i + h][0][s]
        if m0[s] is None or m1 is None:
            continue
        f = (b0[s] if base else (f0[s][h] if f0[s] else None))
        if f is None:
            continue
        X.append(([1.0] + f) if base else f)
        y.append(m1 - m0[s])
    return np.array(X), np.array(y)


def ridge(X, y, lam=RIDGE):
    reg = lam * np.eye(X.shape[1])
    reg[0, 0] = 0
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


def r2(X, y, w):
    e = y - X @ w
    return 1 - e.var() / y.var()


def main():
    days = {d: collect(d) for d in (-2, -1)}
    coef = {}
    for s in tr.LIMIT:
        coef[s] = {}
        print(f'{s}: out-of-sample R^2 (train one day, test other)   new vs baseline')
        for h in tr.HORIZONS:
            res = []
            for base in (False, True):
                sc = []
                for tr_d, te_d in ((-2, -1), (-1, -2)):
                    Xa, ya = design(days[tr_d], s, h, base)
                    Xb, yb = design(days[te_d], s, h, base)
                    sc.append(r2(Xb, yb, ridge(Xa, ya)))
                res.append(np.mean(sc))
            Xs, ys = zip(*(design(days[d], s, h) for d in days))
            w = ridge(np.vstack(Xs), np.concatenate(ys))
            coef[s][h] = [round(float(v), 4) for v in w]
            print(f'   h={h:>2d}  new {res[0]:.3f}   baseline-features {res[1]:.3f}')
    if '--write' in sys.argv:
        path = os.path.join(os.path.dirname(HERE), 'trader.py')
        src = open(path).read()
        start = src.index('COEF = {')
        end = src.index('\n', start)
        body = 'COEF = {\n' + ''.join(
            f"    '{s}': {{\n" + ''.join(f'        {h}: {coef[s][h]},\n' for h in tr.HORIZONS) + '    },\n'
            for s in tr.LIMIT) + '}'
        if src[start:end].strip() != 'COEF = {}':     # replace a previously written multi-line block
            end = src.index('\n}\n', start) + 2
        open(path, 'w').write(src[:start] + body + src[end:])
        print('wrote COEF into trader.py')


if __name__ == '__main__':
    main()
