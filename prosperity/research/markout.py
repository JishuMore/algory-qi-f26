"""Fill-level diagnostics: taker vs maker markouts and position usage (OOS coefs)."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import backtest, trader as tr
from evaluate import coefs_from
from common import load_day
both = lambda v: {'AMBER': v, 'JADE': v}
P = {'make_h': both(10), 'take_h': both(10), 'take_edge': both(1.0), 'make_edge': both(2.0), 'skew': both(0.02)}
for test, train in ((-1, -2),):
    tr.COEF.clear(); tr.COEF.update(coefs_from([train]))
    fills, poss = [], []
    class T(tr.Trader):
        def __init__(self):
            super().__init__()
            for k, v in P.items(): setattr(self, k, v)
            self.books = {}
        def run(self, state):
            for s, lst in state.own_trades.items():
                for t in lst:
                    b = self.books.get((t.timestamp, s))
                    taker = b is not None and ((t.buyer == 'SUBMISSION' and t.price in b.sell_orders) or (t.seller == 'SUBMISSION' and t.price in b.buy_orders))
                    fills.append((t.timestamp, s, 1 if t.buyer == 'SUBMISSION' else -1, t.price, t.quantity, taker))
            for s, d in state.order_depths.items():
                self.books[(state.timestamp, s)] = d
            poss.append(dict(state.position))
            return super().run(state)
    pnl = backtest.run_day(T, test)[0]
    df = load_day(test)
    F = pd.DataFrame(fills, columns=['ts', 's', 'side', 'px', 'q', 'taker'])
    for s in tr.LIMIT:
        m = df[s].ffill()
        f = F[F.s == s].copy()
        for h in (0, 1, 5, 10, 30, 100):
            fut = m.reindex(f.ts + 100 * h).values
            f[f'm{h}'] = f.side * (fut - f.px) * f.q
        g = f.groupby('taker')
        print(s, 'pnl', round(pnl[s]))
        print(pd.concat([g.q.sum().rename('qty'), g[[f'm{h}' for h in (0, 1, 5, 10, 30, 100)]].sum().round(0)], axis=1))
        p = np.array([x.get(s, 0) for x in poss])
        print('  |pos|>=45 frac', np.mean(np.abs(p) >= 45).round(3), ' mean|pos|', np.abs(p).mean().round(1))
