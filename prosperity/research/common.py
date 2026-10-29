"""Shared loaders for research scripts."""
import numpy as np
import pandas as pd

MU = {'AMBER': 1000.0, 'JADE': 500.0}
DATA = __file__.rsplit('/', 2)[0] + '/data'


def load_day(d):
    """Per-tick frame: mids (NaN in blackout) and net signed qty-7 / other-qty flow per product.

    Event at tick t is the informed trade observed *in the book of t*; a live trader only
    learns of it at t+1 (market_trades), so features must be lagged by >= 1.
    """
    p = pd.read_csv(f'{DATA}/prices_day_{d}.csv', sep=';')
    t = pd.read_csv(f'{DATA}/trades_day_{d}.csv', sep=';')
    ts = np.sort(p.timestamp.unique())
    out = pd.DataFrame(index=ts)
    for s in MU:
        q = p[p['product'] == s].set_index('timestamp').reindex(ts)
        out[s] = (q.bid_price_1 + q.ask_price_1) / 2
        out[s + '_bid'] = q.bid_price_1
        out[s + '_ask'] = q.ask_price_1
        tt = t[t.symbol == s]
        m = out[s].reindex(tt.timestamp).values
        side = np.sign(tt.price.values - m)
        sq = pd.DataFrame({'ts': tt.timestamp.values, 'side': side, 'q': tt.quantity.values})
        inf = sq[sq.q == 7].groupby('ts').side.sum()
        oth = (sq[sq.q < 7].assign(v=lambda x: x.side * x.q)).groupby('ts').v.sum()
        out[s + '_ev'] = inf.reindex(ts).fillna(0).clip(-1, 1)
        out[s + '_flow'] = oth.reindex(ts).fillna(0)
    out['day'] = d
    return out


def load_all():
    return [load_day(-2), load_day(-1)]
