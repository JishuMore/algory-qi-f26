"""Test candidate features: out-of-sample R^2 gain at several horizons (pandas, causal constructions)."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from fit import collect, ridge, r2
import trader as tr
from common import DATA
days = {d: collect(d) for d in (-2, -1)}
vol = {}
for d in days:
    p = pd.read_csv(f'{DATA}/prices_day_{d}.csv', sep=';')
    vol[d] = {s: p[p['product'] == s].set_index('timestamp') for s in tr.LIMIT}
def build(d, s, h, extra):
    rows = days[d]; X, y = [], []
    ts = sorted(vol[d][s].index)
    valid_hist = []
    for i in range(len(rows) - h):
        m0, f0, _ = rows[i]; m1 = rows[i + h][0][s]
        if m0[s] is not None: valid_hist.append(m0[s])
        if m0[s] is None or m1 is None or f0[s] is None: continue
        f = list(f0[s][h]); q = vol[d][s].loc[ts[i]]
        if 'lags' in extra:
            f += [valid_hist[-1] - valid_hist[-1 - k] if len(valid_hist) > k else 0 for k in (34, 55)]
        if 'imb' in extra:
            f += [(q.bid_volume_1 - q.ask_volume_1) / (q.bid_volume_1 + q.ask_volume_1),
                  (q.bid_volume_2 - q.ask_volume_2) / (q.bid_volume_2 + q.ask_volume_2)]
        if 'spr' in extra:
            f += [q.ask_price_1 - q.bid_price_1 - (6 if s == 'AMBER' else 4),
                  (q.ask_price_2 + q.bid_price_2) / 2 - m0[s]]
        if 'cap' in extra:
            nb = len(tr.AGE); off = 3 if s == 'AMBER' else 5
            f += [float(np.sign(sum(f[off:off + 8])))]      # sign of active (age<=30) pulses
        X.append(f); y.append(m1 - m0[s])
    return np.array(X), np.array(y)
def oos(s, h, extra):
    sc = []
    for a, b in ((-2, -1), (-1, -2)):
        Xa, ya = build(a, s, h, extra); Xb, yb = build(b, s, h, extra); sc.append(r2(Xb, yb, ridge(Xa, ya)))
    return np.mean(sc)
for s in tr.LIMIT:
    for h in (5, 10, 20):
        base = oos(s, h, ())
        print(s, h, 'base', round(base, 4), {e: round(oos(s, h, (e,)) - base, 4) for e in ('lags', 'imb', 'spr', 'cap')})
