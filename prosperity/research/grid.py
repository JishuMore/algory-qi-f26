import itertools, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import backtest, baseline_trader as bl
from evaluate import run, fmt, _ROWS, coefs_from
from fit import design, ridge
both = lambda v: {'AMBER': v, 'JADE': v}

# baseline with honest (other-day) coefficients
coefs_from([-2, -1])
saved = dict(bl.COEF); res = {}
for test, train in ((-1, -2), (-2, -1)):
    for s in bl.LIMIT:
        for h in (3, 5, 7, 10, 15, 25, 40):
            if h not in (3,5,7,10,15,25,40): continue
            X, y = design(_ROWS[train], s, h, base=True) if h in __import__('trader').HORIZONS else (None, None)
            if X is None: continue
            bl.COEF[f'{s}_{h}'] = list(ridge(X[:, 1:], y, 1.0) if False else np.linalg.lstsq(X[:, 1:], y, rcond=None)[0])
    res[test] = backtest.run_day(bl.Trader, test)[0]
bl.COEF.clear(); bl.COEF.update(saved)
print('BASELINE (honest coefs)', fmt(res))

out = []
for h, te, me, sk in itertools.product([7, 10, 12], [0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [0.02, 0.05]):
    r = run({'take_h': both(h), 'make_h': both(h), 'take_edge': both(te), 'make_edge': both(me), 'skew': both(sk)})
    out.append((sum(sum(p.values()) for p in r.values()), h, te, me, sk, r))
out.sort(key=lambda x: -x[0])
for tot, h, te, me, sk, r in out[:12]:
    print(f'h={h:>2d} te={te} me={me} skew={sk}', fmt(r))
