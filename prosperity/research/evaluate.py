"""Honest cross-day evaluation: trade day X with forecast coefficients fit only on day Y.

Usage: python research/evaluate.py            (grid over a few execution settings)
"""
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import backtest  # noqa: E402
import trader as tr  # noqa: E402
from fit import collect, design, event_cols, ridge  # noqa: E402

_ROWS = {}


def coefs_from(days, noev=False):
    for d in days:
        if d not in _ROWS:
            _ROWS[d] = collect(d)
    out = {}
    for s in tr.LIMIT:
        out[s] = {}
        for h in tr.HORIZONS:
            Xs, ys = zip(*(design(_ROWS[d], s, h) for d in days))
            X = np.vstack(Xs)
            if noev:
                X[:, event_cols(s)] = 0.0
            out[s][h] = list(ridge(X, np.concatenate(ys)))
    return out


def run(params, match='all', feed='normal'):
    """params: dict of Trader attribute overrides. Returns {day: pnl dict} trading OOS."""
    res = {}
    for test, train in ((-1, -2), (-2, -1)):
        tr.COEF.clear()
        tr.COEF.update(coefs_from([train]))
        tr.COEF_NOEV.clear()
        tr.COEF_NOEV.update(coefs_from([train], noev=True))

        class T(tr.Trader):
            def __init__(self):
                super().__init__()
                self.log = False
                for k, v in params.items():
                    setattr(self, k, dict(v) if isinstance(v, dict) else v)

        res[test] = backtest.run_day(T, test, match, feed=feed)[0]
    return res


def fmt(res):
    return '  '.join(f'd{d}: A {p["AMBER"]:7.0f} J {p["JADE"]:7.0f}' for d, p in res.items()) + \
        f'  | total {sum(sum(p.values()) for p in res.values()):8.0f}'


if __name__ == '__main__':
    both = lambda v: {'AMBER': v, 'JADE': v}  # noqa: E731
    print('default        ', fmt(run({})))
    for th, mh in itertools.product([10, 15, 20, 25], [5, 10]):
        print(f'take_h={th:>2d} make_h={mh:>2d}', fmt(run({'take_h': both(th), 'make_h': both(mh)})))
