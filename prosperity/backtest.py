"""Local backtester for the AMBER/JADE round.

Matching model (same conventions as the community `prosperity backtester`):
  1. Orders are checked against position limits; if the worst case (all buys or all sells
     filling) would breach the limit, every order for that product is cancelled.
  2. Orders are matched against the visible book at the current tick (we take at book prices).
  3. Remaining orders rest for the tick and are matched against that tick's market trades:
     a trade at price p fills our buy at b if p <= b ('all') or p < b ('worse'), at our price,
     up to the trade's quantity. Consumed quantity is removed from the trade.
  4. Trades of tick t (what is left of them) are shown as market_trades at t+1; our fills as own_trades.
PnL is cash + position marked at the last valid mid of the day. Each day starts flat with a fresh Trader.

Usage: python backtest.py [module ...] [--match all|worse] [--days -2 -1]
"""
import argparse
import importlib
import os
import sys
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from datamodel import Order, OrderDepth, Trade, TradingState, Observation  # noqa: E402

LIMIT = {'AMBER': 50, 'JADE': 50}


def load(day):
    p = pd.read_csv(f'{HERE}/data/prices_day_{day}.csv', sep=';')
    t = pd.read_csv(f'{HERE}/data/trades_day_{day}.csv', sep=';')
    books = defaultdict(dict)
    for r in p.itertuples(index=False):
        d = OrderDepth()
        for i in (1, 2, 3):
            bp, bv = getattr(r, f'bid_price_{i}'), getattr(r, f'bid_volume_{i}')
            ap, av = getattr(r, f'ask_price_{i}'), getattr(r, f'ask_volume_{i}')
            if bp == bp:
                d.buy_orders[int(bp)] = int(bv)
            if ap == ap:
                d.sell_orders[int(ap)] = -int(av)
        books[r.timestamp][r.product] = d
    trades = defaultdict(lambda: defaultdict(list))
    for r in t.itertuples(index=False):
        trades[r.timestamp][r.symbol].append([int(r.price), int(r.quantity)])
    return books, trades


def run_day(trader_cls, day, match='all', verbose=False, feed='normal', fresh=False):
    """feed: 'normal' | 'none' (no market trades) | 'dup' (trades re-sent next tick too) | 'late' (one extra tick delay)
    | 'sticky' (last trades repeated until new ones print)."""
    books, trades = load(day)
    trader = trader_cls()
    pos = {s: 0 for s in LIMIT}
    cash = {s: 0.0 for s in LIMIT}
    last_mid = {}
    own_prev = {s: [] for s in LIMIT}
    mkt_prev = {s: [] for s in LIMIT}
    mkt_prev2 = {s: [] for s in LIMIT}
    sticky = {s: [] for s in LIMIT}
    trader_data = ''
    n_fills = defaultdict(int)
    vol = defaultdict(int)
    curve = []
    errors = 0
    for ts in sorted(books):
        depths = books[ts]
        shown = {'normal': mkt_prev, 'none': {s: [] for s in LIMIT}, 'late': mkt_prev2,
                 'dup': {s: mkt_prev2[s] + mkt_prev[s] for s in LIMIT}, 'sticky': sticky}[feed]
        state = TradingState(trader_data, ts, {}, {s: _copy(depths[s]) for s in depths},
                             {s: list(own_prev[s]) for s in LIMIT}, {s: list(shown[s]) for s in LIMIT},
                             dict(pos), Observation())
        if fresh:                      # engine that re-creates the Trader every tick (state only via traderData)
            trader = trader_cls()
        try:
            out = trader.run(state)
        except Exception as e:         # engines skip a tick whose run() raises
            errors += 1
            if errors == 1:
                print(f'  [run() raised at {day}/{ts}: {e!r}]')
            out = ({}, 0, trader_data)
        orders, trader_data = (out[0], out[2]) if isinstance(out, tuple) else (out, '')
        own_now = {s: [] for s in LIMIT}
        tick_trades = {s: [list(x) for x in trades[ts].get(s, [])] for s in LIMIT}
        for s, ol in (orders or {}).items():
            if s not in LIMIT or not ol:
                continue
            buys = sum(o.quantity for o in ol if o.quantity > 0)
            sells = -sum(o.quantity for o in ol if o.quantity < 0)
            if pos[s] + buys > LIMIT[s] or pos[s] - sells < -LIMIT[s]:
                if verbose:
                    print(f'{day} {ts} {s}: limit breach, orders cancelled')
                continue
            d = depths[s]
            for o in ol:
                q = o.quantity
                if q > 0:
                    for px in sorted(d.sell_orders):
                        if px > o.price or q == 0:
                            break
                        f = min(q, -d.sell_orders[px])
                        d.sell_orders[px] += f
                        if d.sell_orders[px] == 0:
                            del d.sell_orders[px]
                        q -= f; pos[s] += f; cash[s] -= f * px; own_now[s].append(Trade(s, px, f, 'SUBMISSION', '', ts))
                    for tr in tick_trades[s]:
                        if q == 0:
                            break
                        if tr[1] > 0 and (tr[0] <= o.price if match == 'all' else tr[0] < o.price):
                            f = min(q, tr[1])
                            tr[1] -= f; q -= f; pos[s] += f; cash[s] -= f * o.price
                            own_now[s].append(Trade(s, o.price, f, 'SUBMISSION', '', ts))
                elif q < 0:
                    q = -q
                    for px in sorted(d.buy_orders, reverse=True):
                        if px < o.price or q == 0:
                            break
                        f = min(q, d.buy_orders[px])
                        d.buy_orders[px] -= f
                        if d.buy_orders[px] == 0:
                            del d.buy_orders[px]
                        q -= f; pos[s] -= f; cash[s] += f * px; own_now[s].append(Trade(s, px, f, '', 'SUBMISSION', ts))
                    for tr in tick_trades[s]:
                        if q == 0:
                            break
                        if tr[1] > 0 and (tr[0] >= o.price if match == 'all' else tr[0] > o.price):
                            f = min(q, tr[1])
                            tr[1] -= f; q -= f; pos[s] -= f; cash[s] += f * o.price
                            own_now[s].append(Trade(s, o.price, f, '', 'SUBMISSION', ts))
            for t_ in own_now[s]:
                n_fills[s] += 1
                vol[s] += t_.quantity
        for s in LIMIT:
            own_prev[s] = own_now[s]
            mkt_prev2[s] = mkt_prev[s]
            mkt_prev[s] = [Trade(s, p, q, '', '', ts) for p, q in tick_trades[s] if q > 0]
            if mkt_prev[s]:
                sticky[s] = mkt_prev[s]
            d = books[ts][s]
            if d.buy_orders and d.sell_orders:
                last_mid[s] = (max(d.buy_orders) + min(d.sell_orders)) / 2
        curve.append(sum(cash[s] + pos[s] * last_mid.get(s, 0) for s in LIMIT))
    pnl = {s: cash[s] + pos[s] * last_mid[s] for s in LIMIT}
    return pnl, dict(vol), curve


def _copy(d):
    c = OrderDepth()
    c.buy_orders = dict(d.buy_orders)
    c.sell_orders = dict(d.sell_orders)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('modules', nargs='*', default=['trader'])
    ap.add_argument('--match', default='all', choices=['all', 'worse'])
    ap.add_argument('--days', nargs='*', type=int, default=[-2, -1])
    ap.add_argument('--fresh', action='store_true', help='new Trader instance every tick')
    ap.add_argument('--feed', default='normal', choices=['normal', 'none', 'dup', 'late', 'sticky'])
    a = ap.parse_args()
    for m in a.modules:
        cls = importlib.import_module(m).Trader
        tot = 0
        for day in a.days:
            pnl, vol, curve = run_day(cls, day, a.match, feed=a.feed, fresh=a.fresh)
            s = sum(pnl.values()); tot += s
            print(f'{m:>18s} day {day:>2d} [{a.match}]  ' + '  '.join(f'{k} {v:9.0f} (vol {vol.get(k, 0)})' for k, v in pnl.items()) + f'  total {s:9.0f}')
        print(f'{m:>18s} TOTAL {tot:.0f}')


if __name__ == '__main__':
    main()
