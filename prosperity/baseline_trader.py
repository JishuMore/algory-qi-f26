import math
from datamodel import OrderDepth, Order


LIMIT = {'AMBER': 50, 'JADE': 50}  
MU = {'AMBER': 1000.0, 'JADE': 500.0}
JADE_LAG = 25
INFORMED_QTY = 7
AGE = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 45), (46, 70)]
LAGS = [1, 2, 3, 5, 8, 13, 21]
HIST = 72

COEF = {
    'AMBER_3': [-0.0082, 0.5063, 0.7049, 0.9211, 1.1155, 1.6814, 1.759, 2.1477, 2.4854, 0.3015, -0.0717, -0.1707, -0.0606, -0.0127, -0.0231, -0.0178, -0.0314, -0.0978],
    'AMBER_5': [-0.0121, 0.8258, 1.1199, 1.6332, 1.7435, 2.6406, 2.8529, 3.6572, 3.0023, 0.3808, -0.0848, -0.1712, -0.0491, -0.0297, -0.024, -0.0274, -0.0526, -0.1406],
    'AMBER_7': [-0.0163, 1.1866, 1.7618, 2.0907, 2.5784, 3.4908, 3.9434, 4.9513, 3.1666, 0.395, -0.0639, -0.1685, -0.0579, -0.0275, -0.0269, -0.0426, -0.0646, -0.1743],
    'AMBER_10': [-0.0223, 2.3367, 2.4936, 3.004, 3.6667, 4.8594, 5.9105, 5.6293, 3.3408, 0.3299, -0.0539, -0.1449, -0.0601, -0.0318, -0.0404, -0.0532, -0.1007, -0.1944],
    'AMBER_15': [-0.034, 3.9314, 4.1608, 4.6785, 5.6044, 7.6288, 7.8115, 5.4378, 3.1403, 0.1794, -0.0655, -0.1216, -0.0729, -0.0427, -0.0468, -0.0752, -0.1723, -0.1758],
    'AMBER_25': [-0.0711, 7.4224, 8.0907, 9.774, 9.6633, 8.9478, 7.3706, 3.8584, 1.7302, 0.0202, -0.1346, -0.1473, -0.0799, -0.0752, -0.1045, -0.1249, -0.1354, -0.1199],
    'AMBER_40': [-0.1762, 9.6485, 9.4796, 9.1635, 8.7487, 7.9481, 6.6388, 3.1629, 1.3447, -0.217, 0.0851, -0.1655, -0.0893, -0.07, -0.0814, -0.0872, -0.1032, -0.1077],
    'JADE_3': [-0.0025, 0.0347, 0.4388, 0.3588, 0.5912, 0.5462, 0.8559, 1.0656, 1.1829, 1.3465, 1.6328, 0.1631, -0.0204, -0.0208, -0.0273, -0.022, 0.0837, -0.0174, -0.0153, -0.2393, 0.1341, 0.1232, 0.013, -0.1813, -0.0499, -0.0215, -0.0216, -0.0047, -0.0259, -0.0799, 0.0036, -0.0023, 0.0135, -0.0048, 0.0145, 0.0021, -0.2554],
    'JADE_5': [-0.005, 0.0461, 0.6005, 0.645, 0.6929, 1.0417, 1.2651, 1.5699, 1.8054, 2.399, 1.8488, 0.1902, -0.0327, -0.0642, 0.0046, 0.0507, 0.0505, -0.0106, -0.2272, -0.2211, 0.1702, 0.0683, 0.0411, -0.1568, -0.0719, -0.0224, -0.0209, -0.0074, -0.0396, -0.1143, 0.0105, 0.0008, -0.0015, 0.0085, 0.0136, -0.0417, -0.3007],
    'JADE_7': [-0.0084, 0.0537, 0.5811, 0.6916, 1.1031, 1.3047, 1.6706, 2.0993, 2.3661, 3.3145, 1.981, 0.2, -0.0485, -0.0721, 0.0117, 0.1424, -0.0071, 0.0624, -0.535, -0.2276, 0.1267, -0.0145, 0.0864, -0.1451, -0.0557, -0.0205, -0.0228, -0.0195, -0.0538, -0.1417, -0.004, -0.0024, 0.0129, 0.0091, 0.0067, -0.1259, -0.2031],
    'JADE_10': [-0.0101, 0.0702, 0.5553, 1.4241, 1.5527, 1.8574, 2.2521, 2.867, 3.5547, 3.652, 2.0843, 0.2501, -0.0457, -0.0019, 0.1155, 0.0714, -0.0276, -0.0251, -0.4702, -0.1496, 0.0121, -0.0038, 0.0496, -0.1267, -0.0684, -0.0139, -0.0284, -0.0305, -0.0818, -0.1616, 0.0065, 0.0051, 0.0107, 0.0141, -0.0008, -0.2677, -0.0586],
    'JADE_15': [-0.0123, 0.0973, 0.4994, 2.1536, 2.3463, 2.7471, 3.3685, 4.4318, 4.6818, 3.299, 1.9154, 0.2194, -0.0569, -0.122, 0.0294, 0.076, -0.0723, -0.3511, 0.0133, -0.1023, 0.0155, 0.0446, -0.0103, -0.1092, -0.066, -0.027, -0.038, -0.0686, -0.1222, -0.1431, 0.0163, 0.0102, 0.003, -0.0023, -0.2049, -0.1401, 0.0452],
    'JADE_25': [-0.0145, 0.1541, 0.4657, 3.7798, 4.2275, 5.3329, 5.3136, 4.9823, 4.2378, 2.4933, 1.3887, 0.2123, -0.1861, -0.1043, 0.0301, 0.0429, 0.0451, 0.027, 0.1794, 0.0273, -0.0311, -0.0913, -0.0023, -0.1069, -0.0646, -0.0465, -0.0881, -0.0907, -0.104, -0.1115, -0.0694, -0.0168, 0.0079, 0.0312, 0.0369, 0.0446, 0.0341],
    'JADE_40': [-0.0494, 0.2191, 0.4352, 4.6745, 4.6887, 4.4979, 4.377, 4.1583, 3.4694, 1.9307, 0.9247, -0.1543, -0.3563, 1.9141, 2.1294, 2.3931, 2.8569, 3.8903, 4.0863, 2.7667, 1.4101, 0.1137, 0.0465, -0.102, -0.0793, -0.0747, -0.0575, -0.0714, -0.0909, -0.0818, -0.0357, -0.0209, 0.0056, -0.0033, -0.0023, -0.0554, -0.0317]
}


class Trader:
    def __init__(self):
        self.horizon = 10
        self.take_edge = {'AMBER': 1.0, 'JADE': 1.0}  
        self.make_edge = {'AMBER': 3.0, 'JADE': 3.0}   
        self.skew = {'AMBER': 0.02, 'JADE': 0.02}    
        self.pred_scale = {'AMBER': 1.0, 'JADE': 1.0}
        self.blackout_quotes = True                    
        self.blackout_edge = 6
        self.blackout_size = 10
        self.mids = {'AMBER': [], 'JADE': []}          
        self.events = {'AMBER': [], 'JADE': []}        
        self.mid = {}
        self.prev_mid = {}

    # ------------------------------------------------------------ signals
    def informed_event(self, state, sym):
        """+1/-1 if this tick's reported trades contain >= 7 lots of bot buying/selling (informed bot)."""
        ref = self.prev_mid.get(sym)
        if ref is None:
            return 0
        buy_q = 0
        sell_q = 0
        for t in state.market_trades.get(sym, []):
            if t.price >= ref:
                buy_q += t.quantity
            else:
                sell_q += t.quantity
        for t in state.own_trades.get(sym, []):
            if t.seller == 'SUBMISSION' and t.price > ref:
                buy_q += t.quantity
            elif t.buyer == 'SUBMISSION' and t.price < ref:
                sell_q += t.quantity
        return (1 if buy_q >= INFORMED_QTY else 0) - (1 if sell_q >= INFORMED_QTY else 0)

    def lagdiff(self, sym, k):
        h = self.mids[sym]
        return h[-1] - h[-1 - k] if len(h) > k else 0.0

    def buckets(self, sym):
        ev = self.events[sym]
        n = len(ev)
        out = []
        for a, b in AGE:
            s = 0
            for k in range(a, b + 1):
                if k <= n:
                    s += ev[-k]
            out.append(s)
        return out

    def features(self, sym):
        if sym == 'AMBER':
            f = [self.mid['AMBER'] - MU['AMBER']]
            f += self.buckets('AMBER')
            f += [self.lagdiff('AMBER', k) for k in LAGS]
            return f
        a = self.mids['AMBER']
        devj = self.mid['JADE'] - MU['JADE']
        if len(a) > JADE_LAG:
            a_lag = a[-1 - JADE_LAG] - MU['AMBER']
            a_now = a[-1] - MU['AMBER']
            gap = 0.5 * a_lag - devj
            lead = 0.5 * (a_now - a_lag)
        else:
            gap = 0.0
            lead = 0.0
        f = [devj, gap, lead]
        f += self.buckets('JADE') + self.buckets('AMBER')
        f += [self.lagdiff('JADE', k) for k in LAGS] + [self.lagdiff('AMBER', k) for k in LAGS]
        return f

    def predicted_move(self, sym):
        c = COEF[f'{sym}_{self.horizon}']
        return sum(x * w for x, w in zip(self.features(sym), c))


    def run(self, state):
        result = {}
        for sym in LIMIT:
            d = state.order_depths.get(sym)
            self.events[sym].append(self.informed_event(state, sym))
            if d and d.buy_orders and d.sell_orders:
                self.mid[sym] = (max(d.buy_orders) + min(d.sell_orders)) / 2.0
            if sym in self.mid:
                self.mids[sym].append(self.mid[sym])
            for key in ('events', 'mids'):
                lst = getattr(self, key)[sym]
                if len(lst) > HIST:
                    del lst[0]

        for sym in LIMIT:
            if sym not in self.mid:
                continue
            d = state.order_depths.get(sym) or OrderDepth()
            pos = state.position.get(sym, 0)
            value = self.mid[sym] + self.pred_scale[sym] * self.predicted_move(sym)
            res = value - self.skew[sym] * pos
            result[sym] = self.orders_for(sym, d, pos, LIMIT[sym], res)
            if d.buy_orders and d.sell_orders:
                self.prev_mid[sym] = self.mid[sym]
        return result, 0, ''

    def orders_for(self, sym, d, pos, lim, res):
        orders = []
        buy_cap = lim - pos
        sell_cap = lim + pos
        te = self.take_edge[sym]
        me = self.make_edge[sym]

        if not d.buy_orders or not d.sell_orders:
            if self.blackout_quotes:
                bq = min(self.blackout_size, buy_cap)
                sq = min(self.blackout_size, sell_cap)
                if bq > 0:
                    orders.append(Order(sym, int(math.floor(res - self.blackout_edge)), bq))
                if sq > 0:
                    orders.append(Order(sym, int(math.ceil(res + self.blackout_edge)), -sq))
            return orders

        for price in sorted(d.sell_orders):
            if price > res - te or buy_cap <= 0:
                break
            q = min(-d.sell_orders[price], buy_cap)
            orders.append(Order(sym, price, q))
            buy_cap -= q
        for price in sorted(d.buy_orders, reverse=True):
            if price < res + te or sell_cap <= 0:
                break
            q = min(d.buy_orders[price], sell_cap)
            orders.append(Order(sym, price, -q))
            sell_cap -= q

        best_bid = max(d.buy_orders)
        best_ask = min(d.sell_orders)
        bid = min(best_bid + 1, int(math.floor(res - me)))
        ask = max(best_ask - 1, int(math.ceil(res + me)))
        if bid >= ask:
            bid = ask - 1
        if buy_cap > 0 and bid < best_ask:
            orders.append(Order(sym, bid, buy_cap))
        if sell_cap > 0 and ask > best_bid:
            orders.append(Order(sym, ask, -sell_cap))
        return orders
