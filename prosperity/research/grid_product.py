"""Per-product execution grid (products are independent in the simulator, so pair configs and read each leg)."""
import itertools, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from evaluate import run
A = list(itertools.product([10, 12, 15], [0.03, 0.05, 0.08], [0.5, 1.0, 1.5]))
J = list(itertools.product([7, 10], [0.0, 0.01, 0.02, 0.03], [0.5, 1.0, 1.5]))
J += J[:len(A) - len(J)]
ra, rj = [], []
for (ha, sa, ta), (hj, sj, tj) in zip(A, J):
    r = run({'take_h': {'AMBER': ha, 'JADE': hj}, 'make_h': {'AMBER': ha, 'JADE': hj},
             'skew': {'AMBER': sa, 'JADE': sj}, 'take_edge': {'AMBER': ta, 'JADE': tj},
             'make_edge': {'AMBER': 2.0, 'JADE': 2.0}})
    ra.append((r[-1]['AMBER'] + r[-2]['AMBER'], ha, sa, ta, round(r[-1]['AMBER']), round(r[-2]['AMBER'])))
    rj.append((r[-1]['JADE'] + r[-2]['JADE'], hj, sj, tj, round(r[-1]['JADE']), round(r[-2]['JADE'])))
for name, rr in (('AMBER', ra), ('JADE', rj)):
    print(name, '(total, h, skew, take_edge, day-1, day-2)')
    for x in sorted(set(rr), key=lambda x: -x[0])[:8]:
        print('  ', x)
