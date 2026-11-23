"""Homework 9 self-check.   python check_hw09.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _check import Checker
import numpy as np

c = Checker("Homework 9 self-check", "hw09_starter.py")
m = c.load(__file__)


def _var_sane(m):
    v = m.var(RET, 0.95)
    return v < np.mean(RET)

def _mc_sane(m):
    finals = np.asarray(m.simulate_paths(100_000.0, 0.08, 0.16, days=252, paths=1000, seed=1))
    return (len(finals) == 1000 and float(finals.min()) > 0
            and 50_000 < float(np.median(finals)) < 200_000)


rng = np.random.default_rng(9)
RET = rng.normal(0.0004, 0.011, 5000)

c.check("recovery from a 50% loss",
        lambda: m.recovery_needed(0.50), 1.00, tol=1e-9,
        note="you need to double to get back to even")
c.check("recovery from a 10% loss",
        lambda: m.recovery_needed(0.10), 1 / 9, tol=1e-9, note="about +11.1%")
c.check("recovery from a 75% loss",
        lambda: m.recovery_needed(0.75), 3.0, tol=1e-9, note="+300%")
c.check("recovery from no loss at all",
        lambda: m.recovery_needed(0.0), 0.0, tol=1e-12, note="nothing to recover")

c.assert_true("annualising multiplies daily volatility by root 252",
              lambda: abs(m.annualise_vol(0.01) - 0.01 * np.sqrt(252)) < 1e-9,
              note="the square-root-of-time rule")

c.assert_true("VaR is reported as a positive loss or a value below the start",
              lambda: _var_sane(m), note="a 95% VaR must sit in the left tail")
c.assert_true("CVaR is always worse than VaR",
              lambda: m.cvar(RET, 0.95) <= m.var(RET, 0.95) + 1e-9,
              note="CVaR averages the losses beyond the VaR threshold, so it cannot be milder")
c.assert_true("a higher confidence level gives a worse VaR",
              lambda: m.var(RET, 0.99) <= m.var(RET, 0.95) + 1e-9,
              note="99% reaches further into the tail than 95%")


c.assert_true("simulated paths start where you told them to",
              lambda: _mc_sane(m), note="a Monte Carlo of 1,000 paths should be well-behaved")


sys.exit(c.report())
