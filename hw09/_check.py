"""Shared self-check harness for the Algory QI homeworks.

Students run `python check_hwNN.py`. Nothing here is a grade: the assignments
are not graded. This exists so you can find your own bug at 11pm instead of
waiting until office hours.

Only the parts of an assignment with one right answer are checked. Your ticker
choices, your plots, your strategy and your written answers are yours, and they
are where most of the learning is.
"""
import importlib.util, pathlib, sys, traceback

G, R, D, B, Y, OFF = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[33m", "\033[0m"


class Checker:
    def __init__(self, title, target_default):
        self.title = title
        self.target = sys.argv[1] if len(sys.argv) > 1 else target_default
        self.results = []
        self.mod = None

    # ---------- loading ----------
    def load(self, caller_file):
        path = pathlib.Path(caller_file).parent / self.target
        if not path.exists():
            print(f"{R}Cannot find {self.target}{OFF}")
            print(f"{D}Run this from the folder holding your solution, or pass the "
                  f"filename: python {pathlib.Path(caller_file).name} my_answers.py{OFF}")
            sys.exit(1)
        spec = importlib.util.spec_from_file_location("submission", path)
        self.mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(self.mod)
        except Exception:
            print(f"\n{R}Your file did not import, so nothing could be checked.{OFF}")
            print(f"{D}Fix this first. Everything below depends on it.{OFF}\n")
            traceback.print_exc()
            sys.exit(1)
        return self.mod

    # ---------- checks ----------
    def check(self, name, fn, expected, tol=0.01, note=""):
        """Compare a value against an expected one, within tolerance."""
        try:
            got = fn()
        except NotImplementedError:
            self.results.append((name, None, "not written yet", note)); return
        except AttributeError as e:
            self.results.append((name, None, f"missing: {e}", note)); return
        except Exception:
            last = traceback.format_exc().strip().split("\n")[-1]
            self.results.append((name, False, last, note)); return
        # numpy comparisons return np.bool_, which is not the Python True
        # singleton, so coerce or every passing check reads as a failure.
        try:
            ok = bool(abs(float(got) - float(expected)) <= tol)
            shown = f"got {float(got):.6g}, expected {float(expected):.6g}"
        except (TypeError, ValueError):
            ok = bool(got == expected)
            shown = f"got {got!r}, expected {expected!r}"
        self.results.append((name, ok, shown, note))

    def assert_true(self, name, fn, note=""):
        """Check a condition that is either satisfied or not."""
        try:
            ok = fn()
        except NotImplementedError:
            self.results.append((name, None, "not written yet", note)); return
        except AttributeError as e:
            self.results.append((name, None, f"missing: {e}", note)); return
        except Exception:
            last = traceback.format_exc().strip().split("\n")[-1]
            self.results.append((name, False, last, note)); return
        self.results.append((name, bool(ok), "condition not satisfied", note))

    def exists(self, name, attr, note=""):
        """Check that a required function is defined at all."""
        ok = hasattr(self.mod, attr) and callable(getattr(self.mod, attr))
        self.results.append((name, ok, f"no callable named {attr}", note))

    # ---------- reporting ----------
    def report(self):
        passed = [x for x in self.results if x[1] is True]
        failed = [x for x in self.results if x[1] is False]
        todo = [x for x in self.results if x[1] is None]

        print(f"\n{B}{self.title}{OFF}  {D}{self.target}{OFF}\n")
        for name, r, detail, note in self.results:
            mark = f"{G}pass{OFF}" if r is True else (f"{D}todo{OFF}" if r is None else f"{R}FAIL{OFF}")
            print(f"  {mark}  {name}")
            if r is False:
                print(f"        {R}{detail}{OFF}")
                if note:
                    print(f"        {D}{note}{OFF}")

        print(f"\n{B}{len(passed)} of {len(self.results)} checks pass{OFF}", end="")
        if todo:
            print(f", {len(todo)} not written yet", end="")
        if failed:
            print(f", {R}{len(failed)} failing{OFF}", end="")
        print()

        if failed:
            print(f"\n{Y}Work on the first failure. Later ones often follow from it.{OFF}\n")
        elif todo:
            print(f"\n{D}Keep going. Re-run this as you fill each function in.{OFF}\n")
        else:
            print(f"\n{G}Every checkable part is correct.{OFF}")
            print(f"{D}Your tickers, plots and written answers are not checked here,")
            print(f"and they are where most of the learning is.{OFF}\n")
        return 1 if failed else 0
