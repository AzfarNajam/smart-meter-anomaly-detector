"""
test_pipeline.py

Smoke and regression tests. Not exhaustive unit tests, but they cover the
failures that actually happened during development, which is what a
regression test is for.

Run from the project root:
    python tests/test_pipeline.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import generate_synthetic_data
from src.attack_injector import inject_all_attacks
from src.detector import ZScoreDetector, CUSUMDetector, ReplayDetector
from src.ml_detector import IsolationForestDetector
from src.evaluate import confusion_counts, metrics_from_counts

results = []


def check(name, condition):
    ok = bool(condition)
    results.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")


def run():
    print("\nPipeline smoke tests\n" + "-" * 64)

    df = generate_synthetic_data(days=90)
    check("90 days hourly gives 2160 rows", len(df) == 2160)
    check("index is a DatetimeIndex", isinstance(df.index, pd.DatetimeIndex))
    check("no negative power readings", (df.true_power_kw >= 0).all())
    check("reported equals true before injection",
          np.allclose(df.true_power_kw, df.reported_power_kw))

    df = inject_all_attacks(df, baseline_hours=720)
    check("all four attack types present",
          set(df.attack_type.unique()) == {"none", "scaling", "ramp", "replay", "noise"})
    check("baseline period left clean", not df.iloc[:720].is_attacked.any())
    check("attack windows do not overlap (250 attacked)",
          int(df.is_attacked.sum()) == 250)

    baseline, test = df.iloc[:720], df.iloc[720:]

    # Regression test for the pandas merge bug: .merge() joins on columns and
    # silently replaces the DatetimeIndex with a RangeIndex. Everything
    # downstream is time-based, so this failure is silent but corrupting.
    z = ZScoreDetector().fit(baseline).score(test)
    check("z-score preserves DatetimeIndex after merge",
          isinstance(z.index, pd.DatetimeIndex))
    check("z-score false positive rate under 2%",
          z[z.attack_type == "none"].zscore_flagged.mean() < 0.02)

    # Regression tests for the two CUSUM failures during development:
    # seasonality on raw readings (31% FP) and accumulator latching (97% FP).
    c = CUSUMDetector().fit(baseline).score(test)
    check("CUSUM false positive rate under 1%",
          c[c.attack_type == "none"].cusum_flagged.mean() < 0.01)
    check("CUSUM catches most of the ramp attack",
          c[c.attack_type == "ramp"].cusum_flagged.mean() > 0.5)

    r = ReplayDetector().fit(baseline).score(test)
    check("replay rule has zero false positives",
          int(r[r.attack_type == "none"].replay_flagged.sum()) == 0)
    check("replay rule catches replay attack",
          r[r.attack_type == "replay"].replay_flagged.mean() > 0.3)

    i = IsolationForestDetector().fit(baseline).score(test)
    check("isolation forest produces flags", int(i.if_flagged.sum()) > 0)

    m = metrics_from_counts(confusion_counts([True, True, False, False],
                                             [True, False, True, False]))
    check("metrics maths correct (P=0.5, R=0.5)",
          m["precision"] == 0.5 and m["recall"] == 0.5)

    m0 = metrics_from_counts(confusion_counts([True, False], [False, False]))
    check("metrics handle divide by zero", m0["precision"] == 0.0)

    print("-" * 64)
    passed = sum(1 for _, ok in results if ok)
    print(f"{passed}/{len(results)} passed\n")
    return passed == len(results)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
