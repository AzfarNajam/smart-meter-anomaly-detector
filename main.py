"""
main.py -- end-to-end pipeline for the Smart Meter FDI Detector.

Run with:
    python main.py

Produces:
    logs/      four structured JSON logs
    outputs/   charts and a results table

PIPELINE STAGES
    1. Load clean baseline consumption data
    2. Inject four classes of false data injection attack (ground truth logged)
    3. Split into baseline (detectors learn) and test (detectors are measured)
    4. Run four detectors
    5. Score every detector against ground truth
    6. Render charts

WHY THE BASELINE/TEST SPLIT IS STRICTLY CHRONOLOGICAL:
Detectors are fitted only on the first 30 days and evaluated only on the
following 60. A random shuffle would let the model see future readings during
fitting, which inflates every metric and is meaningless for a system that has
to run forward in time. Time series data must be split by time.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.logger_config import log_operational
from src.data_loader import generate_synthetic_data
from src.attack_injector import inject_all_attacks
from src.detector import ZScoreDetector, CUSUMDetector, ReplayDetector
from src.ml_detector import IsolationForestDetector
from src.evaluate import evaluate_all, per_attack_recall, build_ensemble
from src import visualize

BASELINE_HOURS = 720   # 30 days used for learning normal behaviour
TOTAL_DAYS     = 90    # 30 baseline + 60 test


def main():
    print("=" * 78)
    print("SMART METER FALSE DATA INJECTION DETECTOR")
    print("=" * 78)

    # ---- 1 & 2: data and attacks -------------------------------------------
    print("\n[1/6] Generating baseline consumption data")
    df = generate_synthetic_data(days=TOTAL_DAYS)

    print("\n[2/6] Injecting attacks")
    df = inject_all_attacks(df, baseline_hours=BASELINE_HOURS)

    # ---- 3: chronological split --------------------------------------------
    print("\n[3/6] Splitting baseline / test")
    baseline = df.iloc[:BASELINE_HOURS]
    test     = df.iloc[BASELINE_HOURS:]
    log_operational("pipeline", "ok",
                    f"Baseline {len(baseline)} readings (clean), "
                    f"test {len(test)} readings "
                    f"({int(test.is_attacked.sum())} attacked)")

    # ---- 4: detectors ------------------------------------------------------
    print("\n[4/6] Running detectors")
    zs = ZScoreDetector(threshold=3.0).fit(baseline)
    cs = CUSUMDetector().fit(baseline)
    rp = ReplayDetector(lag_hours=24).fit(baseline)
    isof = IsolationForestDetector(contamination=0.05).fit(baseline)

    # Each detector returns its own frame; merge the flag columns onto `test`
    res = test.copy()
    z_out = zs.score(test)
    c_out = cs.score(test)
    r_out = rp.score(test)
    i_out = isof.score(test)

    for src_df, cols in [(z_out, ["zscore", "zscore_flagged"]),
                         (c_out, ["residual", "cusum_pos", "cusum_neg", "cusum_flagged"]),
                         (r_out, ["replay_match", "replay_flagged"]),
                         (i_out, ["if_score", "if_flagged"])]:
        for c in cols:
            res[c] = src_df[c].values

    detectors = {
        "Z-Score (seasonal)":  "zscore_flagged",
        "CUSUM (residual)":    "cusum_flagged",
        "Replay rule (24h)":   "replay_flagged",
        "Isolation Forest":    "if_flagged",
    }

    # Ensemble: flag if any detector fires
    res["ensemble_flagged"] = build_ensemble(res, detectors, strategy="union")
    detectors_with_ens = {**detectors, "Ensemble (union)": "ensemble_flagged"}

    # ---- 5: evaluation -----------------------------------------------------
    print("\n[5/6] Evaluating against ground truth")
    results = evaluate_all(res, detectors_with_ens)
    per_atk = per_attack_recall(res, detectors_with_ens)

    print("\n" + "-" * 78)
    print("OVERALL PERFORMANCE (sorted by F1)")
    print("-" * 78)
    print(results.to_string())

    print("\n" + "-" * 78)
    print("RECALL BY ATTACK TYPE  (1.00 = every attacked reading caught)")
    print("-" * 78)
    print(per_atk.to_string())

    # ---- 6: charts ---------------------------------------------------------
    print("\n[6/6] Rendering charts")
    visualize.plot_overview(res)
    for atk in ["scaling", "ramp", "replay", "noise"]:
        visualize.plot_attack_zoom(res, atk, detectors)
    visualize.plot_metrics_comparison(results)
    visualize.plot_per_attack_heatmap(per_atk)
    visualize.plot_cusum_trace(res, cs.h)

    out = Path(__file__).resolve().parent / "outputs"
    results.to_csv(out / "results_overall.csv")
    per_atk.to_csv(out / "results_per_attack.csv")

    print("\n" + "=" * 78)
    print("DONE")
    print(f"  logs/     4 structured JSON logs")
    print(f"  outputs/  8 charts + 2 CSV result tables")
    print("=" * 78)
    return res, results, per_atk


if __name__ == "__main__":
    main()
