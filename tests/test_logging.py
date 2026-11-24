"""
test_logging.py

Smoke test for the logging framework. Run this FIRST, before writing any
detection logic, to confirm all three logs write correctly.

Run it with:
    python tests/test_logging.py

(run from the project root folder, not from inside tests/)
"""

import json
import sys
from pathlib import Path

# Allow this script to import from src/ when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import generate_synthetic_data
from src.detector import compute_seasonal_baselines
from src.logger_config import LOG_DIR, log_detection, log_operational, log_audit


def run_smoke_test():
    print("Running logging framework smoke test...\n")

    # Simulate the pipeline starting up successfully
    log_operational("data_ingestion", "ok", "Pipeline started successfully")

    # Simulate a flagged anomaly — this is the exact call Day 3's real
    # detection logic will make when it finds something suspicious
    log_detection(
        meter_id="METER_001",
        raw_reading=12.4,
        expected_value=3.1,
        deviation_score=8.7,
        detector_name="zscore_v1",
        suspected_attack_type="scaling_attack",
        severity="high",
    )

    # Simulate the pipeline hitting a problem (so you can see error-level
    # entries look different from normal ones)
    log_operational("data_ingestion", "error", "Lost connection to data source")

    # Simulate a config change — this is what Day 3+ threshold tuning will log
    log_audit(
        change_type="threshold_update",
        old_value=3.0,
        new_value=3.5,
        reason="reducing false positives during testing",
        changed_by="smoke-test",
    )

    baseline = generate_synthetic_data(days=30)
    compute_seasonal_baselines(baseline)
    try:
        compute_seasonal_baselines(baseline.iloc[:0])
    except ValueError:
        pass
    else:
        raise AssertionError("An empty seasonal baseline should be rejected")

    rows = {}
    for category in ("detection", "operational", "audit"):
        path = LOG_DIR / f"{category}.log"
        rows[category] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert rows[category] and all({"timestamp", "level", "event"} <= row.keys()
                                      for row in rows[category])
    assert any(row["event"] == "anomaly_detected" for row in rows["detection"])
    assert any(row["event"] == "config_change" for row in rows["audit"])
    assert any(row["level"] == "ERROR" for row in rows["operational"])
    assert any(row["level"] == "WARNING" and "smallest group" in row["message"]
               for row in rows["operational"])
    print("Structured logging and seasonal baseline checks passed.")


if __name__ == "__main__":
    run_smoke_test()
