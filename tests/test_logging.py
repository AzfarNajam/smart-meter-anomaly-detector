"""
test_logging.py

Smoke test for the logging framework. Run this FIRST, before writing any
detection logic, to confirm all three logs write correctly.

Run it with:
    python tests/test_logging.py

(run from the project root folder, not from inside tests/)
"""

import sys
from pathlib import Path

# Allow this script to import from src/ when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logger_config import log_detection, log_operational, log_audit


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
        changed_by="azfar",
    )

    print("\nSmoke test complete. Check the logs/ folder for three new files:")
    print("  - logs/detection.log")
    print("  - logs/operational.log")
    print("  - logs/audit.log")
    print("\nEach line in those files should be a single JSON object.")


if __name__ == "__main__":
    run_smoke_test()
