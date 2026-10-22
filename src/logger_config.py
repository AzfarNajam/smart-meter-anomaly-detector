"""
logger_config.py

Structured JSON logging framework for the Smart Meter False Data Injection
Detector.

WHY FOUR SEPARATE LOGS INSTEAD OF ONE FILE:
This mirrors how real security monitoring and incident response systems are
built. A single combined log is hard to investigate later, because
"something was flagged", "the system itself broke", "someone changed a
threshold" and "here is the truth we are measuring against" are four
different questions. Splitting them means each question can be answered
without filtering noise from the other three.

  1. detection.log    -> WHAT was flagged (every anomaly, with its score)
  2. operational.log  -> IS THE DETECTOR ALIVE (pipeline health)
  3. audit.log        -> WHAT CHANGED (thresholds, config, and why)
  4. ground_truth.log -> WHAT WE ACTUALLY INJECTED (labels for Day 4 scoring)

WHY JSON LINES (one JSON object per line) INSTEAD OF PLAIN TEXT:
Plain text logs like "WARNING: meter 001 looks weird" require regex to
parse later. JSON lines can be loaded directly into pandas, a SIEM, or
Elasticsearch with no parsing code. The rule of thumb from production ops:
log for the machine that will read it in six months, not for the human
reading it right now.

ALTERNATIVES CONSIDERED:
  - `structlog` / `loguru`: nicer APIs, but add a dependency for something
    the standard library already does well. Kept stdlib so the project runs
    with zero installs on Day 1.
  - Single log with a "category" field: simpler, but then every query needs
    a filter, and a crash in the detector buries itself among thousands of
    detection rows.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class JsonLineFormatter(logging.Formatter):
    """Formats each log record as a single JSON object on one line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
        }
        if isinstance(record.msg, dict):
            payload.update(record.msg)
        else:
            payload["message"] = record.getMessage()
        return json.dumps(payload)


def _build_logger(name: str, filename: str, console: bool = False) -> logging.Logger:
    """
    Build a logger writing JSON lines to its own dedicated file.

    console=False by default because the detection log can produce hundreds
    of entries in one run; echoing all of them to the terminal buries the
    actual pipeline output. Operational logs echo to console since those are
    the ones you want to watch live.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # keep logs isolated, no cross-bleeding

    if not logger.handlers:  # prevents duplicate handlers on re-import
        fh = logging.FileHandler(LOG_DIR / filename, encoding="utf-8", mode="w")
        fh.setFormatter(JsonLineFormatter())
        logger.addHandler(fh)

        if console:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("  [%(name)s] %(message)s"))
            logger.addHandler(ch)

    return logger


detection_logger    = _build_logger("detection",    "detection.log")
operational_logger  = _build_logger("operational",  "operational.log", console=True)
audit_logger        = _build_logger("audit",        "audit.log")
ground_truth_logger = _build_logger("ground_truth", "ground_truth.log")


def log_detection(meter_id: str, raw_reading: float, expected_value: float,
                  deviation_score: float, detector_name: str,
                  suspected_attack_type: str, severity: str) -> None:
    """Log a single flagged anomaly."""
    detection_logger.info({
        "event": "anomaly_detected",
        "meter_id": meter_id,
        "raw_reading": raw_reading,
        "expected_value": expected_value,
        "deviation_score": deviation_score,
        "detector": detector_name,
        "suspected_attack_type": suspected_attack_type,
        "severity": severity,
    })


def log_operational(component: str, status: str, message: str = "") -> None:
    """Log pipeline health, separate from what the pipeline detects."""
    level = logging.INFO if status == "ok" else logging.ERROR
    operational_logger.log(level, {
        "event": "operational_status",
        "component": component,
        "status": status,
        "message": message,
    })


def log_audit(change_type: str, old_value, new_value, reason: str,
              changed_by: str = "system") -> None:
    """Log any change to detector configuration or thresholds."""
    audit_logger.info({
        "event": "config_change",
        "change_type": change_type,
        "old_value": old_value,
        "new_value": new_value,
        "reason": reason,
        "changed_by": changed_by,
    })


def log_ground_truth(attack_type: str, start_idx: int, end_idx: int,
                     params: dict, n_readings: int) -> None:
    """Log an injected attack so Day 4 can score detectors against truth."""
    ground_truth_logger.info({
        "event": "attack_injected",
        "attack_type": attack_type,
        "window_start_index": start_idx,
        "window_end_index": end_idx,
        "n_readings_affected": n_readings,
        "parameters": params,
    })
