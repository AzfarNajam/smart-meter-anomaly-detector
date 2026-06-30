"""
logger_config.py

Structured JSON logging framework for the Smart Meter / Solar Inverter
False Data Injection Detector.

WHY THREE SEPARATE LOGS INSTEAD OF ONE:
This mirrors how real security monitoring / incident response systems are
built. A single combined log file is hard to investigate later because
"the system flagged something" and "the system itself broke" and "someone
changed a threshold" are three different kinds of events that need to be
queried separately during an investigation.

1. detection.log   -> WHAT happened (every anomaly a detector flagged)
2. operational.log -> Is the detector itself alive and working, or silently
                       failing? (a detector that crashes silently is worse
                       than no detector at all)
3. audit.log        -> WHO changed WHAT configuration/threshold, and WHY
                       (so a tuning change is never untraceable later)

Each log is written as JSON Lines (one JSON object per line) rather than
free-text sentences, so it can be parsed, filtered, or fed into a SIEM /
analysis tool later without writing regex to scrape a sentence.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class JsonLineFormatter(logging.Formatter):
    """Formats each log record as a single JSON line (JSON Lines format)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
        }
        # Structured loggers in this project pass a dict as the log message.
        if isinstance(record.msg, dict):
            payload.update(record.msg)
        else:
            payload["message"] = record.getMessage()
        return json.dumps(payload)


def _build_logger(name: str, filename: str) -> logging.Logger:
    """Builds a logger that writes JSON lines to its own dedicated file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # keep each log isolated, no cross-bleeding

    if not logger.handlers:  # prevents duplicate handlers on re-import
        file_handler = logging.FileHandler(LOG_DIR / filename, encoding="utf-8")
        file_handler.setFormatter(JsonLineFormatter())
        logger.addHandler(file_handler)

        # Echo to console too while developing, so you can see it live
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(console_handler)

    return logger


detection_logger = _build_logger("detection", "detection.log")
operational_logger = _build_logger("operational", "operational.log")
audit_logger = _build_logger("audit", "audit.log")


def log_detection(meter_id: str, raw_reading: float, expected_value: float,
                   deviation_score: float, detector_name: str,
                   suspected_attack_type: str, severity: str) -> None:
    """Log a flagged anomaly. Called by detection logic on Day 3+."""
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
    """Log the health of the pipeline itself (separate from what it detects)."""
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
