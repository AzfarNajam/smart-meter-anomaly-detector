# Smart Meter / Solar Inverter False Data Injection Detector

A Python anomaly detection system for identifying manipulated smart meter
and solar inverter data built around a real, current threat rather than
a textbook exercise.

## Why this project

Smart meters and solar inverters are increasingly recognized as a
significant blind spot in Europe's energy infrastructure security.
Researchers have shown that smart meter readings can be manipulated
through false data injection attacks, and EU regulators flagged solar
inverter supply chains as a high-risk dependency in 2026, following a real
attack on Poland's solar infrastructure in December 2025. Despite growing
regulatory attention (NIS2, the Cyber Resilience Act), accessible
open-source detection tooling for this specific attack class is rare,
most of what exists stays inside academic papers rather than usable
software.

This project builds a working, explainable false data injection detector
for time series energy data, combining statistical methods with machine
learning, and with investigation grade structured logging built in from
day one rather than bolted on at the end.

## Project status

Day 1 of 5 — project skeleton and structured logging framework complete
and tested. Detection logic, attack simulation, and ML comparison land
over the following days.

## Project structure

```
smart-meter-anomaly-detector/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   └── logger_config.py      # structured JSON logging framework
├── data/
│   └── README.md             # how to fetch the dataset (not committed)
├── logs/
│   └── README.md             # generated logs land here (not committed)
└── tests/
    └── test_logging.py       # smoke test for the logging framework
```

## Setup

```
pip install -r requirements.txt
python tests/test_logging.py
```

If the smoke test runs cleanly, you'll see three new files appear in
`logs/`, each containing structured JSON-line entries.

## Logging design

Three separate logs are used deliberately, rather than one combined file —
mirroring how real security monitoring and incident response systems are
built, so that an investigation can ask one specific question at a time:

1. **detection.log** — what was flagged, and why (meter ID, raw vs.
   expected reading, deviation score, suspected attack type, severity)
2. **operational.log** — is the detection pipeline itself healthy, or
   silently failing? (a detector that crashes quietly is worse than no
   detector at all)
3. **audit.log** — what configuration or threshold changed, when, and why
   (so tuning decisions are never untraceable later)

Every log entry is a single JSON object, so it can be parsed and queried
directly rather than scraped out of free-text sentences.

## Roadmap

- [x] Day 1 — project skeleton + structured logging framework
- [ ] Day 2 — baseline data + simulated attack injection (scaling, ramp,
      replay, noise)
- [ ] Day 3 — statistical detection engine (seasonal z-score + CUSUM)
- [ ] Day 4 — ML comparison layer (Isolation Forest) + precision/recall
      evaluation against ground truth
- [ ] Day 5 — documentation, visualizations, polish
