"""
data_loader.py  --  Day 2

Provides the clean baseline consumption data that everything else builds on.

WHY REAL-SHAPED DATA MATTERS (and why not just random numbers):
A detector tested on uniform random noise looks flawless and proves nothing.
Real household consumption has strong structure: an overnight trough, a
morning peak, a midday dip, an evening peak, and a different shape at
weekends. The Day 3 seasonal z-score depends on that structure existing --
if every hour looked the same, grouping by hour would be meaningless.

TWO DATA SOURCES ARE PROVIDED:

  generate_synthetic_data()  -- default. Calibrated to match the statistical
                                fingerprint of the UCI dataset below
                                (mean ~1.1 kW, std ~1.0 kW, same daily and
                                weekly cycles). Runs instantly, no download.

  load_uci_dataset()         -- the real thing. UCI "Individual Household
                                Electric Power Consumption": 2,075,259
                                minute-level readings from one French
                                household, 2006-2010.

Both return an identical schema, so swapping one for the other changes
nothing downstream. That is deliberate: develop fast on synthetic, then
validate on real data by changing one line in main.py.

WHY SYNTHETIC IS THE DEFAULT RATHER THAN THE REAL DATASET:
The UCI file is ~130 MB and needs either `ucimlrepo` or a manual download
and unzip. Making that a hard requirement means the repo does not run for
anyone who clones it without reading the README first. A project that runs
on `python main.py` immediately is worth far more in a portfolio than one
that fails on a missing dataset.

ALTERNATIVES CONSIDERED:
  - Commit a data sample to the repo: rejected, git is for source not data.
  - Pecan Street / UMass Smart* datasets: better (real smart meter, multiple
    homes) but require registration or academic access, so not reproducible
    for anyone reading the repo.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.logger_config import log_operational


def generate_synthetic_data(days: int = 90, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic hourly household consumption data.

    Parameters
    ----------
    days : int
        Days to simulate. Default 90 = 30 days baseline + 60 days test,
        which is enough room for all four attack windows plus gaps.
    seed : int
        Fixed seed so results are reproducible. This matters more than it
        looks: comparing two detectors is meaningless if the underlying
        data changes between runs.

    Returns
    -------
    DataFrame indexed by hourly timestamp with columns:
        true_power_kw      -- actual consumption (what a healthy meter reports)
        reported_power_kw  -- what the meter transmits (attacks modify this)
        is_attacked        -- ground truth label, False until injection
        attack_type        -- ground truth label, "none" until injection
    """
    log_operational("data_loader", "ok",
                    f"Generating {days} days of synthetic hourly data (seed={seed})")

    rng = np.random.default_rng(seed)
    n_hours = days * 24
    timestamps = pd.date_range("2024-01-01", periods=n_hours, freq="h")

    hour_of_day = timestamps.hour
    is_weekend = (timestamps.dayofweek >= 5).astype(float)

    # 24 hourly baseline values, calibrated against UCI's daily profile.
    # Values are kW of average draw during that hour.
    hourly_shape = np.array([
        0.30, 0.25, 0.22, 0.22, 0.25, 0.40,   # 00-05  asleep
        0.85, 2.00, 2.10, 1.40, 0.90, 0.85,   # 06-11  morning peak
        0.85, 0.80, 0.85, 0.90, 1.20, 1.80,   # 12-17  midday dip, ramp up
        2.20, 2.10, 1.70, 1.20, 0.80, 0.50,   # 18-23  evening peak
    ])

    base = hourly_shape[hour_of_day]

    # Weekends: lower overall (people away or routines shift) with variation
    base = np.maximum(0.1, base + is_weekend * rng.normal(-0.20, 0.05, n_hours))

    # Ordinary measurement and behavioural noise
    true_power = np.maximum(0.05, base + rng.normal(0, 0.15, n_hours))

    df = pd.DataFrame({
        "true_power_kw":     np.round(true_power, 4),
        "reported_power_kw": np.round(true_power, 4),
        "is_attacked":       False,
        "attack_type":       "none",
    }, index=timestamps)
    df.index.name = "timestamp"

    log_operational("data_loader", "ok",
                    f"Generated {len(df)} rows | mean={df.true_power_kw.mean():.3f} kW "
                    f"| std={df.true_power_kw.std():.3f} kW")
    return df


def load_uci_dataset() -> pd.DataFrame:
    """
    Load the real UCI household power dataset, resampled to hourly means.

    Requires: pip install ucimlrepo
    Returns the same four-column schema as generate_synthetic_data().

    WHY RESAMPLE MINUTE DATA TO HOURLY:
    Minute-level data is noisy enough that a kettle switching on looks like
    an anomaly. Hourly means smooth out appliance-level spikes while keeping
    the daily and weekly patterns an attacker would have to hide inside.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        log_operational("data_loader", "ok", "Fetching UCI dataset (~30s)")
        raw = fetch_ucirepo(id=235)
        df = raw.data.features.copy()

        df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"],
                                         format="%d/%m/%Y %H:%M:%S")
        df = df.set_index("timestamp").sort_index()
        df["Global_active_power"] = pd.to_numeric(df["Global_active_power"],
                                                  errors="coerce")
        df = df.dropna(subset=["Global_active_power"])

        hourly = df["Global_active_power"].resample("h").mean().dropna()
        result = pd.DataFrame({
            "true_power_kw":     hourly.values.round(4),
            "reported_power_kw": hourly.values.round(4),
            "is_attacked":       False,
            "attack_type":       "none",
        }, index=hourly.index)
        result.index.name = "timestamp"

        log_operational("data_loader", "ok",
                        f"UCI loaded: {len(result)} hourly rows | "
                        f"mean={result.true_power_kw.mean():.3f} kW")
        return result

    except ImportError:
        log_operational("data_loader", "error",
                        "ucimlrepo not installed. Run: pip install ucimlrepo")
        raise
    except Exception as exc:
        log_operational("data_loader", "error", f"UCI load failed: {exc}")
        raise
