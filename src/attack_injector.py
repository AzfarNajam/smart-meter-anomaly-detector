"""
attack_injector.py  --  Day 2

Injects four classes of False Data Injection (FDI) attack into clean data,
and records every injection to ground_truth.log so Day 4 can measure how
well each detector actually performed.

WHY BUILD AN INJECTOR INSTEAD OF USING A LABELLED ATTACK DATASET:
Public labelled FDI datasets almost all target transmission-grid state
estimation -- they assume you have a full power-flow model of a network
(IEEE 14-bus, 118-bus) and tooling like MATPOWER. That is a different and
much heavier problem than household smart meter tampering. At the meter
level there is no good public labelled set, so building the injector is the
standard approach at this scope, not a shortcut.

The upside: we get perfect ground truth. We know exactly which readings were
tampered with and how, which is what makes Day 4's precision and recall
numbers trustworthy rather than guesswork.

WHY THESE FOUR ATTACKS AND NOT FOUR FLAVOURS OF RANDOM NOISE:
Each one defeats a different detection strategy. If all four were simple
magnitude changes, the z-score would catch everything and the project would
prove nothing.

  1. SCALING -- multiply readings by a factor (0.4 = report 40% of real use)
     Real world: meter tampering for energy theft. The classic FDI attack.
     Detection: easy for z-score, the magnitude shift is immediate.
     Why include an easy one: it is the control case. If a detector cannot
     catch this, it is broken.

  2. RAMP -- add a slowly growing offset (+0.05 kW each hour)
     Real world: stealthy long-term manipulation, solar feed-in fraud.
     Detection: THIS IS WHY CUSUM EXISTS IN THIS PROJECT. Each individual
     step is far inside normal variation, so a z-score checking one reading
     at a time is blind to it by construction. Only cumulative deviation
     over many hours exposes it.

  3. REPLAY -- retransmit readings from 24 hours earlier
     Real world: classic ICS/SCADA concealment. Hide a fault or an ongoing
     theft by making the monitoring system believe today looks like
     yesterday. Stuxnet used this idea.
     Detection: hardest of the four. Every replayed value is a genuine
     reading that really occurred, just at the wrong time. There is no
     magnitude anomaly at all. It only shows as a temporal mismatch.

  4. NOISE -- add uniform random noise up to +/- 0.8 kW
     Real world: signal corruption, jamming, blinding a detector.
     Also serves as the calibration case: 0.8 kW is comparable to genuine
     peak-hour swings, so how much of this gets flagged tells us how
     trigger-happy our thresholds are.

WHY THE ATTACK WINDOWS DO NOT OVERLAP, AND HAVE 50-HOUR GAPS:
Two reasons. Non-overlap means a flag can be attributed to exactly one
attack type, so per-attack recall is unambiguous. The gaps matter because
CUSUM accumulates: without a drain period, leftover accumulation from the
ramp attack would still be above threshold when the replay window starts,
and CUSUM would get credit for detecting an attack it never actually saw.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.logger_config import log_operational, log_ground_truth


def inject_scaling_attack(df: pd.DataFrame, start: int, end: int,
                          scale_factor: float = 0.4) -> pd.DataFrame:
    """
    Multiply reported readings by scale_factor over [start:end].

    scale_factor=0.4 means the meter reports 40% of real consumption.
    Chosen deliberately over something extreme like 0.05: a real thief wants
    a believable bill, not a zero bill that triggers a manual inspection.
    """
    df = df.copy()
    win = df.index[start:end]
    df.loc[win, "reported_power_kw"] = (df.loc[win, "true_power_kw"] * scale_factor).round(4)
    df.loc[win, "is_attacked"] = True
    df.loc[win, "attack_type"] = "scaling"

    log_ground_truth("scaling", start, end, {"scale_factor": scale_factor}, len(win))
    log_operational("attack_injector", "ok",
                    f"Scaling attack injected: {len(win)} readings, factor={scale_factor}")
    return df


def inject_ramp_attack(df: pd.DataFrame, start: int, end: int,
                       ramp_per_step: float = 0.05,
                       direction: str = "up") -> pd.DataFrame:
    """
    Add a linearly growing offset across the window.

    At 0.05 kW/hour over 100 hours the final offset is +5.0 kW -- enormous
    in total, invisible step by step. That gap between "obvious in aggregate"
    and "invisible pointwise" is the entire point of this attack.
    """
    df = df.copy()
    win = df.index[start:end]
    n = len(win)
    sign = 1.0 if direction == "up" else -1.0
    offset = np.arange(n, dtype=float) * ramp_per_step * sign

    df.loc[win, "reported_power_kw"] = np.maximum(
        0.0, df.loc[win, "true_power_kw"].values + offset).round(4)
    df.loc[win, "is_attacked"] = True
    df.loc[win, "attack_type"] = "ramp"

    log_ground_truth("ramp", start, end,
                     {"ramp_per_step": ramp_per_step, "direction": direction}, n)
    log_operational("attack_injector", "ok",
                    f"Ramp attack injected: {n} readings, step={ramp_per_step}/h, {direction}")
    return df


def inject_replay_attack(df: pd.DataFrame, start: int, end: int,
                         replay_lag_hours: int = 24) -> pd.DataFrame:
    """
    Replace readings with values from replay_lag_hours earlier.

    24 hours is chosen because it aligns with the daily cycle -- the replayed
    data has the right shape for the time of day, making it maximally
    plausible. A 6-hour lag would be far easier to spot (evening peak values
    appearing at 1am), which would make this attack artificially easy.
    """
    df = df.copy()
    win = df.index[start:end]
    n = len(win)

    src_start = max(0, start - replay_lag_hours)
    src = df.iloc[src_start:src_start + n]["true_power_kw"].values

    if len(src) < n:  # window starts too early in the series to have a source
        src = np.concatenate([np.full(n - len(src), df.true_power_kw.mean()), src])

    df.loc[win, "reported_power_kw"] = np.round(src, 4)
    df.loc[win, "is_attacked"] = True
    df.loc[win, "attack_type"] = "replay"

    log_ground_truth("replay", start, end, {"replay_lag_hours": replay_lag_hours}, n)
    log_operational("attack_injector", "ok",
                    f"Replay attack injected: {n} readings, lag={replay_lag_hours}h")
    return df


def inject_noise_attack(df: pd.DataFrame, start: int, end: int,
                        amplitude: float = 0.8, seed: int = 99) -> pd.DataFrame:
    """
    Add uniform noise in [-amplitude, +amplitude] to reported readings.

    amplitude=0.8 kW sits deliberately close to real peak-hour variation.
    Set it to 5.0 and every detector catches it trivially; set it to 0.05
    and nothing catches it and nothing is learned. 0.8 is the interesting
    middle where the threshold choice actually matters.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    win = df.index[start:end]
    n = len(win)

    df.loc[win, "reported_power_kw"] = np.maximum(
        0.0, df.loc[win, "true_power_kw"].values + rng.uniform(-amplitude, amplitude, n)
    ).round(4)
    df.loc[win, "is_attacked"] = True
    df.loc[win, "attack_type"] = "noise"

    log_ground_truth("noise", start, end, {"amplitude": amplitude}, n)
    log_operational("attack_injector", "ok",
                    f"Noise attack injected: {n} readings, amplitude=+/-{amplitude}")
    return df


def inject_all_attacks(df: pd.DataFrame, baseline_hours: int = 720) -> pd.DataFrame:
    """
    Inject all four attacks into non-overlapping windows after the baseline.

    Schedule, where b = baseline_hours (default 720 = 30 days):
        scaling  b+30  .. b+80    50 h
        ramp     b+130 .. b+230  100 h   (long: drift needs time to build)
        replay   b+280 .. b+330   50 h
        noise    b+380 .. b+430   50 h

    The first 30 hours after the baseline are left clean so the detectors
    have a settling period before anything happens.
    """
    b = baseline_hours
    log_operational("attack_injector", "ok",
                    f"Injecting 4 attacks. Baseline period: {b} hours (untouched).")

    df = inject_scaling_attack(df, b + 30,  b + 80,  scale_factor=0.4)
    df = inject_ramp_attack(   df, b + 130, b + 230, ramp_per_step=0.05, direction="up")
    df = inject_replay_attack( df, b + 280, b + 330, replay_lag_hours=24)
    df = inject_noise_attack(  df, b + 380, b + 430, amplitude=0.8)

    log_operational("attack_injector", "ok",
                    f"Injection complete. {int(df.is_attacked.sum())} of {len(df)} "
                    f"readings are attacked ({100*df.is_attacked.mean():.1f}%).")
    return df
