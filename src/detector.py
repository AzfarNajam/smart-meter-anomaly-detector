"""
detector.py  --  Day 3

Two statistical detectors that run in parallel. Neither is better overall --
they fail on different attacks, which is exactly why both exist.

-----------------------------------------------------------------------------
DETECTOR 1: SEASONAL Z-SCORE
-----------------------------------------------------------------------------
Compare each reading against the historical distribution for that specific
(hour of day, day of week) slot. The question it asks is not "is 2.2 kW
high?" but "is 2.2 kW high FOR A TUESDAY AT 07:00?".

    z = (reading - mean_for_that_slot) / std_for_that_slot
    flag when |z| > threshold

WHY SEASONAL GROUPING RATHER THAN ONE GLOBAL MEAN:
The global mean here is about 0.98 kW with std 0.65. Every morning and
evening peak sits 2+ std above that, so a global z-score would flag roughly
a third of all normal readings. Grouping by (hour, weekday) makes the
comparison like-for-like: 2.2 kW at 07:00 is unremarkable, the same 2.2 kW
at 03:00 is a genuine outlier. Seasonality is not a detail here, it is what
makes the method work at all.

WHY threshold = 3.0:
The 3-sigma convention. For roughly normal data about 99.7% of readings fall
inside 3 std, so the expected false positive rate is ~0.3%. Lower it to 2.0
and false positives rise about tenfold; raise it to 4.0 and subtle attacks
slip through. 3.0 is the standard starting point, and Day 4 measures whether
it was the right call rather than assuming it.

STRONG AGAINST: scaling (immediate magnitude shift, large |z| straight away)
BLIND TO:       ramp (each +0.05 kW step is ~0.08 std, invisible pointwise)

-----------------------------------------------------------------------------
DETECTOR 2: CUSUM (Cumulative Sum Control Chart)
-----------------------------------------------------------------------------
Accumulate signed deviations from the baseline mean over time and alarm when
the running total crosses a decision threshold.

    C_pos[t] = max(0, C_pos[t-1] + (x[t] - mu - k))
    C_neg[t] = max(0, C_neg[t-1] + (mu - x[t] - k))
    flag when C_pos > h  or  C_neg > h

    mu = baseline mean
    k  = slack, set to 0.5 * baseline std
    h  = decision threshold, set to 5.0 * baseline std

WHY THE max(0, ...) RESET:
Without it the sum drifts forever on ordinary noise and eventually alarms on
nothing. Flooring at zero means the accumulator drains during normal
operation and only climbs during a persistent one-directional deviation.
That is the whole trick: CUSUM ignores random scatter and reacts to bias.

WHY k = 0.5 sigma:
k is a deadband. Deviations smaller than k actively shrink the accumulator.
Setting k to half a std is the classical choice for detecting a sustained
shift of about one std -- small enough to catch a real drift, large enough
that ordinary noise never accumulates.

WHY h = 5 sigma:
Standard control-chart practice. With k = 0.5 sigma, h = 5 sigma gives a
long average run length between false alarms while still detecting a
1-sigma shift within roughly 10 samples. Lower h reacts faster and false
alarms more; higher h is quieter and slower.

STRONG AGAINST: ramp (50 hours of +0.05 kW bias accumulates far past h even
                though no single reading is unusual)
WEAKER ON:      short sharp spikes, because the accumulator needs time to build

-----------------------------------------------------------------------------
WHY STATISTICS FIRST AND NOT AN LSTM OR AUTOENCODER
-----------------------------------------------------------------------------
1. Explainability. "4.2 std above the Tuesday-14:00 mean" is a sentence an
   operator can act on and challenge. A neural network score is a number
   with no defensible reason attached, which is a serious problem when the
   output accuses someone of tampering or theft.
2. Data budget. 30 days of baseline is nowhere near enough to train a
   sequence model that generalises. Statistical baselines are stable on
   exactly this amount of data.
3. Transfer. Moving to a different household means recomputing mu and sigma.
   A deep model needs full retraining per meter, which does not scale to a
   utility with millions of meters.
4. Compute. These run on a Raspberry Pi at the edge. That matters for a
   threat model where the meter itself may be the compromised component.

Day 4 adds Isolation Forest anyway -- not because ML is expected to win, but
because measuring the tradeoff is a far stronger result than assuming it.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.logger_config import log_detection, log_audit, log_operational


def compute_seasonal_baselines(baseline_df: pd.DataFrame) -> pd.DataFrame:
    """
    Learn (mean, std) for every (hour, is_weekend) slot from clean data.

    WHY (hour, is_weekend) AND NOT (hour, dayofweek):
    This is a bias-variance tradeoff and it was chosen by measurement, not
    taste. Grouping by (hour, dayofweek) creates 24 x 7 = 168 groups. A
    30-day baseline is 720 readings, so each group gets about 4 samples --
    and a standard deviation estimated from 4 points is close to noise.
    Measured false positive rate with that grouping was 3.2%, roughly ten
    times the 0.3% a 3-sigma threshold should give.

    Grouping by (hour, is_weekend) creates 24 x 2 = 48 groups with about 15
    samples each. It gives up the ability to distinguish Tuesday from
    Thursday -- which for household consumption is a difference that barely
    exists -- and buys much more stable variance estimates in return.

    If a longer baseline were available (90+ days), (hour, dayofweek) would
    become viable again. The count column and the warning below exist so
    this constraint is visible at runtime rather than silently degrading.

    std is floored at 0.05 to avoid division by a near-zero denominator
    during very stable overnight hours, which would otherwise turn tiny
    harmless fluctuations into enormous z-scores.
    """
    tmp = baseline_df.copy()
    tmp["hour"] = tmp.index.hour
    tmp["is_weekend"] = (tmp.index.dayofweek >= 5).astype(int)

    stats = (tmp.groupby(["hour", "is_weekend"])["reported_power_kw"]
                .agg(["mean", "std", "count"]).reset_index())
    stats["std"] = stats["std"].fillna(0.1).clip(lower=0.05)

    min_n = int(stats["count"].min())
    log_operational("detector", "ok",
                    f"Seasonal baselines built: {len(stats)} (hour, is_weekend) groups "
                    f"from {len(baseline_df)} clean readings; smallest group n={min_n}")
    if min_n < 10:
        log_operational("detector", "error",
                        f"Baseline too short: smallest group has only {min_n} samples. "
                        f"std estimates will be unstable. Use >= 30 days of baseline.")
    return stats


class ZScoreDetector:
    """Seasonal z-score detector. Flags readings where |z| exceeds threshold."""

    def __init__(self, threshold: float = 3.0, meter_id: str = "METER_001"):
        self.threshold = threshold
        self.meter_id = meter_id
        self._baselines = None
        log_audit("threshold_init", None, threshold,
                  "ZScoreDetector initialised with 3-sigma convention", "system")

    def fit(self, baseline_df: pd.DataFrame) -> "ZScoreDetector":
        self._baselines = compute_seasonal_baselines(baseline_df)
        return self

    def update_threshold(self, new_value: float, reason: str) -> None:
        """Change the threshold and record why in audit.log."""
        log_audit("threshold_update", self.threshold, new_value, reason, "developer")
        self.threshold = new_value

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score every reading. Adds columns: zscore, zscore_flagged.

        NOTE ON THE INDEX: pandas .merge() joins on columns and silently
        discards the DatetimeIndex, replacing it with a RangeIndex. Every
        downstream step here is time-based, so the original index is saved
        before the merge and restored after. This is a real bug that is easy
        to miss because the code still runs -- it just quietly loses time.
        """
        if self._baselines is None:
            raise RuntimeError("Call .fit(baseline_df) before .score()")

        df = df.copy()
        original_index = df.index          # preserve before merge destroys it
        df["hour"] = df.index.hour
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

        df = df.merge(self._baselines, on=["hour", "is_weekend"], how="left")
        df.index = original_index          # restore; merge preserves row order

        df["zscore"] = ((df["reported_power_kw"] - df["mean"]) / df["std"]).round(4)
        df["zscore_flagged"] = df["zscore"].abs() > self.threshold

        for ts, row in df[df["zscore_flagged"]].iterrows():
            log_detection(
                meter_id=self.meter_id,
                raw_reading=float(row["reported_power_kw"]),
                expected_value=round(float(row["mean"]), 4),
                deviation_score=abs(float(row["zscore"])),
                detector_name="zscore_seasonal",
                suspected_attack_type=str(row.get("attack_type", "unknown")),
                severity="high" if abs(row["zscore"]) > 5 else "medium",
            )

        log_operational("detector", "ok",
                        f"Z-Score scored {len(df)} readings, flagged "
                        f"{int(df.zscore_flagged.sum())} (threshold={self.threshold})")
        return df


class CUSUMDetector:
    """
    CUSUM control chart applied to SEASONAL RESIDUALS, not raw readings.

    WHY RESIDUALS -- THE MOST IMPORTANT DESIGN DECISION IN THIS FILE:
    The obvious implementation runs CUSUM directly on the meter readings
    against a single global mean. That was built first, and measured, and it
    failed badly: 601 false positives out of 1910 clean readings, a 31% false
    positive rate.

    The reason is structural, not a tuning problem. The global mean here is
    about 0.98 kW, but the evening peak is 2.2 kW. Every single evening, real
    consumption sits roughly 1.2 kW above the global mean for six consecutive
    hours. CUSUM is built to detect exactly that -- a sustained one-directional
    deviation -- so it faithfully alarms every night on entirely normal
    behaviour. No value of h fixes it: raise h enough to survive the evening
    peak and the ramp attack disappears under it too.

    The fix is to remove the seasonality before accumulating. Instead of
    x[t] - mu_global, accumulate on the residual:

        residual[t] = reading[t] - seasonal_mean(hour, is_weekend)

    Residuals from clean data centre on zero with no daily cycle, so CUSUM
    finally sees what it was designed for: a flat signal where any sustained
    drift is genuinely anomalous. Same algorithm, correct input.

    This is the general lesson: CUSUM assumes a stationary process. Feed it a
    seasonal one and it will detect the season.

    THE ALGORITHM (unchanged, now on residuals with mu = 0):
        C_pos[t] = max(0, C_pos[t-1] + (r[t] - k))
        C_neg[t] = max(0, C_neg[t-1] + (-r[t] - k))
        flag when C_pos > h or C_neg > h

    WHY max(0, ...):
    Without the floor the sum random-walks forever and eventually alarms on
    nothing. Flooring at zero drains the accumulator during normal operation,
    so it only climbs under persistent bias. That is what separates CUSUM
    from a plain running total.

    WHY k = 0.5 * sigma_residual:
    k is a deadband. Deviations smaller than k shrink the accumulator rather
    than growing it, so ordinary noise never builds up. Half a std is the
    classical choice for detecting a sustained shift of about one std.

    WHY h = 5.0 * sigma_residual:
    Standard control-chart practice. With k = 0.5 sigma this gives a long
    average run length between false alarms while still catching a 1-sigma
    shift within roughly 10 samples. Lower h reacts sooner and false-alarms
    more; higher h is quieter and slower. Day 4 measures the actual tradeoff
    on this data rather than trusting the convention.

    STRONG AGAINST: ramp attacks. A +0.05 kW/hour bias is far below the
                    z-score threshold at every individual point, but the
                    residual accumulates past h within a few dozen hours.
    WEAKER ON:      isolated single-point spikes, since the accumulator needs
                    consecutive samples to build.
    """

    def __init__(self, k: float = None, h: float = None, meter_id: str = "METER_001"):
        # Left as None so fit() derives them from the residual std, which
        # keeps the detector portable across meters of different scale.
        self.k = k
        self.h = h
        self.meter_id = meter_id
        self._baselines = None
        self._k_auto = k is None
        self._h_auto = h is None

    def _residuals(self, df: pd.DataFrame) -> np.ndarray:
        """Subtract the seasonal expectation from each reading."""
        tmp = df.copy()
        idx = tmp.index
        tmp["hour"] = tmp.index.hour
        tmp["is_weekend"] = (tmp.index.dayofweek >= 5).astype(int)
        tmp = tmp.merge(self._baselines[["hour", "is_weekend", "mean"]],
                        on=["hour", "is_weekend"], how="left")
        tmp.index = idx
        return (tmp["reported_power_kw"] - tmp["mean"]).values

    def fit(self, baseline_df: pd.DataFrame) -> "CUSUMDetector":
        """Learn the seasonal profile, then set k and h from residual spread."""
        self._baselines = compute_seasonal_baselines(baseline_df)
        resid = self._residuals(baseline_df)
        sigma = float(np.nanstd(resid))

        if self._k_auto:
            self.k = 0.5 * sigma
        if self._h_auto:
            self.h = 5.0 * sigma

        log_audit("cusum_params_init", None,
                  {"sigma_residual": round(sigma, 4),
                   "k": round(self.k, 4), "h": round(self.h, 4)},
                  "CUSUM parameters derived from seasonal residual sigma", "system")
        log_operational("detector", "ok",
                        f"CUSUM fitted on residuals: sigma={sigma:.4f}, "
                        f"k={self.k:.4f}, h={self.h:.4f}")
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds columns: residual, cusum_pos, cusum_neg, cusum_flagged."""
        if self._baselines is None:
            raise RuntimeError("Call .fit(baseline_df) before .score()")

        df = df.copy()
        r = self._residuals(df)
        r = np.nan_to_num(r, nan=0.0)   # unseen slot -> treat as no deviation
        n = len(r)

        # RESET-ON-ALARM: after the accumulator crosses h and raises an alarm,
        # it is reset to zero. Without this the detector latches -- measured
        # at 97% false positives, because the ramp attack drove c_pos past 100
        # and the max(0, ...) floor only drains it by k (~0.07) per step,
        # leaving it alarmed for the next ~1400 readings. Reset-on-alarm is
        # standard control-chart practice: each alarm is an independent event,
        # and the chart restarts to watch for the next one.
        c_pos = np.zeros(n)
        c_neg = np.zeros(n)
        flagged = np.zeros(n, dtype=bool)
        acc_pos = acc_neg = 0.0

        for i in range(n):
            acc_pos = max(0.0, acc_pos + (r[i] - self.k))
            acc_neg = max(0.0, acc_neg + (-r[i] - self.k))
            c_pos[i], c_neg[i] = acc_pos, acc_neg

            if acc_pos > self.h or acc_neg > self.h:
                flagged[i] = True
                acc_pos = acc_neg = 0.0   # reset after alarm

        df["residual"] = np.round(r, 4)
        df["cusum_pos"] = np.round(c_pos, 4)
        df["cusum_neg"] = np.round(c_neg, 4)
        df["cusum_flagged"] = flagged

        for ts, row in df[df["cusum_flagged"]].iterrows():
            score_val = float(max(row["cusum_pos"], row["cusum_neg"]))
            log_detection(
                meter_id=self.meter_id,
                raw_reading=float(row["reported_power_kw"]),
                expected_value=round(float(row["reported_power_kw"] - row["residual"]), 4),
                deviation_score=round(score_val, 4),
                detector_name="cusum_residual",
                suspected_attack_type=str(row.get("attack_type", "unknown")),
                severity="high" if score_val > self.h * 2 else "medium",
            )

        log_operational("detector", "ok",
                        f"CUSUM scored {n} readings, flagged "
                        f"{int(df.cusum_flagged.sum())} (h={self.h:.4f})")
        return df


class ReplayDetector:
    """
    Rule-based detector for replay attacks. Deliberately not an anomaly detector.

    -------------------------------------------------------------------------
    WHY THIS EXISTS -- THE KEY FINDING OF THE PROJECT
    -------------------------------------------------------------------------
    Measured on this data, the replay attack defeated every general detector:

        z-score            0 / 50 caught
        CUSUM              3 / 50 caught
        Isolation Forest   0 / 50 caught

    That is not three implementation bugs. It is one structural fact.

    All three detectors answer the question "is this value unusual?". A replay
    attack retransmits a reading that genuinely occurred 24 hours earlier, so
    the value is not unusual at all -- it is a real measurement, at a plausible
    magnitude, for the right time of day.

    Isolation Forest is the clearest case. It was given lag_24_diff
    (reading[t] - reading[t-24]) as a feature, which is exactly 0.0 during a
    replay. That looks like a giveaway, but Isolation Forest isolates points in
    SPARSE regions of the feature space, and 0.0 is the single densest part of
    that distribution -- consecutive days normally differ by only small noise.
    The attack does not produce an outlier. It produces something
    suspiciously average, and hides in the middle of the distribution where
    density-based methods are blind by design.

    THE LESSON: "anomalous" and "malicious" are not the same thing. Some
    attacks are detectable precisely because the data is TOO clean, TOO
    repetitive, TOO normal. Detecting them requires asking a different
    question, not tuning a better model.

    -------------------------------------------------------------------------
    THE RULE
    -------------------------------------------------------------------------
    Flag any run of consecutive readings where |x[t] - x[t-lag]| < epsilon.

    Real meter data essentially never repeats to 4 decimal places across a
    24-hour gap. A sustained run of near-exact matches is not a coincidence;
    it means something is replaying a buffer.

    WHY REQUIRE A RUN INSTEAD OF FLAGGING SINGLE MATCHES:
    Isolated near-matches do occur by chance, especially during flat overnight
    hours where consumption barely moves. Requiring min_run consecutive matches
    removes those. This is the same idea as CUSUM's slack parameter: demand
    persistence before believing the signal.

    WHY epsilon = 0.001 AND NOT EXACT EQUALITY:
    Exact equality works against this simulator, which copies values verbatim.
    Real replayed telemetry passes through float conversion, unit scaling, and
    protocol serialisation, all of which perturb the low bits. A small
    tolerance keeps the rule usable on real data instead of only on this
    testbed. It is deliberately far below genuine measurement variation
    (~0.15 kW), so it cannot fire on ordinary similarity between two days.

    LIMITATION, STATED PLAINLY: this rule only catches the lag it is told to
    look for. An attacker replaying at 25-hour or 168-hour (weekly) offsets
    evades a 24-hour check. Production use means running several lags in
    parallel -- 24h, 48h, 168h -- which is cheap, since each pass is O(n).
    """

    def __init__(self, lag_hours: int = 24, epsilon: float = 0.001,
                 min_run: int = 3, meter_id: str = "METER_001"):
        self.lag_hours = lag_hours
        self.epsilon = epsilon
        self.min_run = min_run
        self.meter_id = meter_id
        log_audit("replay_params_init", None,
                  {"lag_hours": lag_hours, "epsilon": epsilon, "min_run": min_run},
                  "ReplayDetector initialised", "system")

    def fit(self, baseline_df: pd.DataFrame) -> "ReplayDetector":
        """No fitting required -- the rule is deterministic, not learned."""
        log_operational("detector", "ok",
                        f"ReplayDetector ready (lag={self.lag_hours}h, "
                        f"epsilon={self.epsilon}, min_run={self.min_run}). No fit needed.")
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds columns: replay_match, replay_flagged."""
        df = df.copy()
        x = df["reported_power_kw"].values
        n = len(x)
        lag = self.lag_hours

        match = np.zeros(n, dtype=bool)
        if n > lag:
            match[lag:] = np.abs(x[lag:] - x[:-lag]) < self.epsilon

        # Keep only runs of at least min_run consecutive matches
        flagged = np.zeros(n, dtype=bool)
        run_start = None
        for i in range(n + 1):
            is_match = match[i] if i < n else False
            if is_match and run_start is None:
                run_start = i
            elif not is_match and run_start is not None:
                if i - run_start >= self.min_run:
                    flagged[run_start:i] = True
                run_start = None

        df["replay_match"] = match
        df["replay_flagged"] = flagged

        for ts, row in df[df["replay_flagged"]].iterrows():
            log_detection(
                meter_id=self.meter_id,
                raw_reading=float(row["reported_power_kw"]),
                expected_value=float(row["reported_power_kw"]),
                deviation_score=0.0,   # no magnitude deviation: that IS the signal
                detector_name=f"replay_rule_lag{self.lag_hours}h",
                suspected_attack_type=str(row.get("attack_type", "unknown")),
                severity="high",
            )

        log_operational("detector", "ok",
                        f"ReplayDetector scored {n} readings, flagged "
                        f"{int(flagged.sum())} (lag={lag}h)")
        return df
