"""
ml_detector.py  --  Day 4

Isolation Forest as a third detector, added to measure a tradeoff rather
than to assume machine learning wins.

-----------------------------------------------------------------------------
WHY ISOLATION FOREST AND NOT SOMETHING ELSE
-----------------------------------------------------------------------------
Requirements: unsupervised (a real utility has no labelled attacks), fast,
and able to use several features at once.

  Isolation Forest    CHOSEN. Isolates points by random splitting; anomalies
                      need fewer splits to isolate, so they get a worse score.
                      Linear time, no distance computations, handles mixed
                      feature scales without normalisation, and has one
                      meaningful hyperparameter (contamination).

  One-Class SVM       Rejected. O(n^2) to O(n^3) training, needs careful
                      kernel and gamma tuning, and degrades badly above a few
                      tens of thousands of points. A real meter fleet is far
                      larger than that.

  Local Outlier Factor  Rejected. Density-based, so it needs a distance
                      metric over the feature space -- and "distance" between
                      a kW value and an hour-of-day integer is not meaningful
                      without careful scaling. Also awkward for streaming.

  LSTM autoencoder    Rejected for this project, though it is the right tool
                      at a larger scale. It needs far more than 30 days of
                      baseline, retraining per meter, and produces a
                      reconstruction error with no human-readable reason
                      attached. When the output accuses a customer of theft,
                      "the network said so" is not defensible.

-----------------------------------------------------------------------------
FEATURE ENGINEERING -- THE PART THAT ACTUALLY MATTERS
-----------------------------------------------------------------------------
The statistical detectors on Day 3 look at one reading at a time. That is
precisely why both of them miss the replay attack almost entirely (z-score
caught 0 of 50, CUSUM caught 3 of 50): every replayed value is a real
reading that genuinely occurred, so pointwise there is nothing wrong with it.

The attack is only visible in the relationship between readings. So the
features below deliberately encode temporal structure:

  reported_power_kw   raw value -- baseline signal
  residual            reading minus its seasonal expectation. Same quantity
                      CUSUM accumulates; gives the forest a deseasonalised view.
  hour_sin, hour_cos  hour of day encoded on a circle. WHY NOT hour AS 0-23:
                      a raw integer tells the model that hour 23 and hour 0
                      are 23 units apart when they are actually adjacent.
                      Sine/cosine encoding makes midnight continuous.
  lag_24_diff         reading[t] - reading[t-24]. THIS IS THE REPLAY DETECTOR.
                      A 24-hour replay makes this value exactly 0.0, which is
                      extraordinarily rare in real data where consecutive days
                      always differ by some noise. The forest isolates it fast.
  rolling_std_6       standard deviation over the last 6 hours. Catches the
                      noise attack (variance spikes) and flags the opposite
                      case too: suspiciously flat data.
  diff_1              reading[t] - reading[t-1]. Short-term jerk, catches
                      abrupt scaling steps at the attack boundary.

THE HONEST POINT ABOUT THIS RESULT:
If Isolation Forest beats the statistical detectors on replay, it is not
because the algorithm reasons better. It is because lag_24_diff hands it the
answer. The lesson is that feature engineering did the work, and the same
feature could be fed to a simple threshold rule for nearly the same result.
Claiming "ML beat statistics" here would be dishonest -- what actually
happened is "a temporal feature beat pointwise features".

-----------------------------------------------------------------------------
WHY contamination = 0.05
-----------------------------------------------------------------------------
contamination tells the forest what proportion of data to treat as anomalous;
it sets the score cutoff. The true attack rate in the test period is about
21%, but that number would never be known in production. 0.05 is a
deliberately conservative operational guess: alarm on the worst 5%, because
an analyst can triage 5% of readings and cannot triage 21%. Day 4's
evaluation reports what this choice costs in recall.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.logger_config import log_detection, log_audit, log_operational
from src.detector import compute_seasonal_baselines

FEATURES = ["reported_power_kw", "residual", "hour_sin", "hour_cos",
            "lag_24_diff", "rolling_std_6", "diff_1"]


def build_features(df: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the feature matrix. See module docstring for why each exists.

    NaNs from lag/rolling windows at the start of the series are filled with
    0.0 rather than dropped, so the output stays aligned row-for-row with the
    input -- alignment matters because Day 4 compares flags against ground
    truth by position.
    """
    out = df.copy()
    idx = out.index

    tmp = out.copy()
    tmp["hour"] = idx.hour
    tmp["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    tmp = tmp.merge(baselines[["hour", "is_weekend", "mean"]],
                    on=["hour", "is_weekend"], how="left")
    tmp.index = idx

    out["residual"] = (tmp["reported_power_kw"] - tmp["mean"]).fillna(0.0).round(4)
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24).round(4)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24).round(4)
    out["lag_24_diff"] = out["reported_power_kw"].diff(24).fillna(0.0).round(4)
    out["rolling_std_6"] = (out["reported_power_kw"].rolling(6, min_periods=1)
                              .std().fillna(0.0).round(4))
    out["diff_1"] = out["reported_power_kw"].diff(1).fillna(0.0).round(4)
    return out


class IsolationForestDetector:
    """Unsupervised anomaly detector over engineered temporal features."""

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200,
                 random_state: int = 42, meter_id: str = "METER_001"):
        self.contamination = contamination
        self.meter_id = meter_id
        self._baselines = None
        self.model = IsolationForest(
            n_estimators=n_estimators,       # 200: score variance flattens out
            contamination=contamination,     # past ~150 trees; 200 is cheap insurance
            random_state=random_state,       # fixed so runs are reproducible
            n_jobs=-1,
        )
        log_audit("if_params_init", None,
                  {"contamination": contamination, "n_estimators": n_estimators},
                  "IsolationForestDetector initialised", "system")

    def fit(self, baseline_df: pd.DataFrame) -> "IsolationForestDetector":
        """
        Fit on CLEAN baseline data only.

        WHY BASELINE-ONLY AND NOT THE WHOLE SERIES:
        Fitting on data that contains the attacks would teach the forest that
        attacks are normal, since it is unsupervised and cannot tell them
        apart. Training on a known-clean window is the semi-supervised
        framing, and it matches reality: a utility can reasonably assume a
        historical period was uncompromised.
        """
        self._baselines = compute_seasonal_baselines(baseline_df)
        feats = build_features(baseline_df, self._baselines)
        self.model.fit(feats[FEATURES].values)
        log_operational("ml_detector", "ok",
                        f"Isolation Forest fitted on {len(feats)} clean readings, "
                        f"{len(FEATURES)} features, contamination={self.contamination}")
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds columns: if_score, if_flagged.

        if_score is sklearn's decision_function: negative means more anomalous.
        It is negated here so that higher = more anomalous, matching the
        direction of the z-score and CUSUM scores for easier comparison.
        """
        if self._baselines is None:
            raise RuntimeError("Call .fit(baseline_df) before .score()")

        feats = build_features(df, self._baselines)
        X = feats[FEATURES].values

        raw = self.model.decision_function(X)      # higher = more normal
        pred = self.model.predict(X)               # -1 anomaly, 1 normal

        out = df.copy()
        out["if_score"] = np.round(-raw, 4)        # flip: higher = more anomalous
        out["if_flagged"] = (pred == -1)

        for ts, row in out[out["if_flagged"]].iterrows():
            log_detection(
                meter_id=self.meter_id,
                raw_reading=float(row["reported_power_kw"]),
                expected_value=float(row["reported_power_kw"]),  # IF gives no expectation
                deviation_score=float(row["if_score"]),
                detector_name="isolation_forest",
                suspected_attack_type=str(row.get("attack_type", "unknown")),
                severity="high" if row["if_score"] > 0.1 else "medium",
            )

        log_operational("ml_detector", "ok",
                        f"Isolation Forest scored {len(out)} readings, "
                        f"flagged {int(out.if_flagged.sum())}")
        return out
