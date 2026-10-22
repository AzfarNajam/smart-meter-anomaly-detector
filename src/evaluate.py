"""
evaluate.py  --  Day 4

Scores every detector against the ground truth recorded during injection.

-----------------------------------------------------------------------------
WHY PRECISION AND RECALL, AND WHY ACCURACY IS THE WRONG METRIC HERE
-----------------------------------------------------------------------------
About 21% of the test period is attacked, so 79% is clean. A detector that
flags nothing at all scores 79% accuracy while catching zero attacks. Accuracy
rewards doing nothing on imbalanced data, which is why it is not reported.

    Precision = TP / (TP + FP)
        Of everything flagged, how much was really an attack?
        Low precision means analysts drown in false alarms and start ignoring
        the system. This is how real SOC alert fatigue happens.

    Recall = TP / (TP + FN)
        Of all real attacks, how many were caught?
        Low recall means theft and tampering continue undetected.

    F1 = harmonic mean of the two
        A single number for comparison. The harmonic mean is used rather than
        the arithmetic mean because it punishes imbalance: 100% recall with
        1% precision averages to 50.5% arithmetically but scores 2% on F1,
        which is the honest reading of a detector that flags everything.

    False Positive Rate = FP / (FP + TN)
        Reported separately because it is the number an operator actually
        budgets against. At one reading per hour per meter, a 1% FPR across
        100,000 meters is 24,000 false alarms per day.

-----------------------------------------------------------------------------
WHY PER-ATTACK-TYPE RECALL IS REPORTED SEPARATELY
-----------------------------------------------------------------------------
An aggregate F1 hides the finding that matters. A detector can post a
respectable overall score while being completely blind to one attack class --
which is exactly what happens here with replay. Breaking recall down by attack
type is what turns "detector A scores 0.62" into "detector A cannot see replay
attacks at all", and only the second statement is actionable.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.logger_config import log_operational


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return TP, FP, TN, FN as plain integers."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    return {
        "TP": int(np.sum(y_true & y_pred)),
        "FP": int(np.sum(~y_true & y_pred)),
        "TN": int(np.sum(~y_true & ~y_pred)),
        "FN": int(np.sum(y_true & ~y_pred)),
    }


def metrics_from_counts(c: dict) -> dict:
    """
    Convert confusion counts into rates.

    Every denominator is guarded: a detector that flags nothing has TP+FP = 0,
    and precision would be 0/0. Returning 0.0 in that case is the standard
    convention and keeps comparison tables readable.
    """
    tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        **c,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
    }


def evaluate_detector(df: pd.DataFrame, flag_col: str, name: str) -> dict:
    """Evaluate one detector's flag column against is_attacked."""
    m = metrics_from_counts(confusion_counts(df["is_attacked"].values,
                                             df[flag_col].values))
    m["detector"] = name
    log_operational("evaluate", "ok",
                    f"{name}: P={m['precision']:.3f} R={m['recall']:.3f} "
                    f"F1={m['f1']:.3f} FPR={m['fpr']:.4f}")
    return m


def evaluate_all(df: pd.DataFrame, detectors: dict) -> pd.DataFrame:
    """
    Evaluate every detector. detectors maps display name -> flag column.
    Returns a DataFrame sorted by F1 descending.
    """
    rows = [evaluate_detector(df, col, name) for name, col in detectors.items()]
    out = pd.DataFrame(rows).set_index("detector")
    out = out[["TP", "FP", "TN", "FN", "precision", "recall", "f1", "fpr"]]
    return out.sort_values("f1", ascending=False)


def per_attack_recall(df: pd.DataFrame, detectors: dict) -> pd.DataFrame:
    """
    Recall broken down by attack type -- the table that exposes blind spots.

    Rows are attack types, columns are detectors, values are recall in that
    class. A 0.00 cell means that detector is structurally blind to that
    attack, which is far more useful than any aggregate score.
    """
    attacks = [a for a in df["attack_type"].unique() if a != "none"]
    rows = {}
    for atk in sorted(attacks):
        sub = df[df["attack_type"] == atk]
        rows[atk] = {name: round(float(sub[col].mean()), 4)
                     for name, col in detectors.items()}
        rows[atk]["n_readings"] = len(sub)
    return pd.DataFrame(rows).T


def build_ensemble(df: pd.DataFrame, detectors: dict,
                   strategy: str = "union") -> pd.Series:
    """
    Combine detectors into one flag.

    WHY UNION (flag if ANY detector fires) IS THE DEFAULT:
    The detectors here have complementary blind spots by design -- CUSUM finds
    drift, the replay rule finds repetition, z-score finds magnitude jumps.
    A majority vote would suppress exactly the cases where only one detector
    can possibly see the attack, which defeats the purpose of running several.

    The cost is precision: union inherits the false positives of every member,
    so the ensemble is never more precise than its most precise member. That
    tradeoff is measured rather than assumed -- see the results table.

    Intersection is provided for completeness. It is very high precision and
    very low recall, appropriate only when an alarm triggers something
    expensive like a site visit or a legal process.
    """
    cols = list(detectors.values())
    if strategy == "union":
        return df[cols].any(axis=1)
    if strategy == "intersection":
        return df[cols].all(axis=1)
    if strategy == "majority":
        return df[cols].sum(axis=1) > (len(cols) / 2)
    raise ValueError(f"Unknown strategy: {strategy}")
