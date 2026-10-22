"""
visualize.py  --  Day 5

Charts for the README and for demonstrating the system.

WHY VISUALS MATTER FOR THIS PROJECT SPECIFICALLY:
A detection result is a claim. "CUSUM catches ramp attacks that z-score
misses" is a sentence someone has to take on trust; a plot of the ramp window
with both detectors' flags marked is evidence they can check themselves. For
a portfolio piece that will be read by people who have no reason to trust the
author yet, showing beats telling.

WHY MATPLOTLIB AND NOT PLOTLY OR SEABORN:
Static PNG output that renders directly in a GitHub README. Plotly produces
interactive HTML that GitHub will not render inline, so the charts would be
invisible where they matter most. Seaborn is a matplotlib wrapper and would
add a dependency for styling that is not needed here.
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend: required for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.logger_config import log_operational

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

ATTACK_COLORS = {
    "scaling": "#e74c3c",
    "ramp":    "#e67e22",
    "replay":  "#9b59b6",
    "noise":   "#16a085",
}


def plot_overview(df: pd.DataFrame, filename: str = "01_overview.png") -> Path:
    """
    Full test period: true vs reported consumption, with attack windows shaded.

    This is the orientation chart. Before any detector results are shown, the
    reader needs to see what the attacks actually look like -- particularly
    that the ramp diverges gradually and the replay is visually invisible.
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df.index, df["true_power_kw"], lw=0.7, color="#2c3e50",
            label="True consumption", alpha=0.8)
    ax.plot(df.index, df["reported_power_kw"], lw=0.7, color="#3498db",
            label="Reported (meter)", alpha=0.9)

    for atk, color in ATTACK_COLORS.items():
        mask = df["attack_type"] == atk
        if mask.any():
            ax.axvspan(df.index[mask][0], df.index[mask][-1],
                       color=color, alpha=0.18, label=f"{atk} attack")

    ax.set_title("Smart meter readings: true vs reported, with injected attack windows")
    ax.set_ylabel("Power (kW)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log_operational("visualize", "ok", f"Saved {filename}")
    return path


def plot_attack_zoom(df: pd.DataFrame, attack_type: str, detectors: dict,
                     pad: int = 40, filename: str = None) -> Path:
    """
    Zoom into one attack window and mark which detectors fired where.

    The ramp plot is the important one: it shows the reported line drifting
    away from truth while each individual point still looks plausible, and
    CUSUM firing partway through once accumulated deviation crosses threshold.
    """
    filename = filename or f"02_zoom_{attack_type}.png"
    idx = np.where(df["attack_type"].values == attack_type)[0]
    if len(idx) == 0:
        raise ValueError(f"No readings with attack_type={attack_type}")

    lo, hi = max(0, idx[0] - pad), min(len(df), idx[-1] + pad)
    sub = df.iloc[lo:hi]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(sub.index, sub["true_power_kw"], lw=1.1, color="#2c3e50", label="True")
    ax.plot(sub.index, sub["reported_power_kw"], lw=1.1, color="#3498db", label="Reported")

    amask = sub["attack_type"] == attack_type
    ax.axvspan(sub.index[amask][0], sub.index[amask][-1],
               color=ATTACK_COLORS.get(attack_type, "#95a5a6"), alpha=0.18,
               label=f"{attack_type} window")

    # Stack detector hits at different heights so overlaps stay readable
    markers = ["v", "o", "s", "D"]
    ymax = float(sub[["true_power_kw", "reported_power_kw"]].max().max())
    for i, (name, col) in enumerate(detectors.items()):
        hits = sub[sub[col]]
        if len(hits):
            ax.scatter(hits.index, np.full(len(hits), ymax * (1.08 + 0.07 * i)),
                       marker=markers[i % len(markers)], s=22, label=f"{name} flag")

    ax.set_title(f"{attack_type.capitalize()} attack: detector response")
    ax.set_ylabel("Power (kW)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log_operational("visualize", "ok", f"Saved {filename}")
    return path


def plot_metrics_comparison(results: pd.DataFrame,
                            filename: str = "03_metrics.png") -> Path:
    """Grouped bars: precision, recall and F1 for every detector."""
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(results))
    w = 0.26

    ax.bar(x - w, results["precision"], w, label="Precision", color="#3498db")
    ax.bar(x,     results["recall"],    w, label="Recall",    color="#e67e22")
    ax.bar(x + w, results["f1"],        w, label="F1",        color="#2ecc71")

    ax.set_xticks(x)
    ax.set_xticklabels(results.index, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Detector performance (higher is better)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log_operational("visualize", "ok", f"Saved {filename}")
    return path


def plot_per_attack_heatmap(per_attack: pd.DataFrame,
                            filename: str = "04_per_attack_recall.png") -> Path:
    """
    Heatmap of recall by attack type and detector -- the blind-spot map.

    Dark cells are misses. The replay row being dark across the general
    detectors is the single most informative result in the project, which is
    why it gets its own chart instead of living in a table.
    """
    data = per_attack.drop(columns=["n_readings"], errors="ignore")
    fig, ax = plt.subplots(figsize=(1.6 * len(data.columns) + 3, 0.9 * len(data) + 2.5))
    im = ax.imshow(data.values.astype(float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)

    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            v = float(data.values[i, j])
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black" if 0.25 < v < 0.85 else "white", fontsize=10)

    ax.set_title("Recall by attack type (1.00 = every attacked reading caught)")
    fig.colorbar(im, ax=ax, label="Recall")
    fig.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log_operational("visualize", "ok", f"Saved {filename}")
    return path


def plot_cusum_trace(df: pd.DataFrame, h: float,
                     filename: str = "05_cusum_trace.png") -> Path:
    """
    The CUSUM accumulators over time against the decision threshold.

    This chart is the argument for why CUSUM exists in the project: during the
    ramp window the accumulator climbs steadily even though no single reading
    is unusual, and the sawtooth pattern shows reset-on-alarm working.
    """
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.plot(df.index, df["cusum_pos"], lw=0.8, color="#c0392b", label="CUSUM+ (upward drift)")
    ax.plot(df.index, df["cusum_neg"], lw=0.8, color="#2980b9", label="CUSUM- (downward drift)")
    ax.axhline(h, ls="--", color="black", lw=1, label=f"Decision threshold h={h:.2f}")

    for atk, color in ATTACK_COLORS.items():
        mask = df["attack_type"] == atk
        if mask.any():
            ax.axvspan(df.index[mask][0], df.index[mask][-1], color=color, alpha=0.15)

    ax.set_title("CUSUM accumulators (sawtooth = reset after each alarm)")
    ax.set_ylabel("Accumulated deviation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log_operational("visualize", "ok", f"Saved {filename}")
    return path
