#!/usr/bin/env python3
"""Render publication-style figures from Avril's frozen public artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
METADATA = ROOT / "Engine" / "target_selection" / "metadata.json"
QUEUE = ROOT / "Engine" / "target_selection" / "tess_discovery_queue_v02.csv"

INK = "#1A1A1A"
MUTED = "#626A73"
GRID = "#D8DDE3"
BLUE = "#3B6FB6"
ORANGE = "#D47A31"
TEAL = "#2C9085"
PURPLE = "#8067A9"
LANE_COLORS = {
    "quick_repeat": BLUE,
    "known_host_residual": TEAL,
    "long_baseline_event": ORANGE,
    "blind_control": PURPLE,
}
LANE_LABELS = {
    "quick_repeat": "Quick repeat",
    "known_host_residual": "Known-host residual",
    "long_baseline_event": "Long-baseline event",
    "blind_control": "Blind control",
}


def configure_style() -> None:
    """Use a compact conference-paper visual language."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.0,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.75,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "legend.frameon": False,
            "text.color": INK,
            "axes.labelcolor": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def figure_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.94, title, fontsize=15.5, fontweight="bold", va="top")
    fig.text(0.055, 0.885, subtitle, fontsize=8.8, color=MUTED, va="top")


def save_figure(fig: plt.Figure, stem: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.10)
    svg_path = ASSETS / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.10)
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def format_axis(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)


def render_validation() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    validation = metadata["training_provenance"]["validation"]

    names = ["Ordinal catalog-yield", "Confirmed-clean", "Occurrence-grid"]
    colors = [BLUE, TEAL, ORANGE]
    lifts = np.asarray(
        [
            validation["frozen_candidate"]["top_decile_lift"],
            validation["confirmed_clean"]["top_decile_lift"],
            validation["occurrence_grid"]["top_decile_lift"],
        ]
    )
    auc = np.asarray(
        [
            validation["frozen_candidate"]["roc_auc"],
            validation["confirmed_clean"]["roc_auc"],
        ]
    )
    ap = np.asarray(
        [
            validation["frozen_candidate"]["average_precision"],
            validation["confirmed_clean"]["average_precision"],
        ]
    )

    fig = plt.figure(figsize=(12.2, 4.55))
    figure_header(
        fig,
        "Planetary Census Gap AI: frozen validation",
        "Five-fold star-separated Kepler DR25 evaluation  ·  catalog-yield enrichment, not TESS planet probability",
    )
    grid = GridSpec(
        1,
        3,
        figure=fig,
        left=0.075,
        right=0.975,
        top=0.77,
        bottom=0.23,
        wspace=0.42,
        width_ratios=[1.35, 1, 1],
    )

    ax_lift = fig.add_subplot(grid[0, 0])
    panel_label(ax_lift, "a")
    y = np.arange(len(names))
    ax_lift.axvline(1.0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    for index, (value, color) in enumerate(zip(lifts, colors, strict=True)):
        ax_lift.hlines(index, 1.0, value, color=color, linewidth=1.7)
        ax_lift.scatter(value, index, s=47, color=color, edgecolor="white", linewidth=0.65, zorder=3)
        ax_lift.text(value + 0.10, index, f"{value:.2f}×", va="center", fontsize=8.5, fontweight="bold")
    ax_lift.set_yticks(y, names)
    ax_lift.invert_yaxis()
    ax_lift.set_xlim(0.7, 5.25)
    ax_lift.set_xlabel("Top-decile lift")
    ax_lift.set_title("Host enrichment", loc="left", pad=10)
    ax_lift.text(
        1.04,
        0.97,
        "1× population rate",
        transform=ax_lift.get_xaxis_transform(),
        fontsize=7.3,
        color=MUTED,
        va="top",
    )
    format_axis(ax_lift)

    model_labels = ["Ordinal", "Confirmed-clean"]
    x = np.arange(2)

    ax_auc = fig.add_subplot(grid[0, 1])
    panel_label(ax_auc, "b")
    ax_auc.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    ax_auc.plot(x, auc, color="#9AA3AD", linewidth=1.0, zorder=1)
    ax_auc.scatter(x, auc, s=58, color=[BLUE, TEAL], edgecolor="white", linewidth=0.7, zorder=3)
    for xpos, value in zip(x, auc, strict=True):
        ax_auc.text(xpos, value + 0.014, f"{value:.4f}", ha="center", fontsize=8.3, fontweight="bold")
    ax_auc.set_xlim(-0.35, 1.35)
    ax_auc.set_ylim(0.47, 0.84)
    ax_auc.set_xticks(x, model_labels)
    ax_auc.set_ylabel("ROC AUC")
    ax_auc.set_title("Star-separated discrimination", loc="left", pad=10)
    ax_auc.text(1.32, 0.507, "chance", color=MUTED, fontsize=7.3, ha="right")
    format_axis(ax_auc, grid_axis="y")

    ax_ap = fig.add_subplot(grid[0, 2])
    panel_label(ax_ap, "c")
    ax_ap.plot(x, ap, color="#9AA3AD", linewidth=1.0, zorder=1)
    ax_ap.scatter(x, ap, s=58, color=[BLUE, TEAL], edgecolor="white", linewidth=0.7, zorder=3)
    for xpos, value in zip(x, ap, strict=True):
        ax_ap.text(xpos, value + 0.0034, f"{value:.4f}", ha="center", fontsize=8.3, fontweight="bold")
    ax_ap.set_xlim(-0.35, 1.35)
    ax_ap.set_ylim(0.0, 0.078)
    ax_ap.set_xticks(x, model_labels)
    ax_ap.set_ylabel("Average precision")
    ax_ap.set_title("Imbalanced-label precision", loc="left", pad=10)
    format_axis(ax_ap, grid_axis="y")

    fig.text(
        0.055,
        0.095,
        "Lift is measured on frozen high-reliability Kepler catalog labels. Values do not validate an individual candidate.",
        fontsize=8.2,
        color=MUTED,
    )
    fig.text(0.975, 0.095, "Source: frozen metadata.json", fontsize=7.6, color=MUTED, ha="right")
    save_figure(fig, "planetary-census-gap-validation")


def read_queue() -> list[dict[str, str]]:
    with QUEUE.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    probability = np.arange(1, len(ordered) + 1) / len(ordered)
    return ordered, probability


def render_queue_atlas() -> None:
    rows = read_queue()
    gap = numeric(rows, "count_gap_percentile")
    detectability = numeric(rows, "tess_detectable_percentile")
    sectors = numeric(rows, "sector_count_recomputed")
    ranks = numeric(rows, "queue_rank")
    lanes = np.asarray([row["discovery_lane"] for row in rows])
    lane_order = list(LANE_COLORS)

    fig = plt.figure(figsize=(12.2, 6.35))
    figure_header(
        fig,
        "Frozen TESS discovery queue",
        "1,000 pre-registered targets  ·  census-gap opportunity, approximate detectability, and multi-sector coverage",
    )
    grid = GridSpec(
        2,
        2,
        figure=fig,
        left=0.075,
        right=0.975,
        top=0.80,
        bottom=0.16,
        wspace=0.32,
        hspace=0.43,
        width_ratios=[1.55, 1],
    )

    ax_scatter = fig.add_subplot(grid[:, 0])
    panel_label(ax_scatter, "a")
    for lane, lane_color in LANE_COLORS.items():
        mask = lanes == lane
        ax_scatter.scatter(
            gap[mask],
            detectability[mask],
            s=7 + 0.85 * sectors[mask],
            color=lane_color,
            alpha=0.36,
            linewidths=0,
            label=LANE_LABELS[lane],
            rasterized=True,
        )
    top_mask = ranks <= 50
    ax_scatter.scatter(
        gap[top_mask],
        detectability[top_mask],
        s=24 + 0.85 * sectors[top_mask],
        facecolors="none",
        edgecolors=INK,
        linewidths=0.55,
        alpha=0.8,
    )
    ax_scatter.axvline(np.median(gap), color=GRID, linewidth=0.7, linestyle=(0, (3, 3)))
    ax_scatter.axhline(np.median(detectability), color=GRID, linewidth=0.7, linestyle=(0, (3, 3)))
    ax_scatter.set_xlim(-0.02, 1.02)
    ax_scatter.set_ylim(-0.02, 1.02)
    ax_scatter.set_xlabel("Planetary census-gap percentile")
    ax_scatter.set_ylabel("Approximate TESS detectability percentile")
    ax_scatter.set_title("Target opportunity before light-curve download", loc="left", pad=10)
    ax_scatter.legend(loc="lower left", ncol=2, fontsize=7.4, handletextpad=0.35, columnspacing=0.9)
    ax_scatter.text(
        0.98,
        0.035,
        "Point area scales with observed sectors\nblack ring: queue rank 1–50",
        transform=ax_scatter.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.3,
        color=MUTED,
    )
    format_axis(ax_scatter, grid_axis="both")

    ax_cdf = fig.add_subplot(grid[0, 1])
    panel_label(ax_cdf, "b")
    for lane in lane_order:
        values, probability = empirical_cdf(sectors[lanes == lane])
        ax_cdf.step(
            values,
            probability,
            where="post",
            color=LANE_COLORS[lane],
            linewidth=1.45,
            label=LANE_LABELS[lane],
        )
    ax_cdf.set_xlim(0, max(sectors) + 1)
    ax_cdf.set_ylim(0, 1.02)
    ax_cdf.set_xlabel("Observed TESS sectors")
    ax_cdf.set_ylabel("Empirical cumulative fraction")
    ax_cdf.set_title("Coverage distribution", loc="left", pad=10)
    ax_cdf.legend(loc="lower right", fontsize=7.2)
    format_axis(ax_cdf, grid_axis="both")

    ax_mix = fig.add_subplot(grid[1, 1])
    panel_label(ax_mix, "c")
    counts = Counter(lanes)
    fractions = np.asarray([counts[lane] / len(rows) for lane in lane_order])
    y = np.arange(len(lane_order))
    for index, lane in enumerate(lane_order):
        ax_mix.hlines(index, 0, fractions[index], color=LANE_COLORS[lane], linewidth=1.35)
        ax_mix.scatter(
            fractions[index],
            index,
            s=43,
            color=LANE_COLORS[lane],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax_mix.text(
            fractions[index] + 0.012,
            index,
            f"{counts[lane]}  ({fractions[index]:.0%})",
            va="center",
            fontsize=8.0,
        )
    ax_mix.set_yticks(y, [LANE_LABELS[lane] for lane in lane_order])
    ax_mix.invert_yaxis()
    ax_mix.set_xlim(0, 0.58)
    ax_mix.set_xlabel("Fraction of frozen queue")
    ax_mix.set_title("Pre-registered discovery-lane allocation", loc="left", pad=10)
    format_axis(ax_mix)

    multi_epoch = sum(row["multi_epoch_coverage"] == "True" for row in rows)
    fig.text(
        0.055,
        0.065,
        f"N = {len(rows):,}  ·  median coverage = {np.median(sectors):.0f} sectors  ·  "
        f"maximum = {np.max(sectors):.0f} sectors  ·  multi-epoch = {multi_epoch:,}/{len(rows):,}",
        fontsize=8.2,
        color=MUTED,
    )
    fig.text(
        0.975,
        0.065,
        "Neither axis is a calibrated planet probability.",
        fontsize=7.6,
        color=MUTED,
        ha="right",
    )
    save_figure(fig, "tess-discovery-queue-atlas")


def main() -> None:
    configure_style()
    render_validation()
    render_queue_atlas()
    print("Rendered research figures in docs/assets")


if __name__ == "__main__":
    main()
