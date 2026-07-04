"""
plots.py
=========

Visualization helpers for the traffic-light optimization project:
convergence curves, algorithm-comparison bar charts, queue-length traces,
and fuzzy membership-function plots.

Every function either takes an existing `matplotlib.axes.Axes` (so callers
can compose multi-panel figures) or, if none is given, creates its own
standalone figure and returns (fig, ax).
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(
    histories: dict[str, np.ndarray],
    title: str = "Optimization Convergence",
    ax: Optional[plt.Axes] = None,
):
    """Plot best-cost-so-far vs. iteration for one or more optimizers.

    `histories` maps a label (e.g. "PSO", "ACO") to a 1-D convergence array.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    for label, hist in histories.items():
        ax.plot(np.arange(len(hist)), hist, marker="o", markersize=3, label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Cost (so far)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    if own_fig:
        fig.tight_layout()
        return fig, ax
    return ax


def plot_metric_comparison(
    labels: Sequence[str],
    metric_values: dict[str, Sequence[float]],
    title: str = "Controller Performance Comparison",
    ylabel: str = "Value",
    ax: Optional[plt.Axes] = None,
):
    """Grouped bar chart comparing several metrics across several controllers.

    `labels` = controller names (x-axis groups), e.g. ["Baseline", "PSO", "ACO"]
    `metric_values` = {"Avg Wait": [...], "Avg Queue": [...], ...}, each list
        aligned with `labels`.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    n_groups = len(labels)
    n_metrics = len(metric_values)
    bar_width = 0.8 / max(n_metrics, 1)
    x = np.arange(n_groups)

    for i, (metric_name, values) in enumerate(metric_values.items()):
        offset = (i - (n_metrics - 1) / 2) * bar_width
        ax.bar(x + offset, values, width=bar_width, label=metric_name)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    if own_fig:
        fig.tight_layout()
        return fig, ax
    return ax


def plot_queue_trace(
    time_axis: np.ndarray,
    queue1: np.ndarray,
    queue2: np.ndarray,
    title: str = "Queue Length Over Time",
    ax: Optional[plt.Axes] = None,
):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(time_axis, queue1, label="Road 1 queue", linewidth=1.2)
    ax.plot(time_axis, queue2, label="Road 2 queue", linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue length (vehicles)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    if own_fig:
        fig.tight_layout()
        return fig, ax
    return ax


def plot_membership_functions(controller, title_prefix: str = ""):
    """Plot the pressure_self, pressure_other, and green_time membership
    functions of a FuzzyTrafficController instance (3-panel figure)."""
    mfs = controller.get_membership_functions()
    labels = ["Low", "Medium", "High"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for mf_arr, name in zip(mfs["pressure_self_mfs"], labels):
        axes[0].plot(mfs["pressure_universe"], mf_arr, label=name)
    axes[0].set_title(f"{title_prefix}Pressure (self)")
    axes[0].set_xlabel("pressure")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for mf_arr, name in zip(mfs["pressure_other_mfs"], labels):
        axes[1].plot(mfs["pressure_universe"], mf_arr, label=name)
    axes[1].set_title(f"{title_prefix}Pressure (other)")
    axes[1].set_xlabel("pressure")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    green_labels = ["Short", "Medium", "Long"]
    for mf_arr, name in zip(mfs["green_mfs"], green_labels):
        axes[2].plot(mfs["green_universe"], mf_arr, label=name)
    axes[2].set_title(f"{title_prefix}Green Duration")
    axes[2].set_xlabel("seconds")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    return fig, axes
