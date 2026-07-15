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

    This shows how the objective function (scalar cost) improves over
    iterations, allowing visual comparison of PSO vs ACO convergence speed
    and final solution quality.

    `histories` maps a label (e.g. "PSO", "ACO") to a 1-D convergence array
    where each element is the best cost found up to that iteration.

    Parameters
    ----------
    histories : dict
        Keys are algorithm names, values are 1-D arrays of best cost per iteration.
    title : str
        Plot title.
    ax : matplotlib Axes or None
        If provided, plot on this axis; otherwise create a new figure.

    Returns
    -------
    fig, ax (if no ax was provided) or just ax
    """
    # Check if we need to create our own figure or use an existing axis
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    # Plot each optimizer's convergence history as a line with markers
    for label, hist in histories.items():
        ax.plot(np.arange(len(hist)), hist, marker="o", markersize=3, label=label)

    ax.set_xlabel("Iteration")          # x-axis: optimization iteration number
    ax.set_ylabel("Best Cost (so far)")  # y-axis: lowest cost found up to that iteration (lower = better)
    ax.set_title(title)
    ax.legend()                          # show which line is PSO vs ACO
    ax.grid(alpha=0.3)                   # light grid for readability

    if own_fig:
        fig.tight_layout()  # prevent label clipping
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

    This creates a side-by-side bar chart where each group on the x-axis is
    a controller (e.g., Baseline, PSO, ACO) and bars within each group
    represent different performance metrics (e.g., Avg Wait, Avg Queue).

    `labels` = controller names (x-axis groups), e.g. ["Baseline", "PSO", "ACO"]
    `metric_values` = {"Avg Wait": [...], "Avg Queue": [...], ...}, each list
        aligned with `labels`. For example:
        {"Avg Wait": [15.2, 12.1, 11.8], "Avg Queue": [10.3, 8.5, 7.9]}

    Parameters
    ----------
    labels : sequence of str
        Names of the controllers being compared (x-axis group labels).
    metric_values : dict
        Keys are metric names, values are lists of numbers (one per controller).
    title : str
        Plot title.
    ylabel : str
        Label for the y-axis.
    ax : matplotlib Axes or None
        If provided, plot on this axis; otherwise create a new figure.

    Returns
    -------
    fig, ax (if no ax was provided) or just ax
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    n_groups = len(labels)                         # how many controllers to compare
    n_metrics = len(metric_values)                  # how many metrics per controller
    bar_width = 0.8 / max(n_metrics, 1)            # width of each bar (fits all bars in one group)
    x = np.arange(n_groups)                         # x positions for groups: [0, 1, 2, ...]

    # Draw bars for each metric, offsetting them so they appear side-by-side
    for i, (metric_name, values) in enumerate(metric_values.items()):
        # Center the bars within each group: bar i is offset from the group center
        offset = (i - (n_metrics - 1) / 2) * bar_width
        ax.bar(x + offset, values, width=bar_width, label=metric_name)

    ax.set_xticks(x)                    # tick marks at group centers
    ax.set_xticklabels(labels)          # label each group with the controller name
    ax.set_ylabel(ylabel)               # e.g., "Value (raw units)" or "Cost"
    ax.set_title(title)
    ax.legend()                         # show which bar color = which metric
    ax.grid(alpha=0.3, axis="y")        # horizontal grid lines only

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
    """Plot the queue length on each road over the course of a simulation episode.

    This shows how the controller manages the two competing queues over time:
    - When queue1 spikes, the controller should give road 1 more green time
    - When queue2 spikes, road 2 should get priority
    - A good controller keeps both queues balanced and prevents unbounded growth

    Parameters
    ----------
    time_axis : np.ndarray
        Time values (in seconds) for the x-axis.
    queue1 : np.ndarray
        Queue length (vehicles) on road 1 at each time step.
    queue2 : np.ndarray
        Queue length (vehicles) on road 2 at each time step.
    title : str
        Plot title.
    ax : matplotlib Axes or None
        If provided, plot on this axis; otherwise create a new figure.

    Returns
    -------
    fig, ax (if no ax was provided) or just ax
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 4))

    # Plot both queue traces as line plots over time
    ax.plot(time_axis, queue1, label="Road 1 queue", linewidth=1.2)  # road 1 in default blue
    ax.plot(time_axis, queue2, label="Road 2 queue", linewidth=1.2)  # road 2 in default orange
    ax.set_xlabel("Time (s)")                         # x-axis: simulation time in seconds
    ax.set_ylabel("Queue length (vehicles)")           # y-axis: how many vehicles are waiting
    ax.set_title(title)
    ax.legend()                                        # distinguish road 1 vs road 2
    ax.grid(alpha=0.3)                                 # light grid for readability

    if own_fig:
        fig.tight_layout()
        return fig, ax
    return ax


def plot_membership_functions(controller, title_prefix: str = ""):
    """Plot the pressure_self, pressure_other, and green_time membership
    functions of a FuzzyTrafficController instance (3-panel figure).

    This creates one row of three subplots showing:
    - Left:   Membership functions for "pressure on my road" (Low / Medium / High)
    - Middle: Membership functions for "pressure on other road" (Low / Medium / High)
    - Right:  Membership functions for "green light duration" (Short / Medium / Long)

    This is useful for seeing how PSO/ACO have reshaped the fuzzy sets
    compared to the hand-tuned baseline. For example, the optimizer might
    shift the "High pressure" threshold or widen the "Medium green" range.

    Parameters
    ----------
    controller : FuzzyTrafficController
        A controller instance with pre-built membership functions (call
        controller.get_membership_functions() to extract them).
    title_prefix : str
        Prefix added to each subplot title (e.g., "BASELINE - " or "PSO - ").

    Returns
    -------
    fig, axes : the matplotlib Figure and array of 3 Axes objects
    """
    # Extract the membership function data from the controller
    mfs = controller.get_membership_functions()
    labels = ["Low", "Medium", "High"]  # linguistic terms for pressure inputs

    # Create a 1x3 grid of subplots (side by side)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ---- Left panel: Pressure (self) membership functions ----
    # Plot each of the three MFs (Low, Medium, High) on the pressure universe
    for mf_arr, name in zip(mfs["pressure_self_mfs"], labels):
        axes[0].plot(mfs["pressure_universe"], mf_arr, label=name)
    axes[0].set_title(f"{title_prefix}Pressure (self)")  # e.g. "BASELINE - Pressure (self)"
    axes[0].set_xlabel("pressure")                        # x-axis: pressure value (queue + lambda*arrival)
    axes[0].legend()                                      # show which curve is Low/Medium/High
    axes[0].grid(alpha=0.3)

    # ---- Middle panel: Pressure (other) membership functions ----
    for mf_arr, name in zip(mfs["pressure_other_mfs"], labels):
        axes[1].plot(mfs["pressure_universe"], mf_arr, label=name)
    axes[1].set_title(f"{title_prefix}Pressure (other)")
    axes[1].set_xlabel("pressure")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # ---- Right panel: Green Duration membership functions ----
    # Note: different labels because the output is "Short/Medium/Long" green time
    green_labels = ["Short", "Medium", "Long"]
    for mf_arr, name in zip(mfs["green_mfs"], green_labels):
        axes[2].plot(mfs["green_universe"], mf_arr, label=name)
    axes[2].set_title(f"{title_prefix}Green Duration")  # e.g. "PSO - Green Duration"
    axes[2].set_xlabel("seconds")                        # x-axis: green light duration in seconds
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()  # adjust spacing so labels don't overlap
    return fig, axes