"""
main.py
========

End-to-end driver: builds a Baseline (hand-tuned) fuzzy controller, then
optimizes the same parameter space with PSO and with ACO, runs all three
through the traffic simulation over multiple random seeds, and produces:

  - a convergence plot (PSO vs ACO)
  - a grouped bar chart comparing Avg Wait / Avg Queue / Stops / Cost
  - a queue-length time trace for each controller (single representative seed)
  - membership-function plots for baseline vs PSO vs ACO
  - a printed summary table

All figures are saved to ./outputs/.

Run with:  python main.py
"""

from __future__ import annotations

import os
import time
from functools import partial

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np

from src.aco import ACOConfig, run_aco
from src.cost import CostWeights, compute_cost_breakdown, evaluate_params
from src.fuzzy_controller import FuzzyTrafficController, default_params
from src.plots import (
    plot_convergence,
    plot_membership_functions,
    plot_metric_comparison,
    plot_queue_trace,
)
from src.pso import PSOConfig, run_pso
from src.simulation import SimulationConfig, TrafficIntersection

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Random seeds used both for optimizer fitness averaging AND for the final
# multi-seed evaluation of each resulting controller (statistical validity).
TRAIN_SEEDS = [1, 2, 3]
EVAL_SEEDS = [101, 102, 103, 104, 105]

SIM_CONFIG = SimulationConfig(arrival_rate1=0.35, arrival_rate2=0.25, num_cycles=60)
WEIGHTS = CostWeights(alpha=1.0, beta=1.0, gamma=0.5)


def evaluate_controller_multi_seed(params: np.ndarray, seeds) -> dict:
    """Run a parameter vector over several seeds and average the raw metrics
    (not just the scalar cost) for reporting purposes."""
    controller = FuzzyTrafficController(params)
    breakdowns = []
    last_result = None
    for seed in seeds:
        cfg = SimulationConfig(
            arrival_rate1=SIM_CONFIG.arrival_rate1,
            arrival_rate2=SIM_CONFIG.arrival_rate2,
            num_cycles=SIM_CONFIG.num_cycles,
            max_queue=SIM_CONFIG.max_queue,
            seed=seed,
        )
        env = TrafficIntersection(cfg)
        result = env.run_episode(controller)
        breakdowns.append(compute_cost_breakdown(result, WEIGHTS))
        last_result = result

    avg = {key: float(np.mean([b[key] for b in breakdowns])) for key in breakdowns[0]}
    avg["last_episode_result"] = last_result
    return avg


def main():
    print("=" * 70)
    print("Fuzzy Traffic Light Controller: Baseline vs PSO vs ACO")
    print("=" * 70)

    fitness_fn = partial(
        evaluate_params, seeds=TRAIN_SEEDS, weights=WEIGHTS, sim_config=SIM_CONFIG
    )

    # ---------------------------------------------------------------
    # 1) Baseline (manually tuned) controller
    # ---------------------------------------------------------------
    print("\n[1/3] Evaluating baseline (hand-tuned) controller...")
    baseline_params = default_params()
    baseline_metrics = evaluate_controller_multi_seed(baseline_params, EVAL_SEEDS)
    print(
        f"  Baseline -> cost={baseline_metrics['cost']:.4f}, "
        f"wait={baseline_metrics['avg_wait_time']:.2f}, "
        f"queue={baseline_metrics['avg_queue_length']:.2f}, "
        f"stops={baseline_metrics['total_stops']:.1f}"
    )

    # ---------------------------------------------------------------
    # 2) PSO optimization
    # ---------------------------------------------------------------
    print("\n[2/3] Running PSO optimization...")
    t0 = time.time()
    pso_result = run_pso(fitness_fn, PSOConfig(num_particles=20, num_iterations=40))
    print(
        f"  PSO finished in {time.time() - t0:.1f}s, "
        f"train-cost={pso_result.best_cost:.4f}"
    )
    pso_metrics = evaluate_controller_multi_seed(pso_result.best_params, EVAL_SEEDS)
    print(
        f"  PSO -> cost={pso_metrics['cost']:.4f}, "
        f"wait={pso_metrics['avg_wait_time']:.2f}, "
        f"queue={pso_metrics['avg_queue_length']:.2f}, "
        f"stops={pso_metrics['total_stops']:.1f}"
    )

    # ---------------------------------------------------------------
    # 3) ACO optimization
    # ---------------------------------------------------------------
    print("\n[3/3] Running ACO (ACOR) optimization...")
    t0 = time.time()
    aco_result = run_aco(
        fitness_fn, ACOConfig(num_ants=20, num_iterations=40, archive_size=30)
    )
    print(
        f"  ACO finished in {time.time() - t0:.1f}s, "
        f"train-cost={aco_result.best_cost:.4f}"
    )
    aco_metrics = evaluate_controller_multi_seed(aco_result.best_params, EVAL_SEEDS)
    print(
        f"  ACO -> cost={aco_metrics['cost']:.4f}, "
        f"wait={aco_metrics['avg_wait_time']:.2f}, "
        f"queue={aco_metrics['avg_queue_length']:.2f}, "
        f"stops={aco_metrics['total_stops']:.1f}"
    )

    # ---------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------
    print("\nGenerating plots...")

    fig, _ = plot_convergence(
        {"PSO": pso_result.convergence_history, "ACO": aco_result.convergence_history},
        title="PSO vs ACO Convergence (training cost)",
    )
    fig.savefig(os.path.join(OUTPUT_DIR, "convergence.png"), dpi=150)
    plt.close(fig)

    labels = ["Baseline", "PSO", "ACO"]
    fig, _ = plot_metric_comparison(
        labels,
        {
            "Avg Wait": [
                baseline_metrics["avg_wait_time"],
                pso_metrics["avg_wait_time"],
                aco_metrics["avg_wait_time"],
            ],
            "Avg Queue": [
                baseline_metrics["avg_queue_length"],
                pso_metrics["avg_queue_length"],
                aco_metrics["avg_queue_length"],
            ],
        },
        title="Wait Time & Queue Length Comparison",
        ylabel="Value (raw units)",
    )
    fig.savefig(os.path.join(OUTPUT_DIR, "metric_comparison.png"), dpi=150)
    plt.close(fig)

    fig, _ = plot_metric_comparison(
        labels,
        {
            "Total Cost": [
                baseline_metrics["cost"],
                pso_metrics["cost"],
                aco_metrics["cost"],
            ]
        },
        title="Scalar Cost Comparison (lower is better)",
        ylabel="Cost C = a*W + b*Q + g*S (normalized)",
    )
    fig.savefig(os.path.join(OUTPUT_DIR, "cost_comparison.png"), dpi=150)
    plt.close(fig)

    for name, metrics in [
        ("baseline", baseline_metrics),
        ("pso", pso_metrics),
        ("aco", aco_metrics),
    ]:
        r = metrics["last_episode_result"]
        fig, _ = plot_queue_trace(
            r.time_axis,
            r.queue1_history,
            r.queue2_history,
            title=f"Queue Trace - {name.upper()}",
        )
        fig.savefig(os.path.join(OUTPUT_DIR, f"queue_trace_{name}.png"), dpi=150)
        plt.close(fig)

    for name, params in [
        ("baseline", baseline_params),
        ("pso", pso_result.best_params),
        ("aco", aco_result.best_params),
    ]:
        controller = FuzzyTrafficController(params)
        fig, _ = plot_membership_functions(
            controller, title_prefix=f"{name.upper()} - "
        )
        fig.savefig(os.path.join(OUTPUT_DIR, f"membership_{name}.png"), dpi=150)
        plt.close(fig)

    # ---------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'Controller':<12}{'Cost':>10}{'AvgWait':>12}{'AvgQueue':>12}{'Stops':>10}")
    print("-" * 70)
    for name, m in [
        ("Baseline", baseline_metrics),
        ("PSO", pso_metrics),
        ("ACO", aco_metrics),
    ]:
        print(
            f"{name:<12}{m['cost']:>10.4f}{m['avg_wait_time']:>12.2f}"
            f"{m['avg_queue_length']:>12.2f}{m['total_stops']:>10.1f}"
        )
    print("=" * 70)
    print(f"\nAll plots saved to: {OUTPUT_DIR}")

    np.save(os.path.join(OUTPUT_DIR, "pso_best_params.npy"), pso_result.best_params)
    np.save(os.path.join(OUTPUT_DIR, "aco_best_params.npy"), aco_result.best_params)


if __name__ == "__main__":
    main()
