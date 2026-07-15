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



# Use non-interactive backend to avoid needing a display (for headless/servers)
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



# Create the output directory for saving all generated plots and results
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)



# Seeds used for training the optimizers (PSO and ACO) — each fitness evaluation
# averages over these seeds to reduce noise and improve generalization
TRAIN_SEEDS = [1, 2, 3]

# Seeds used for the final fair comparison of all three controllers (Baseline, PSO, ACO)
# — ensures the evaluation is statistically robust and not tied to a single random run
EVAL_SEEDS = [101, 102, 103, 104, 105]



# Shared simulation settings used across all experiments:
# - arrival_rate1/2: probability per time step that a vehicle arrives on each road
# - num_cycles: how many green-red cycles the simulation runs (length of episode)
SIM_CONFIG = SimulationConfig(arrival_rate1=0.35, arrival_rate2=0.25, num_cycles=60)

# Weight coefficients for the scalar cost function:
#   Cost = alpha * (avg_wait) + beta * (avg_queue) + gamma * (total_stops)
# These weights are normalised internally so the three terms contribute comparably.
WEIGHTS = CostWeights(alpha=1.0, beta=1.0, gamma=0.5)





def evaluate_controller_multi_seed(params: np.ndarray, seeds) -> dict:

    """
    Evaluate a given fuzzy-controller parameter vector over several random seeds
    and return the average of the raw performance metrics (not just the scalar cost).

    Parameters
    ----------
    params : np.ndarray
        Flattened parameter vector for the fuzzy controller (membership-function
        boundaries, rule weights, etc.).
    seeds : iterable of int
        List of random seeds to run the simulation with.

    Returns
    -------
    dict
        Dictionary containing:
        - averaged metrics (avg_wait_time, avg_queue_length, total_stops, cost, ...)
        - 'last_episode_result': the full result object from the final seed,
          kept for later plotting (e.g., queue traces).
    """

    # Build the fuzzy controller once with the given parameters
    controller = FuzzyTrafficController(params)
    breakdowns = []   # stores cost breakdown per seed
    last_result = None

    # Run the simulation for each seed and collect the cost breakdown
    for seed in seeds:
        # Create a fresh simulation config with the current seed; arrival rates
        # and number of cycles are taken from the global SIM_CONFIG
        cfg = SimulationConfig(
            arrival_rate1=SIM_CONFIG.arrival_rate1,
            arrival_rate2=SIM_CONFIG.arrival_rate2,
            num_cycles=SIM_CONFIG.num_cycles,
            max_queue=SIM_CONFIG.max_queue,
            seed=seed,
        )
        env = TrafficIntersection(cfg)
        # Run one full episode using the given controller
        result = env.run_episode(controller)
        # Decompose the episode result into the weighted cost components
        breakdowns.append(compute_cost_breakdown(result, WEIGHTS))
        last_result = result



    # Average all metrics over the seeds to get a stable estimate of performance
    avg = {key: float(np.mean([b[key] for b in breakdowns])) for key in breakdowns[0]}
    # Attach the full result of the last seed for later visualisation
    avg["last_episode_result"] = last_result
    return avg





def main():

    # Print a header so the console output is easy to read
    print("=" * 70)
    print("Fuzzy Traffic Light Controller: Baseline vs PSO vs ACO")
    print("=" * 70)



    # Create the fitness (cost) function that the optimizers will minimise.
    # It uses TRAIN_SEEDS to average out randomness and the global WEIGHTS
    # and SIM_CONFIG to keep the evaluation consistent.
    fitness_fn = partial(
        evaluate_params, seeds=TRAIN_SEEDS, weights=WEIGHTS, sim_config=SIM_CONFIG
    )



    # ---------------------------------------------------------------
    # 1) Baseline — hand‑tuned fuzzy controller (no optimisation)
    # ---------------------------------------------------------------
    print("\n[1/3] Evaluating baseline (hand-tuned) controller...")

    # Get the default, manually designed parameters for the fuzzy controller
    baseline_params = default_params()

    # Evaluate it over the EVAL_SEEDS to obtain fair, multi‑run metrics
    baseline_metrics = evaluate_controller_multi_seed(baseline_params, EVAL_SEEDS)
    print(
        f"  Baseline -> cost={baseline_metrics['cost']:.4f}, "
        f"wait={baseline_metrics['avg_wait_time']:.2f}, "
        f"queue={baseline_metrics['avg_queue_length']:.2f}, "
        f"stops={baseline_metrics['total_stops']:.1f}"
    )



    # ---------------------------------------------------------------
    # 2) PSO (Particle Swarm Optimisation) — tune the fuzzy params
    # ---------------------------------------------------------------
    print("\n[2/3] Running PSO optimization...")
    t0 = time.time()

    # Run PSO with 20 particles over 40 iterations; each fitness call
    # internally averages over TRAIN_SEEDS
    pso_result = run_pso(fitness_fn, PSOConfig(num_particles=20, num_iterations=40))
    print(
        f"  PSO finished in {time.time() - t0:.1f}s, "
        f"train-cost={pso_result.best_cost:.4f}"
    )

    # Evaluate the best PSO parameters over the independent EVAL_SEEDS
    pso_metrics = evaluate_controller_multi_seed(pso_result.best_params, EVAL_SEEDS)
    print(
        f"  PSO -> cost={pso_metrics['cost']:.4f}, "
        f"wait={pso_metrics['avg_wait_time']:.2f}, "
        f"queue={pso_metrics['avg_queue_length']:.2f}, "
        f"stops={pso_metrics['total_stops']:.1f}"
    )



    # ---------------------------------------------------------------
    # 3) ACO (Ant Colony Optimisation for continuous domains) — tune
    # ---------------------------------------------------------------
    print("\n[3/3] Running ACO (ACOR) optimization...")
    t0 = time.time()

    # Run ACO with 20 ants, 40 iterations, and an archive of 30 best solutions
    aco_result = run_aco(
        fitness_fn, ACOConfig(num_ants=20, num_iterations=40, archive_size=30)
    )
    print(
        f"  ACO finished in {time.time() - t0:.1f}s, "
        f"train-cost={aco_result.best_cost:.4f}"
    )

    # Evaluate the best ACO parameters on the independent EVAL_SEEDS
    aco_metrics = evaluate_controller_multi_seed(aco_result.best_params, EVAL_SEEDS)
    print(
        f"  ACO -> cost={aco_metrics['cost']:.4f}, "
        f"wait={aco_metrics['avg_wait_time']:.2f}, "
        f"queue={aco_metrics['avg_queue_length']:.2f}, "
        f"stops={aco_metrics['total_stops']:.1f}"
    )



    # ---------------------------------------------------------------
    # Generate all comparison plots and save them to OUTPUT_DIR
    # ---------------------------------------------------------------
    print("\nGenerating plots...")



    # ---- Convergence history: how training cost decreased over iterations ----
    fig, _ = plot_convergence(
        {"PSO": pso_result.convergence_history, "ACO": aco_result.convergence_history},
        title="PSO vs ACO Convergence (training cost)",
    )
    fig.savefig(os.path.join(OUTPUT_DIR, "convergence.png"), dpi=150)
    plt.close(fig)   # free memory



    # ---- Bar chart: Avg Wait & Avg Queue side by side for the 3 controllers ----
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



    # ---- Bar chart: Total Scalar Cost (the objective function) ----
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



    # ---- Queue-length time traces (one representative episode) for each controller ----
    for name, metrics in [
        ("baseline", baseline_metrics),
        ("pso", pso_metrics),
        ("aco", aco_metrics),
    ]:
        r = metrics["last_episode_result"]   # full result from the last eval seed
        fig, _ = plot_queue_trace(
            r.time_axis,
            r.queue1_history,
            r.queue2_history,
            title=f"Queue Trace - {name.upper()}",
        )
        fig.savefig(os.path.join(OUTPUT_DIR, f"queue_trace_{name}.png"), dpi=150)
        plt.close(fig)



    # ---- Membership-function plots: show the input/output fuzzy sets ----
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
    # Print a final summary table to the console for quick comparison
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



    # Save the best-found parameter vectors as NumPy files for later reuse
    np.save(os.path.join(OUTPUT_DIR, "pso_best_params.npy"), pso_result.best_params)
    np.save(os.path.join(OUTPUT_DIR, "aco_best_params.npy"), aco_result.best_params)





# Standard entry-point guard: only run main() if this script is executed directly
if __name__ == "__main__":
    main()
