"""
cost.py
========

Implements the cost function from the project brief (Section 3, Eq. 1):

    C = alpha * W + beta * Q + gamma * S

where W = average wait time, Q = average queue length, S = number of stops.

Because W, Q, S live on different numeric scales (S can be in the hundreds
over an episode while W/Q are per-second averages of a handful of
vehicles), we normalize each term by a fixed reference scale BEFORE
weighting, so alpha/beta/gamma behave as genuine relative-importance knobs
rather than being dominated by whichever raw term happens to be numerically
largest. This normalization is a standard practice for multi-objective
scalarization and does not change *what* is being optimized, only how the
weights are interpreted.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Optional, Sequence

import numpy as np

from .fuzzy_controller import FuzzyTrafficController
from .simulation import EpisodeResult, SimulationConfig, TrafficIntersection

# Reference scales used purely for normalization (rough orders of magnitude
# observed from an unoptimized baseline controller). These do not need to
# be exact -- they just need to bring W, Q, S onto comparable footing.
REF_WAIT = 15.0
REF_QUEUE = 15.0
REF_STOPS = 40.0


@dataclass
class CostWeights:
    alpha: float = 1.0  # weight on average wait time
    beta: float = 1.0  # weight on average queue length
    gamma: float = 0.5  # weight on number of stops


def compute_cost(result: EpisodeResult, weights: CostWeights = CostWeights()) -> float:
    """Compute the scalar cost C = alpha*W + beta*Q + gamma*S for a single
    simulated episode, using fixed reference-scale normalization."""
    w_norm = result.avg_wait_time / REF_WAIT
    q_norm = result.avg_queue_length / REF_QUEUE
    s_norm = result.total_stops / REF_STOPS

    return float(
        weights.alpha * w_norm + weights.beta * q_norm + weights.gamma * s_norm
    )


def compute_cost_breakdown(
    result: EpisodeResult, weights: CostWeights = CostWeights()
) -> dict:
    """Same as compute_cost but also returns the raw, un-normalized metrics,
    useful for reporting/plotting in main.py."""
    return {
        "cost": compute_cost(result, weights),
        "avg_wait_time": result.avg_wait_time,
        "avg_queue_length": result.avg_queue_length,
        "total_stops": result.total_stops,
    }


def evaluate_params(
    params: np.ndarray,
    seeds: Sequence[int],
    weights: CostWeights = CostWeights(),
    sim_config: Optional[SimulationConfig] = None,
) -> float:
    """
    THE fitness function shared by pso.py and aco.py.

    Builds a FuzzyTrafficController from `params`, runs one simulated
    episode per seed in `seeds`, and returns the mean cost across seeds.

    Averaging over multiple random seeds is important here: traffic
    arrivals are stochastic, so evaluating a candidate on a single seed
    would make the optimizer chase noise rather than genuine parameter
    quality. This is the standard "multiple replications" approach for
    optimizing under stochastic simulation.
    """
    try:
        controller = FuzzyTrafficController(params)
    except Exception:
        # Any malformed parameter vector (e.g. produced by an overly
        # aggressive mutation) is simply penalized rather than crashing
        # the whole optimization run.
        return 1e6

    base_config = sim_config or SimulationConfig()
    costs = []
    for seed in seeds:
        cfg = SimulationConfig(
            arrival_rate1=base_config.arrival_rate1,
            arrival_rate2=base_config.arrival_rate2,
            num_cycles=base_config.num_cycles,
            max_queue=base_config.max_queue,
            seed=seed,
        )
        env = TrafficIntersection(cfg)
        result = env.run_episode(controller)
        costs.append(compute_cost(result, weights))

    return float(np.mean(costs))
