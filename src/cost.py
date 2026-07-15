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
# be exact -- they just need to bring W, Q, S onto comparable footing
# so that alpha=beta=gamma=1 gives roughly equal importance to all three terms.
REF_WAIT = 15.0   # typical average wait time per vehicle (seconds) in a baseline run
REF_QUEUE = 15.0  # typical average queue length (vehicles) in a baseline run
REF_STOPS = 40.0  # typical total number of vehicle stops over a full episode


@dataclass
class CostWeights:
    """Weights that define the relative importance of each objective in the scalar cost function.
    
    The cost is: C = alpha * (W/REF_WAIT) + beta * (Q/REF_QUEUE) + gamma * (S/REF_STOPS)
    where W = average wait time, Q = average queue length, S = total stops.
    """
    alpha: float = 1.0  # weight on average wait time — higher alpha penalizes waiting more
    beta: float = 1.0   # weight on average queue length — higher beta penalizes long queues more
    gamma: float = 0.5  # weight on number of stops — higher gamma penalizes frequent stopping more


def compute_cost(result: EpisodeResult, weights: CostWeights = CostWeights()) -> float:
    """Compute the scalar cost C = alpha*W + beta*Q + gamma*S for a single
    simulated episode, using fixed reference-scale normalization.
    
    The normalization divides each raw metric by its reference value so that
    all three terms have roughly the same order of magnitude. Without this,
    a term with larger raw numbers (e.g., total_stops ≈ 40) would dominate
    a term with smaller numbers (e.g., avg_wait_time ≈ 15) even when the
    weights are equal.

    Parameters
    ----------
    result : EpisodeResult
        The result of one traffic simulation episode, containing raw metrics
        like avg_wait_time, avg_queue_length, and total_stops.
    weights : CostWeights
        The alpha, beta, gamma coefficients that control the trade-off
        between wait time, queue length, and stops.

    Returns
    -------
    float
        The scalar cost (lower is better), normalized and weighted.
    """
    # Normalize each raw metric by its reference scale to bring them to comparable magnitude
    w_norm = result.avg_wait_time / REF_WAIT      # normalized wait time (dimensionless)
    q_norm = result.avg_queue_length / REF_QUEUE  # normalized queue length (dimensionless)
    s_norm = result.total_stops / REF_STOPS       # normalized stops (dimensionless)

    # Compute the weighted sum: C = alpha * w_norm + beta * q_norm + gamma * s_norm
    return float(
        weights.alpha * w_norm + weights.beta * q_norm + weights.gamma * s_norm
    )


def compute_cost_breakdown(
    result: EpisodeResult, weights: CostWeights = CostWeights()
) -> dict:
    """Same as compute_cost but also returns the raw, un-normalized metrics,
    useful for reporting/plotting in main.py.
    
    This gives a complete picture: the scalar cost (used for optimization)
    plus the actual physical quantities (used for human-readable comparison).

    Parameters
    ----------
    result : EpisodeResult
        The result of one traffic simulation episode.
    weights : CostWeights
        The alpha, beta, gamma coefficients.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'cost': the scalar cost (normalized + weighted)
        - 'avg_wait_time': raw average wait time per vehicle
        - 'avg_queue_length': raw average queue length
        - 'total_stops': raw total number of stops
    """
    return {
        "cost": compute_cost(result, weights),        # the objective function value
        "avg_wait_time": result.avg_wait_time,         # raw metric: how long vehicles wait on average
        "avg_queue_length": result.avg_queue_length,   # raw metric: how many vehicles are queued on average
        "total_stops": result.total_stops,             # raw metric: how many times vehicles had to stop
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

    Parameters
    ----------
    params : np.ndarray
        A 24-D parameter vector encoding the fuzzy controller's membership
        function boundaries and rule weights (see fuzzy_controller.py).
    seeds : Sequence[int]
        List of random seeds to evaluate on. The function runs one full
        simulation episode per seed and averages the costs. More seeds =
        less noisy fitness estimate, but slower per evaluation.
    weights : CostWeights
        The alpha, beta, gamma weights for the cost function.
    sim_config : SimulationConfig or None
        Base simulation configuration (arrival rates, number of cycles, etc.).
        If None, default settings are used.

    Returns
    -------
    float
        The mean scalar cost across all seeds. Lower = better traffic
        performance. Returns a large penalty (1e6) if the parameter vector
        is invalid (e.g., violates ordering constraints of membership
        function breakpoints).
    """
    try:
        # Attempt to build the fuzzy controller from the given parameter vector.
        # This will raise an exception if the parameters are invalid
        # (e.g., membership function breakpoints are out of order).
        controller = FuzzyTrafficController(params)
    except Exception:
        # Any malformed parameter vector (e.g. produced by an overly
        # aggressive mutation) is simply penalized rather than crashing
        # the whole optimization run. Returning a very large cost ensures
        # this candidate will never be selected as "best".
        return 1e6

    # Use the provided simulation config, or fall back to defaults
    base_config = sim_config or SimulationConfig()
    costs = []  # will store the cost for each seed

    # Evaluate the controller on each random seed to get a robust fitness estimate
    for seed in seeds:
        # Build a simulation config with the current seed; arrival rates,
        # number of cycles, etc. are taken from the base config
        cfg = SimulationConfig(
            arrival_rate1=base_config.arrival_rate1,  # probability of new car on road 1 per time step
            arrival_rate2=base_config.arrival_rate2,  # probability of new car on road 2 per time step
            num_cycles=base_config.num_cycles,        # how many green-red cycles to simulate
            max_queue=base_config.max_queue,          # maximum queue capacity before overflow
            seed=seed,                                # random seed for this specific run
        )
        # Create the traffic intersection environment with these settings
        env = TrafficIntersection(cfg)
        # Run one full episode: the controller decides green-light duration
        # at each cycle based on queue lengths, and vehicles arrive/leave
        result = env.run_episode(controller)
        # Compute the scalar cost for this episode
        costs.append(compute_cost(result, weights))

    # Return the average cost over all seeds — this smooths out randomness
    # and gives a more reliable estimate of the controller's true performance
    return float(np.mean(costs))