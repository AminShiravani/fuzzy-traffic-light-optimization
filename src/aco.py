
"""
aco.py
=======

Ant Colony Optimization for tuning the fuzzy controller's 24-dimensional
parameter vector (see fuzzy_controller.py for the encoding).

WHY ACOR (continuous ACO) AND NOT "TEXTBOOK" DISCRETE ACO
------------------------------------------------------------
Classical Ant Colony Optimization was designed for problems with a discrete
graph of choices (e.g. which city to visit next), where pheromone lives on
edges. Our search space -- membership-function breakpoints and rule weights
-- is continuous, so there is no natural graph to lay pheromone on.

We therefore use ACOR (Ant Colony Optimization for continuous domains,
Socha & Dorigo, 2008), which is the standard, well-established adaptation
of ACO to continuous parameter search and is what most "ACO for continuous
optimization" implementations mean in practice:

  - Maintain a SOLUTION ARCHIVE of the k best parameter vectors found so
    far (this archive plays the role of "pheromone": good regions of the
    search space are reinforced by being remembered, and are queried more
    often as `rank` improves, mirroring Eq. 4 in the project brief where
    better/more-recently-reinforced paths (tau_ij) get chosen more often).
  - Each ant builds a new solution GENE-BY-GENE. For each gene, the ant
    picks one archive member as a "seed" (weighted by rank -- better
    solutions are picked more often, i.e. more pheromone) and then samples
    a Gaussian around that seed's value for that gene. The Gaussian's
    standard deviation is proportional to how spread out the archive is on
    that gene (this gives the "pheromone evaporation" effect: as the
    archive converges, exploration naturally shrinks).
  - New solutions are evaluated and merged into the archive, keeping only
    the best `archive_size` overall (evaporation = discarding the worst).

This preserves the two defining ACO ideas -- stigmergic reinforcement of
good paths and probabilistic, exploration/exploitation-balanced solution
construction -- while remaining tractable on a continuous 24-D space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .fuzzy_controller import PARAM_DIM, default_params, param_bounds


@dataclass
class ACOConfig:
    """Configuration parameters for the ACOR (Ant Colony Optimization for continuous domains) algorithm.
    
    These control how many ants explore per iteration, how long to run, how many
    top solutions to keep in the pheromone archive, and how exploration is balanced.
    """
    num_ants: int = 20  # new solutions constructed per iteration — each ant builds one complete parameter vector
    num_iterations: int = 40  # how many generations/iterations the algorithm runs for
    archive_size: int = 30  # k: number of best solutions kept in the archive as "pheromone memory"
    q: float = 0.3  # locality of search — smaller q makes the selection more greedy, favoring top-ranked solutions strongly
    xi: float = 0.85  # evaporation-like spread/convergence speed — higher xi keeps more exploration, lower xi converges faster
    seed: Optional[int] = 7  # random seed for reproducibility (None = truly random each run)


@dataclass
class ACOResult:
    """Container for the results returned by the ACOR optimization.
    
    Holds the best parameter vector found, its cost, and the convergence history
    (cost of the best solution at each iteration) for plotting.
    """
    best_params: np.ndarray  # the best 24-D parameter vector for the fuzzy controller found during optimization
    best_cost: float  # the scalar cost achieved by best_params (lower = better traffic flow)
    convergence_history: np.ndarray  # array of best cost at each iteration, used to plot convergence curve


def run_aco(
    fitness_fn: Callable[[np.ndarray], float],
    config: ACOConfig = ACOConfig(),
) -> ACOResult:
    """
    Run ACOR (Ant Colony Optimization for continuous domains) to minimize
    `fitness_fn(params) -> float` over the fuzzy controller parameter space.
    
    The fitness function takes a parameter vector and returns a scalar cost
    (lower is better), which the algorithm tries to minimize.

    Parameters
    ----------
    fitness_fn : callable
        Function that receives a 24-D parameter vector and returns a scalar cost.
        This is typically a wrapper that runs the traffic simulation and computes
        the weighted sum of wait time, queue length, and stops.
    config : ACOConfig, optional
        Configuration for the ACOR algorithm (number of ants, iterations, etc.).

    Returns
    -------
    ACOResult
        Object containing the best parameter vector found, its cost, and the
        convergence history for plotting.
    """
    # Create a random number generator with the given seed for reproducibility
    rng = np.random.default_rng(config.seed)
    
    # Get the allowed minimum/maximum bounds for each of the 24 parameters
    # bounds shape: (24, 2) -> [lower_bounds, upper_bounds]
    bounds = param_bounds()
    lo, hi = bounds[:, 0], bounds[:, 1]  # lo = array of lower bounds, hi = array of upper bounds
    span = hi - lo  # range (width) of each parameter's allowed values

    k = config.archive_size  # number of solutions kept in the pheromone archive

    # --- initialize archive with random solutions plus the hand-tuned baseline -----------
    # Create k random solutions within bounds, then overwrite the first one with
    # the default (hand-tuned) parameters to give the optimizer a good starting point
    archive = lo + rng.random((k, PARAM_DIM)) * span  # random solutions in [lo, hi]
    archive[0] = default_params()  # seed the archive with the manually designed parameters
    # Evaluate fitness for every initial archive member
    archive_cost = np.array([fitness_fn(sol) for sol in archive])

    # Sort archive from best (lowest cost) to worst
    order = np.argsort(archive_cost)
    archive = archive[order]  # reorder solutions so archive[0] = best
    archive_cost = archive_cost[order]  # reorder costs accordingly

    # Compute rank-based selection weights for each archive position.
    # Solutions with better rank (lower index = lower cost) get higher weight,
    # meaning they are more likely to be chosen as "seeds" when ants build new solutions.
    # This is the ACOR Gaussian-kernel weighting scheme from Socha & Dorigo (2008).
    ranks = np.arange(1, k + 1)  # rank 1 = best, rank k = worst
    weights = (1.0 / (config.q * k * np.sqrt(2 * np.pi))) * np.exp(
        -((ranks - 1) ** 2) / (2 * (config.q * k) ** 2)
    )
    weights /= weights.sum()  # normalize so weights sum to 1 (valid probability distribution)

    # Record the best cost from the initial archive as iteration 0
    convergence = [float(archive_cost[0])]

    # Main optimization loop: each iteration = one "generation" of ants
    for _ in range(config.num_iterations):
        # Pre-allocate array to store the new solutions built by ants this iteration
        new_solutions = np.zeros((config.num_ants, PARAM_DIM))

        # Each ant constructs a complete new parameter vector
        for ant in range(config.num_ants):
            # Build the solution one gene (parameter) at a time
            for gene in range(PARAM_DIM):
                # 1) Choose a "guiding" archive member for this gene via
                #    roulette-wheel selection on rank weights.
                #    Better-ranked solutions (lower cost) have higher weight,
                #    mimicking the effect of stronger pheromone trails.
                chosen = rng.choice(k, p=weights)

                # 2) Compute standard deviation (sigma) for the Gaussian sampling.
                #    Sigma = xi * average absolute distance from the chosen solution's
                #    gene value to all other archive members' gene values.
                #    When archive members are spread out (diverse), sigma is large
                #    (more exploration). When they converge, sigma shrinks naturally
                #    (less exploration, more exploitation). This is the "evaporation"
                #    analogue in continuous ACO.
                sigma = config.xi * np.mean(
                    np.abs(archive[:, gene] - archive[chosen, gene])
                )
                sigma = max(sigma, 1e-6)  # avoid a fully collapsed distribution (zero sigma would break sampling)

                # Sample a new value from a Gaussian centered on the chosen archive
                # member's gene value, with the computed sigma
                sample = rng.normal(loc=archive[chosen, gene], scale=sigma)
                # Clip to the allowed bounds for this parameter
                new_solutions[ant, gene] = np.clip(sample, lo[gene], hi[gene])

        # Evaluate fitness for all new solutions built by the ants
        new_costs = np.array([fitness_fn(sol) for sol in new_solutions])

        # Merge new solutions into the archive, evaporate (discard) the worst ones,
        # and keep only the best k solutions overall. This is analogous to pheromone
        # evaporation in discrete ACO: old, poor solutions are forgotten.
        combined = np.vstack([archive, new_solutions])  # stack old archive + new solutions
        combined_cost = np.concatenate([archive_cost, new_costs])  # stack their costs
        order = np.argsort(combined_cost)[:k]  # indices of the k best (lowest cost)
        archive = combined[order]  # keep only the k best solutions
        archive_cost = combined_cost[order]  # keep their costs

        # Record the best cost found so far (archive is sorted, so index 0 is best)
        convergence.append(float(archive_cost[0]))

    # Return the optimization result with the best solution found
    return ACOResult(
        best_params=archive[0].copy(),  # copy to avoid mutation issues
        best_cost=float(archive_cost[0]),  # scalar cost of the best solution
        convergence_history=np.array(convergence),  # cost history for plotting
    )
