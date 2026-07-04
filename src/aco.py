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
    num_ants: int = 20  # new solutions constructed per iteration
    num_iterations: int = 40
    archive_size: int = 30  # k: number of solutions kept as "pheromone"
    q: float = 0.3  # locality of search (smaller -> favors best-ranked archive members)
    xi: float = 0.85  # evaporation-like spread/convergence speed
    seed: Optional[int] = 7


@dataclass
class ACOResult:
    best_params: np.ndarray
    best_cost: float
    convergence_history: np.ndarray


def run_aco(
    fitness_fn: Callable[[np.ndarray], float],
    config: ACOConfig = ACOConfig(),
) -> ACOResult:
    """
    Run ACOR to minimize `fitness_fn(params) -> float`.
    """
    rng = np.random.default_rng(config.seed)
    bounds = param_bounds()
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo

    k = config.archive_size

    # --- initialize archive with random + baseline solutions -----------
    archive = lo + rng.random((k, PARAM_DIM)) * span
    archive[0] = default_params()
    archive_cost = np.array([fitness_fn(sol) for sol in archive])

    order = np.argsort(archive_cost)
    archive = archive[order]
    archive_cost = archive_cost[order]

    # Rank-based selection weights (solution rank 1 = best gets highest
    # weight). This is the ACOR Gaussian-kernel weighting scheme.
    ranks = np.arange(1, k + 1)
    weights = (1.0 / (config.q * k * np.sqrt(2 * np.pi))) * np.exp(
        -((ranks - 1) ** 2) / (2 * (config.q * k) ** 2)
    )
    weights /= weights.sum()

    convergence = [float(archive_cost[0])]

    for _ in range(config.num_iterations):
        new_solutions = np.zeros((config.num_ants, PARAM_DIM))

        for ant in range(config.num_ants):
            for gene in range(PARAM_DIM):
                # 1) choose a "guiding" archive member for this gene via
                #    roulette-wheel selection on rank weights (pheromone-
                #    like preference for better solutions)
                chosen = rng.choice(k, p=weights)

                # 2) standard deviation = xi * average distance from the
                #    chosen solution's gene value to all other archive
                #    members' gene values (this shrinks automatically as
                #    the archive converges -> implicit "evaporation")
                sigma = config.xi * np.mean(
                    np.abs(archive[:, gene] - archive[chosen, gene])
                )
                sigma = max(sigma, 1e-6)  # avoid a fully collapsed distribution

                sample = rng.normal(loc=archive[chosen, gene], scale=sigma)
                new_solutions[ant, gene] = np.clip(sample, lo[gene], hi[gene])

        new_costs = np.array([fitness_fn(sol) for sol in new_solutions])

        # Merge new ants into the archive, evaporate (drop) the worst,
        # keep only the best `k` overall.
        combined = np.vstack([archive, new_solutions])
        combined_cost = np.concatenate([archive_cost, new_costs])
        order = np.argsort(combined_cost)[:k]
        archive = combined[order]
        archive_cost = combined_cost[order]

        convergence.append(float(archive_cost[0]))

    return ACOResult(
        best_params=archive[0].copy(),
        best_cost=float(archive_cost[0]),
        convergence_history=np.array(convergence),
    )
