"""
pso.py
=======

Particle Swarm Optimization (PSO) for tuning the fuzzy controller's
membership-function breakpoints and rule weights (see fuzzy_controller.py
for the full explanation of the 24-dimensional parameter encoding).

Each PARTICLE is simply one 24-length parameter vector -- i.e. one complete
candidate fuzzy controller. The swarm explores this 24-D continuous space
directly using the canonical velocity/position update equations from the
project brief (Eq. 2, 3):

    v_i(t+1) = w * v_i(t) + c1*r1*(p_i - x_i(t)) + c2*r2*(g - x_i(t))
    x_i(t+1) = x_i(t) + v_i(t+1)

where:
  - v_i(t)    : velocity of particle i at iteration t (how fast and in what direction it moves)
  - x_i(t)    : position of particle i (a 24-D parameter vector = candidate controller)
  - w         : inertia weight (keeps particles moving in the same direction)
  - c1        : cognitive coefficient (how much the particle trusts its own best experience)
  - r1        : random number [0,1] for stochastic exploration
  - p_i       : personal best position of particle i (best params it has ever found)
  - c2        : social coefficient (how much the particle trusts the swarm's global best)
  - r2        : random number [0,1] for stochastic exploration
  - g         : global best position (best params ANY particle has ever found)

Bounds handling: after the position update, each gene is clipped back into
its valid range (from `fuzzy_controller.param_bounds()`), and any velocity
that would push a gene out of bounds is zeroed (a standard "absorbing wall"
strategy) to avoid particles permanently flying off into invalid regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .fuzzy_controller import PARAM_DIM, default_params, param_bounds


@dataclass
class PSOConfig:
    """Configuration parameters for the Particle Swarm Optimization algorithm.
    
    These control how many particles explore the space, how long the algorithm
    runs, and the balance between exploration (searching new areas) and
    exploitation (refining known good areas).
    """
    num_particles: int = 20      # swarm size — more particles = better exploration but slower per iteration
    num_iterations: int = 40     # how many generations the swarm evolves for
    inertia_w: float = 0.72      # initial inertia weight — keeps particles moving; decays over time to shift from exploration to exploitation
    cognitive_c1: float = 1.49   # cognitive (personal best) acceleration coefficient — how strongly each particle is pulled toward its own best position
    social_c2: float = 1.49      # social (global best) acceleration coefficient — how strongly each particle is pulled toward the swarm's best position
    inertia_decay: float = (
        0.99  # multiplies inertia_w each iteration (helps convergence — gradually reduces exploration)
    )
    seed: Optional[int] = 42     # random seed for reproducibility (None = truly random each run)


@dataclass
class PSOResult:
    """Container for the results returned by the PSO optimization.
    
    Holds the best parameter vector found, its cost, and the convergence
    history (cost of the best solution at each iteration) for plotting.
    """
    best_params: np.ndarray           # the best 24-D parameter vector found during optimization
    best_cost: float                  # the scalar cost achieved by best_params (lower = better traffic flow)
    convergence_history: np.ndarray   # array of best cost at each iteration, used to plot convergence curve


def run_pso(
    fitness_fn: Callable[[np.ndarray], float],
    config: PSOConfig = PSOConfig(),
) -> PSOResult:
    """
    Run Particle Swarm Optimization to minimize `fitness_fn(params) -> float`
    over the fuzzy controller parameter space.
    
    The algorithm works as follows:
    1. Initialize a swarm of particles at random positions in the search space
    2. Each particle represents a complete 24-D fuzzy controller parameter vector
    3. Particles move through the space influenced by:
       - Their own velocity (inertia — keeps them going)
       - Their personal best position (cognitive — remembers own success)
       - The global best position (social — learns from the swarm)
    4. Over iterations, the swarm converges toward the best solution

    `fitness_fn` is expected to be `functools.partial(evaluate_params,
    seeds=..., weights=..., sim_config=...)` from cost.py, but any
    callable(params: np.ndarray) -> float works.

    Parameters
    ----------
    fitness_fn : callable
        Function that receives a 24-D parameter vector and returns a scalar cost.
        Lower cost = better controller.
    config : PSOConfig
        Configuration for the PSO algorithm (swarm size, iterations, etc.).

    Returns
    -------
    PSOResult
        Object containing the best parameter vector found, its cost, and the
        convergence history for plotting.
    """
    # Create a random number generator with the given seed for reproducibility
    rng = np.random.default_rng(config.seed)
    
    # Get the allowed minimum/maximum bounds for each of the 24 parameters
    # bounds shape: (24, 2) -> [lower_bounds, upper_bounds]
    bounds = param_bounds()
    lo, hi = bounds[:, 0], bounds[:, 1]  # lo = array of lower bounds for each gene, hi = array of upper bounds
    span = hi - lo                         # range (width) of each parameter's allowed values

    # --- initialize swarm ---------------------------------------------
    # Create random positions for all particles within the allowed bounds
    positions = lo + rng.random((config.num_particles, PARAM_DIM)) * span
    # Seed one particle at the hand-tuned baseline so PSO never does worse
    # than manual tuning by pure bad luck. This guarantees the optimizer
    # starts with at least one known-reasonable solution.
    positions[0] = default_params()
    
    # Initialize velocities randomly, scaled to 10% of the parameter range
    # Small initial velocities prevent particles from shooting out of bounds immediately
    velocities = (rng.random((config.num_particles, PARAM_DIM)) - 0.5) * span * 0.1

    # Each particle remembers its own best position (initially just its starting position)
    personal_best_pos = positions.copy()
    # Evaluate fitness for all initial particle positions
    personal_best_cost = np.array([fitness_fn(p) for p in positions])

    # Find the global best among all particles (lowest cost)
    global_best_idx = int(np.argmin(personal_best_cost))
    global_best_pos = personal_best_pos[global_best_idx].copy()  # copy to avoid mutation
    global_best_cost = float(personal_best_cost[global_best_idx])

    # Record the initial best cost as iteration 0
    convergence = [global_best_cost]
    inertia = config.inertia_w  # start with the full inertia weight

    # Main optimization loop: each iteration moves all particles and updates bests
    for _ in range(config.num_iterations):
        # Generate random numbers for stochastic exploration (different for each particle and gene)
        r1 = rng.random((config.num_particles, PARAM_DIM))  # randomness for cognitive component
        r2 = rng.random((config.num_particles, PARAM_DIM))  # randomness for social component

        # ----- Velocity update (PSO core equation) -----
        # v_new = inertia * v_old                    (keep moving in same direction)
        #       + c1 * r1 * (personal_best - current) (move toward own best)
        #       + c2 * r2 * (global_best - current)   (move toward swarm's best)
        velocities = (
            inertia * velocities
            + config.cognitive_c1 * r1 * (personal_best_pos - positions)  # pull toward personal best
            + config.social_c2 * r2 * (global_best_pos - positions)       # pull toward global best
        )
        
        # ----- Position update -----
        # x_new = x_old + v_new
        proposed = positions + velocities

        # Absorbing-wall bounds handling: if a particle tries to go outside
        # the allowed range, clip its position back and zero out the velocity
        # component that pushed it out (prevents particles from "bouncing" or
        # getting stuck at the boundary with residual velocity).
        out_of_bounds = (proposed < lo) | (proposed > hi)  # boolean mask: which genes are out of bounds
        velocities[out_of_bounds] = 0.0                     # kill velocity for out-of-bounds components
        positions = np.clip(proposed, lo, hi)               # clip positions back to valid range

        # Evaluate fitness for all new particle positions
        costs = np.array([fitness_fn(p) for p in positions])

        # Update personal bests: if a particle found a better position, remember it
        improved = costs < personal_best_cost  # boolean mask: which particles improved
        personal_best_pos[improved] = positions[improved]
        personal_best_cost[improved] = costs[improved]

        # Update global best: if any particle beat the previous global best, update it
        iter_best_idx = int(np.argmin(personal_best_cost))
        if personal_best_cost[iter_best_idx] < global_best_cost:
            global_best_cost = float(personal_best_cost[iter_best_idx])
            global_best_pos = personal_best_pos[iter_best_idx].copy()

        # Record the best cost found so far for convergence plotting
        convergence.append(global_best_cost)
        
        # Decay inertia weight: gradually reduce exploration, increase exploitation
        # Early iterations: high inertia = explore widely
        # Late iterations: low inertia = fine-tune near the best solution
        inertia *= config.inertia_decay

    # Return the optimization result with the best solution found
    return PSOResult(
        best_params=global_best_pos,                    # the 24-D parameter vector of the best controller
        best_cost=global_best_cost,                     # its cost (lower = better)
        convergence_history=np.array(convergence),      # cost history for plotting convergence curve
    )