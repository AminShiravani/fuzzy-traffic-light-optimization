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
    num_particles: int = 20
    num_iterations: int = 40
    inertia_w: float = 0.72
    cognitive_c1: float = 1.49
    social_c2: float = 1.49
    inertia_decay: float = (
        0.99  # multiplies inertia_w each iteration (helps convergence)
    )
    seed: Optional[int] = 42


@dataclass
class PSOResult:
    best_params: np.ndarray
    best_cost: float
    convergence_history: np.ndarray  # best-so-far cost at each iteration


def run_pso(
    fitness_fn: Callable[[np.ndarray], float],
    config: PSOConfig = PSOConfig(),
) -> PSOResult:
    """
    Run PSO to minimize `fitness_fn(params) -> float`.

    `fitness_fn` is expected to be `functools.partial(evaluate_params,
    seeds=..., weights=..., sim_config=...)` from cost.py, but any
    callable(params: np.ndarray) -> float works.
    """
    rng = np.random.default_rng(config.seed)
    bounds = param_bounds()
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo

    # --- initialize swarm ---------------------------------------------
    positions = lo + rng.random((config.num_particles, PARAM_DIM)) * span
    # seed one particle at the hand-tuned baseline so PSO never does worse
    # than manual tuning by pure bad luck
    positions[0] = default_params()
    velocities = (rng.random((config.num_particles, PARAM_DIM)) - 0.5) * span * 0.1

    personal_best_pos = positions.copy()
    personal_best_cost = np.array([fitness_fn(p) for p in positions])

    global_best_idx = int(np.argmin(personal_best_cost))
    global_best_pos = personal_best_pos[global_best_idx].copy()
    global_best_cost = float(personal_best_cost[global_best_idx])

    convergence = [global_best_cost]
    inertia = config.inertia_w

    for _ in range(config.num_iterations):
        r1 = rng.random((config.num_particles, PARAM_DIM))
        r2 = rng.random((config.num_particles, PARAM_DIM))

        velocities = (
            inertia * velocities
            + config.cognitive_c1 * r1 * (personal_best_pos - positions)
            + config.social_c2 * r2 * (global_best_pos - positions)
        )
        proposed = positions + velocities

        # Absorbing-wall bounds handling: clip position, zero velocity
        # component that went out of bounds.
        out_of_bounds = (proposed < lo) | (proposed > hi)
        velocities[out_of_bounds] = 0.0
        positions = np.clip(proposed, lo, hi)

        costs = np.array([fitness_fn(p) for p in positions])

        improved = costs < personal_best_cost
        personal_best_pos[improved] = positions[improved]
        personal_best_cost[improved] = costs[improved]

        iter_best_idx = int(np.argmin(personal_best_cost))
        if personal_best_cost[iter_best_idx] < global_best_cost:
            global_best_cost = float(personal_best_cost[iter_best_idx])
            global_best_pos = personal_best_pos[iter_best_idx].copy()

        convergence.append(global_best_cost)
        inertia *= config.inertia_decay

    return PSOResult(
        best_params=global_best_pos,
        best_cost=global_best_cost,
        convergence_history=np.array(convergence),
    )
