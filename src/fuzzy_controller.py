"""
fuzzy_controller.py
====================

Parametrized Mamdani Fuzzy Logic Controller (FLC) for adaptive traffic-light
green-time computation.

DESIGN / PARAMETER-MAPPING LOGIC (read this before touching pso.py / aco.py)
-----------------------------------------------------------------------------
The project brief asks for two crisp inputs per road — Queue Length (Q) and
Arrival Rate (A) — and a single crisp output — Green Duration (G). A fully
cross-combined 4-antecedent system (Q1, A1, Q2, A2) would need up to 3^4 = 81
rules, which is impossible to hand-design sensibly and turns "rule base
optimization" into an intractable combinatorial problem for continuous
optimizers like PSO/ACO.

Instead we use a standard, well-justified engineering simplification:

    pressure_i = Q_i + LAMBDA_ARRIVAL * A_i

"Pressure" is a crisp pre-processing feature that folds the *current* queue
and the *rate at which it will keep growing* into a single demand signal per
road (LAMBDA_ARRIVAL is a fixed, non-optimized weighting constant — you can
tune it in `config.py`-style constants below). This is analogous to a
feature-engineering step upstream of a smaller, well-conditioned FLC.

The FLC itself is then a classical 2-input / 1-output Mamdani system:

    Inputs:  pressure_self  (Low / Medium / High)
             pressure_other (Low / Medium / High)
    Output:  green_time     (Short / Medium / Long)

which keeps the rule base at a manageable 3x3 = 9 rules (matching the scale
of the example rules given in the project PDF) while still using both Q and
A, as required.

WHY WE DON'T USE skfuzzy's `ctrl.ControlSystem`
------------------------------------------------
skfuzzy's high-level `ctrl` API is great for static, hand-tuned systems, but
it is awkward to re-parametrize on every PSO/ACO fitness evaluation (you'd
rebuild fuzzy variable objects thousands of times, and it does not expose a
clean hook for per-rule *weights*). Instead we implement the Mamdani
inference pipeline manually using `skfuzzy.membership` primitives
(`skfuzzy.trimf`) for fuzzification and `skfuzzy.defuzz` for centroid
defuzzification. This gives us:

  1. A flat, bounded parameter vector describing every membership function
     -> directly usable as a PSO particle / ACO solution.
  2. Per-rule firing-strength WEIGHTS (9 extra parameters) so the
     optimizer can also tune the *influence* of the rule base, which is the
     practical, continuous-optimization-friendly stand-in for "optimizing
     the rule base" (searching over the discrete space of rule structures
     is a combinatorial problem outside PSO/ACO's natural domain).

PARAMETER VECTOR ENCODING (used by pso.py / aco.py)
------------------------------------------------------
Each of the 3 fuzzy variables (pressure_self, pressure_other, green_time)
is described by only 3 sorted CENTERS  c1 <= c2 <= c3  over its universe
[0, U], defining a classical Ruspini strong fuzzy partition (the standard
"50%-overlap" triangular partition taught in every fuzzy-logic course):

    Low    = trimf(x, [c1, c1, c2])
    Medium = trimf(x, [c1, c2, c3])
    High   = trimf(x, [c2, c3, c3])

This construction guarantees, for ANY sorted (c1, c2, c3):
  - every point x in [c1, c3] belongs to at most two adjacent terms
  - at each partition boundary ((c1+c2)/2 and (c2+c3)/2) the two
    overlapping terms are both exactly 0.5 -> the textbook "50% overlap"
  - the memberships sum to exactly 1.0 everywhere in [c1, c3] (a genuine
    Ruspini / strong fuzzy partition), so there are no "gap" regions with
    low total membership in every term, unlike a looser independent
    5-breakpoint encoding.

    3 variables x 3 centers           = 9 continuous genes
    9 rule weights (clipped to [0,1]) = 9 continuous genes
    ------------------------------------------------------
    TOTAL SEARCH-SPACE DIMENSION      = 18

This exact 18-length vector is what `pso.py` and `aco.py` optimize.
"""

from __future__ import annotations

import numpy as np
import skfuzzy as fuzz

# ---------------------------------------------------------------------------
# Fixed (non-optimized) constants
# ---------------------------------------------------------------------------
QUEUE_UNIVERSE_MAX = 40.0  # max meaningful queue length (vehicles)
ARRIVAL_UNIVERSE_MAX = 3.0  # max meaningful arrival rate (veh/step)
LAMBDA_ARRIVAL = 5.0  # weight converting arrival-rate into "pressure" units
PRESSURE_UNIVERSE_MAX = QUEUE_UNIVERSE_MAX + LAMBDA_ARRIVAL * ARRIVAL_UNIVERSE_MAX

GREEN_MIN = 5.0  # seconds, minimum green (safety floor)
GREEN_MAX = 60.0  # seconds, maximum green (fairness ceiling)

N_MF_POINTS_PER_VAR = 3  # 3-center Ruspini-partition encoding per fuzzy variable
N_VARS = 3  # pressure_self, pressure_other, green_time
N_RULES = 9
PARAM_DIM = N_VARS * N_MF_POINTS_PER_VAR + N_RULES  # = 18

# Discretization resolution for the fuzzy universes (affects accuracy/speed)
_RESOLUTION = 121

# Rule table: (self_level, other_level) -> output_level
# levels: 0=Low, 1=Medium, 2=High  (both antecedents and consequent share
# the Low/Medium/High naming, consequent semantically = Short/Medium/Long)
_RULE_TABLE = [
    # (self, other) -> consequent
    ((0, 0), 0),  # self Low,  other Low    -> Short
    ((0, 1), 0),  # self Low,  other Medium -> Short
    ((0, 2), 0),  # self Low,  other High   -> Short
    ((1, 0), 2),  # self Med,  other Low    -> Long   (other doesn't need road)
    ((1, 1), 1),  # self Med,  other Medium -> Medium
    ((1, 2), 0),  # self Med,  other High   -> Short  (yield to busier road)
    ((2, 0), 2),  # self High, other Low    -> Long
    ((2, 1), 2),  # self High, other Medium -> Long
    ((2, 2), 1),  # self High, other High   -> Medium (fair compromise)
]


def default_params() -> np.ndarray:
    """
    A sane, hand-tuned (non-optimized) starting parameter vector.
    Used both as the PSO/ACO initialization seed and as the "baseline,
    manually-tuned" controller for comparison in main.py.
    """
    pu = PRESSURE_UNIVERSE_MAX
    gu = GREEN_MAX - GREEN_MIN

    # 3 centers per variable: c1 (Low peak), c2 (Medium peak), c3 (High peak)
    pressure_self_pts = np.array([0.0, pu * 0.5, pu])
    pressure_other_pts = np.array([0.0, pu * 0.5, pu])
    green_pts = np.array([GREEN_MIN, GREEN_MIN + gu * 0.5, GREEN_MAX])

    rule_weights = np.ones(N_RULES)

    return np.concatenate(
        [pressure_self_pts, pressure_other_pts, green_pts, rule_weights]
    )


def param_bounds() -> np.ndarray:
    """
    Returns an array of shape (PARAM_DIM, 2) with (low, high) bounds for
    every gene, for use by PSO/ACO. The 3 centers per variable are each
    bounded to their variable's universe; rule weights to [0, 1].
    """
    bounds = []
    bounds += [(0.0, PRESSURE_UNIVERSE_MAX)] * N_MF_POINTS_PER_VAR  # pressure_self
    bounds += [(0.0, PRESSURE_UNIVERSE_MAX)] * N_MF_POINTS_PER_VAR  # pressure_other
    bounds += [(GREEN_MIN, GREEN_MAX)] * N_MF_POINTS_PER_VAR  # green_time
    bounds += [(0.0, 1.0)] * N_RULES  # rule weights
    return np.array(bounds, dtype=float)


def _repair_sorted(points: np.ndarray) -> np.ndarray:
    """Force a 3-tuple to be strictly non-decreasing (keeps triangular MFs
    valid even if an optimizer proposes an out-of-order vector), with a
    tiny epsilon nudge so degenerate c1==c2==c3 cases don't produce
    zero-width membership functions."""
    s = np.sort(points)
    eps = 1e-6
    if s[1] - s[0] < eps:
        s[1] = s[0] + eps
    if s[2] - s[1] < eps:
        s[2] = s[1] + eps
    return s


def _trimf_partition(
    universe: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (Low, Medium, High) triangular membership arrays from a sorted
    3-center vector (c1, c2, c3), following the Ruspini strong-partition
    construction documented in the module docstring:

        Low    = trimf(x, [c1, c1, c2])
        Medium = trimf(x, [c1, c2, c3])
        High   = trimf(x, [c2, c3, c3])

    For any c1 <= c2 <= c3 this guarantees memberships sum to 1.0 at every
    x in [c1, c3], with adjacent terms crossing at exactly 0.5 at the
    midpoints (c1+c2)/2 and (c2+c3)/2 -- i.e. genuine 50% overlap and no
    "gap" regions of low total membership.
    """
    c1, c2, c3 = _repair_sorted(points)
    low = fuzz.trimf(universe, [c1, c1, c2])
    medium = fuzz.trimf(universe, [c1, c2, c3])
    high = fuzz.trimf(universe, [c2, c3, c3])
    return low, medium, high


class FuzzyTrafficController:
    """
    A single reusable Mamdani FLC instance. Build once per parameter vector
    (cheap: a handful of numpy arrays), then call `.compute()` many times
    during a simulation episode.
    """

    def __init__(self, params: np.ndarray | None = None):
        if params is None:
            params = default_params()
        params = np.asarray(params, dtype=float)
        if params.shape[0] != PARAM_DIM:
            raise ValueError(
                f"Expected parameter vector of length {PARAM_DIM}, got {params.shape[0]}"
            )
        self.params = params
        self._build()

    def _build(self) -> None:
        p = self.params
        self.pressure_self_pts = p[0:3]
        self.pressure_other_pts = p[3:6]
        self.green_pts = p[6:9]
        self.rule_weights = np.clip(p[9:18], 0.0, 1.0)

        self.u_pressure = np.linspace(0.0, PRESSURE_UNIVERSE_MAX, _RESOLUTION)
        self.u_green = np.linspace(GREEN_MIN, GREEN_MAX, _RESOLUTION)

        self.self_mfs = _trimf_partition(self.u_pressure, self.pressure_self_pts)
        self.other_mfs = _trimf_partition(self.u_pressure, self.pressure_other_pts)
        self.green_mfs = _trimf_partition(self.u_green, self.green_pts)

    @staticmethod
    def _pressure(queue: float, arrival_rate: float) -> float:
        raw = queue + LAMBDA_ARRIVAL * arrival_rate
        return float(np.clip(raw, 0.0, PRESSURE_UNIVERSE_MAX))

    def _fuzzify(
        self,
        value: float,
        mfs: tuple[np.ndarray, np.ndarray, np.ndarray],
        universe: np.ndarray,
    ) -> tuple[float, float, float]:
        low_deg = fuzz.interp_membership(universe, mfs[0], value)
        med_deg = fuzz.interp_membership(universe, mfs[1], value)
        high_deg = fuzz.interp_membership(universe, mfs[2], value)
        return low_deg, med_deg, high_deg

    def compute(
        self,
        queue_self: float,
        arrival_self: float,
        queue_other: float,
        arrival_other: float,
    ) -> float:
        """
        Run one Mamdani inference pass and return a crisp green-time (s).

        queue_self / queue_other: current queue length (vehicles) for the
            road being controlled ("self") and the competing road ("other").
        arrival_self / arrival_other: recent arrival rate (veh/step) for the
            same two roads.
        """
        p_self = self._pressure(queue_self, arrival_self)
        p_other = self._pressure(queue_other, arrival_other)

        self_deg = self._fuzzify(p_self, self.self_mfs, self.u_pressure)
        other_deg = self._fuzzify(p_other, self.other_mfs, self.u_pressure)

        # Mamdani aggregation: for each rule, firing strength = AND (min) of
        # antecedent degrees, scaled by the (optimizable) rule weight; the
        # rule's consequent MF is clipped at that strength; all clipped
        # consequents are aggregated with max, then centroid-defuzzified.
        aggregate = np.zeros_like(self.u_green)
        any_fired = False
        for idx, ((self_lvl, other_lvl), out_lvl) in enumerate(_RULE_TABLE):
            strength = (
                min(self_deg[self_lvl], other_deg[other_lvl]) * self.rule_weights[idx]
            )
            if strength <= 0.0:
                continue
            clipped = np.fmin(strength, self.green_mfs[out_lvl])
            aggregate = np.fmax(aggregate, clipped)
            any_fired = True

        if not any_fired or aggregate.sum() == 0.0:
            # Degenerate case (e.g. all rule weights collapsed to 0):
            # fall back to a neutral mid-range green time rather than
            # crashing the defuzzifier on an all-zero aggregate.
            return float((GREEN_MIN + GREEN_MAX) / 2.0)

        crisp = fuzz.defuzz(self.u_green, aggregate, "centroid")
        return float(np.clip(crisp, GREEN_MIN, GREEN_MAX))

    def get_membership_functions(self):
        """Expose MF arrays/universes for plotting (see plots.py)."""
        return {
            "pressure_universe": self.u_pressure,
            "green_universe": self.u_green,
            "pressure_self_mfs": self.self_mfs,
            "pressure_other_mfs": self.other_mfs,
            "green_mfs": self.green_mfs,
        }
