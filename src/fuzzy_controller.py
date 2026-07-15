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
QUEUE_UNIVERSE_MAX = 40.0  # max meaningful queue length (vehicles) — beyond this the queue is considered "full"
ARRIVAL_UNIVERSE_MAX = 3.0  # max meaningful arrival rate (veh/step) — beyond this arrivals are "saturated"
LAMBDA_ARRIVAL = 5.0  # weight converting arrival-rate into "pressure" units — how much 1 veh/step of arrival counts as equivalent queue vehicles
PRESSURE_UNIVERSE_MAX = QUEUE_UNIVERSE_MAX + LAMBDA_ARRIVAL * ARRIVAL_UNIVERSE_MAX  # max possible pressure = 40 + 5*3 = 55

GREEN_MIN = 5.0  # seconds, minimum green (safety floor — at least 5s of green to let one car pass safely)
GREEN_MAX = 60.0  # seconds, maximum green (fairness ceiling — prevents one road from hogging the green forever)

N_MF_POINTS_PER_VAR = 3  # 3-center Ruspini-partition encoding per fuzzy variable (c1, c2, c3)
N_VARS = 3  # pressure_self, pressure_other, green_time — the three fuzzy variables in the system
N_RULES = 9  # total rules in the 3x3 rule table (3 self pressure levels x 3 other pressure levels)
PARAM_DIM = N_VARS * N_MF_POINTS_PER_VAR + N_RULES  # = 9 membership centers + 9 rule weights = 18 total parameters to optimize

# Discretization resolution for the fuzzy universes (affects accuracy/speed)
# Higher = more accurate defuzzification but slightly slower per inference
_RESOLUTION = 121

# Rule table: (self_level, other_level) -> output_level
# levels: 0=Low, 1=Medium, 2=High  (both antecedents and consequent share
# the Low/Medium/High naming, consequent semantically = Short/Medium/Long)
# 
# The logic behind each rule:
# - If self pressure is Low (few cars on our road), give Short green regardless of other road
# - If self is Medium and other is Low, give Long green (our road needs it, other doesn't)
# - If self is Medium and other is High, give Short green (yield to the busier road)
# - If self is High and other is Low, give Long green (clear our big queue)
# - If both are High, give Medium green (fair compromise, split time evenly)
_RULE_TABLE = [
    # (self, other) -> consequent
    ((0, 0), 0),  # self Low,  other Low    -> Short  (neither road has much traffic, keep it brief)
    ((0, 1), 0),  # self Low,  other Medium -> Short  (other road has more demand, yield quickly)
    ((0, 2), 0),  # self Low,  other High   -> Short  (other road is very busy, give it the green soon)
    ((1, 0), 2),  # self Med,  other Low    -> Long   (our road has moderate demand, other doesn't need green)
    ((1, 1), 1),  # self Med,  other Medium -> Medium (both roads have similar moderate demand)
    ((1, 2), 0),  # self Med,  other High   -> Short  (yield to busier road)
    ((2, 0), 2),  # self High, other Low    -> Long   (our road has big queue, clear it)
    ((2, 1), 2),  # self High, other Medium -> Long   (our road has bigger demand)
    ((2, 2), 1),  # self High, other High   -> Medium (fair compromise when both roads are congested)
]


def default_params() -> np.ndarray:
    """
    A sane, hand-tuned (non-optimized) starting parameter vector.
    Used both as the PSO/ACO initialization seed and as the "baseline,
    manually-tuned" controller for comparison in main.py.
    
    Returns an 18-element vector:
    - First 3: pressure_self membership centers (c1, c2, c3)
    - Next 3:  pressure_other membership centers (c1, c2, c3)
    - Next 3:  green_time membership centers (c1, c2, c3)
    - Last 9:  rule weights (all 1.0 = equal influence for all rules)
    """
    pu = PRESSURE_UNIVERSE_MAX  # shorthand for the maximum pressure value (55.0)
    gu = GREEN_MAX - GREEN_MIN  # range of green time (55.0)

    # 3 centers per variable: c1 (Low peak), c2 (Medium peak), c3 (High peak)
    # Pressure centers: evenly spread across [0, PRESSURE_UNIVERSE_MAX]
    pressure_self_pts = np.array([0.0, pu * 0.5, pu])   # Low peak at 0, Medium at half, High at max
    pressure_other_pts = np.array([0.0, pu * 0.5, pu])  # same symmetric partitioning for the other road
    # Green time centers: evenly spread across [GREEN_MIN, GREEN_MAX]
    green_pts = np.array([GREEN_MIN, GREEN_MIN + gu * 0.5, GREEN_MAX])  # Short=5s, Medium=32.5s, Long=60s

    # All rules initially have weight 1.0 (fully active, optimizer can reduce some)
    rule_weights = np.ones(N_RULES)

    # Concatenate all parameters into a single flat vector
    return np.concatenate(
        [pressure_self_pts, pressure_other_pts, green_pts, rule_weights]
    )


def param_bounds() -> np.ndarray:
    """
    Returns an array of shape (PARAM_DIM, 2) with (low, high) bounds for
    every gene, for use by PSO/ACO. The 3 centers per variable are each
    bounded to their variable's universe; rule weights to [0, 1].
    
    This defines the search space boundaries: each of the 18 parameters
    must stay within its allowed range.
    """
    bounds = []
    # Pressure_self centers: must be between 0 and PRESSURE_UNIVERSE_MAX
    bounds += [(0.0, PRESSURE_UNIVERSE_MAX)] * N_MF_POINTS_PER_VAR  # pressure_self (3 bounds)
    # Pressure_other centers: same range
    bounds += [(0.0, PRESSURE_UNIVERSE_MAX)] * N_MF_POINTS_PER_VAR  # pressure_other (3 bounds)
    # Green time centers: must be between GREEN_MIN and GREEN_MAX
    bounds += [(GREEN_MIN, GREEN_MAX)] * N_MF_POINTS_PER_VAR  # green_time (3 bounds)
    # Rule weights: must be between 0 (rule disabled) and 1 (fully active)
    bounds += [(0.0, 1.0)] * N_RULES  # rule weights (9 bounds)
    return np.array(bounds, dtype=float)


def _repair_sorted(points: np.ndarray) -> np.ndarray:
    """Force a 3-tuple to be strictly non-decreasing (keeps triangular MFs
    valid even if an optimizer proposes an out-of-order vector), with a
    tiny epsilon nudge so degenerate c1==c2==c3 cases don't produce
    zero-width membership functions.
    
    This is a safety net: PSO/ACO might occasionally propose parameters
    where c2 < c1 (out of order). Sorting fixes this, and the epsilon
    ensures we never have exactly equal centers (which would make
    triangular MFs collapse to zero width).
    """
    s = np.sort(points)  # force c1 <= c2 <= c3
    eps = 1e-6  # tiny separation to avoid degenerate membership functions
    # Ensure minimum gap between consecutive centers
    if s[1] - s[0] < eps:
        s[1] = s[0] + eps  # push c2 slightly above c1
    if s[2] - s[1] < eps:
        s[2] = s[1] + eps  # push c3 slightly above c2
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
    
    This is the standard textbook fuzzy partition: at any point, you belong
    to at most two sets, and the membership degrees always add up to 1
    (a proper fuzzy partition, not just three independent triangles).
    """
    c1, c2, c3 = _repair_sorted(points)  # ensure valid ordering with gaps
    # Build the three triangular membership functions
    low = fuzz.trimf(universe, [c1, c1, c2])     # Low: peaks at c1, drops to 0 at c2
    medium = fuzz.trimf(universe, [c1, c2, c3])   # Medium: 0 at c1, peaks at c2, 0 at c3
    high = fuzz.trimf(universe, [c2, c3, c3])     # High: 0 at c2, peaks at c3
    return low, medium, high


class FuzzyTrafficController:
    """
    A single reusable Mamdani FLC instance. Build once per parameter vector
    (cheap: a handful of numpy arrays), then call `.compute()` many times
    during a simulation episode.
    
    This is the actual traffic light controller: it takes the current queue
    lengths and arrival rates for both roads and returns the green light
    duration (in seconds) for the road being controlled.
    """

    def __init__(self, params: np.ndarray | None = None):
        """
        Initialize the fuzzy controller with a given parameter vector.
        
        Parameters
        ----------
        params : np.ndarray or None
            18-element parameter vector. If None, uses default_params().
            Structure: [pressure_self_pts(3), pressure_other_pts(3),
                       green_pts(3), rule_weights(9)]
        """
        if params is None:
            params = default_params()  # use hand-tuned defaults if no params given
        params = np.asarray(params, dtype=float)
        if params.shape[0] != PARAM_DIM:
            raise ValueError(
                f"Expected parameter vector of length {PARAM_DIM}, got {params.shape[0]}"
            )
        self.params = params  # store the full parameter vector
        self._build()  # pre-compute all membership functions

    def _build(self) -> None:
        """
        Pre-compute the membership functions and discretized universes.
        Called once during __init__ to avoid repeated computation.
        
        This unpacks the flat parameter vector into:
        - 3 centers for pressure_self membership functions
        - 3 centers for pressure_other membership functions
        - 3 centers for green_time membership functions
        - 9 rule weights
        and builds the actual membership function arrays over the discretized universes.
        """
        p = self.params
        # Extract the three groups of membership function centers
        self.pressure_self_pts = p[0:3]   # c1, c2, c3 for "my road's pressure" MFs
        self.pressure_other_pts = p[3:6]  # c1, c2, c3 for "other road's pressure" MFs
        self.green_pts = p[6:9]           # c1, c2, c3 for "green time" MFs
        # Extract rule weights and clip to [0, 1] for safety
        self.rule_weights = np.clip(p[9:18], 0.0, 1.0)

        # Create discretized universes for pressure and green time
        self.u_pressure = np.linspace(0.0, PRESSURE_UNIVERSE_MAX, _RESOLUTION)  # 121 points from 0 to 55
        self.u_green = np.linspace(GREEN_MIN, GREEN_MAX, _RESOLUTION)            # 121 points from 5 to 60

        # Build the actual membership function arrays for each variable
        # Each is a tuple of 3 arrays: (low_mf, medium_mf, high_mf)
        self.self_mfs = _trimf_partition(self.u_pressure, self.pressure_self_pts)
        self.other_mfs = _trimf_partition(self.u_pressure, self.pressure_other_pts)
        self.green_mfs = _trimf_partition(self.u_green, self.green_pts)

    @staticmethod
    def _pressure(queue: float, arrival_rate: float) -> float:
        """
        Compute the "pressure" on a road by combining current queue length
        and arrival rate into a single demand signal.
        
        Pressure = queue + LAMBDA_ARRIVAL * arrival_rate
        
        This pre-processing step simplifies the fuzzy system from 4 inputs
        (Q1, A1, Q2, A2) down to 2 inputs (pressure_self, pressure_other),
        making the rule base manageable (3x3=9 rules instead of 3^4=81).
        
        Parameters
        ----------
        queue : float
            Current number of vehicles waiting on this road.
        arrival_rate : float
            Recent arrival rate (vehicles per time step) for this road.
            
        Returns
        -------
        float
            Pressure value, clipped to [0, PRESSURE_UNIVERSE_MAX].
        """
        raw = queue + LAMBDA_ARRIVAL * arrival_rate  # combine queue and incoming traffic
        return float(np.clip(raw, 0.0, PRESSURE_UNIVERSE_MAX))  # keep within valid range

    def _fuzzify(
        self,
        value: float,
        mfs: tuple[np.ndarray, np.ndarray, np.ndarray],
        universe: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Fuzzify a crisp input value into membership degrees for the three
        fuzzy sets (Low, Medium, High).
        
        Parameters
        ----------
        value : float
            The crisp input value to fuzzify.
        mfs : tuple of 3 arrays
            The membership function arrays for Low, Medium, High.
        universe : np.ndarray
            The discretized universe over which the MFs are defined.
            
        Returns
        -------
        tuple[float, float, float]
            Membership degrees: (mu_Low, mu_Medium, mu_High).
        """
        # Interpolate to find the membership degree of `value` in each fuzzy set
        low_deg = fuzz.interp_membership(universe, mfs[0], value)   # how "Low" is this value?
        med_deg = fuzz.interp_membership(universe, mfs[1], value)   # how "Medium" is this value?
        high_deg = fuzz.interp_membership(universe, mfs[2], value)  # how "High" is this value?
        return low_deg, med_deg, high_deg

    def compute(
        self,
        queue_self: float,
        arrival_self: float,
        queue_other: float,
        arrival_other: float,
    ) -> float:
        """
        Run one Mamdani inference pass and return a crisp green-time (seconds)
        for the road being controlled (the "self" road).
        
        This is the main method called by the traffic simulator at each
        decision point. It performs the full fuzzy inference chain:
        1. Pre-processing: combine queue + arrival rate into "pressure"
        2. Fuzzification: convert crisp pressures to membership degrees
        3. Rule evaluation: compute firing strength for each of the 9 rules
        4. Aggregation: combine all rule outputs using MAX
        5. Defuzzification: centroid method to get crisp green time

        Parameters
        ----------
        queue_self : float
            Current queue length (vehicles) on the road being controlled.
        arrival_self : float
            Recent arrival rate (veh/step) for the road being controlled.
        queue_other : float
            Current queue length (vehicles) on the competing road.
        arrival_other : float
            Recent arrival rate (veh/step) for the competing road.

        Returns
        -------
        float
            Crisp green-light duration in seconds, clipped to [GREEN_MIN, GREEN_MAX].
        """
        # Step 1: Pre-processing — convert raw queue + arrival into "pressure" signals
        p_self = self._pressure(queue_self, arrival_self)    # demand on our road
        p_other = self._pressure(queue_other, arrival_other)  # demand on the other road

        # Step 2: Fuzzification — get membership degrees for each input
        self_deg = self._fuzzify(p_self, self.self_mfs, self.u_pressure)    # (mu_Low, mu_Med, mu_High) for self
        other_deg = self._fuzzify(p_other, self.other_mfs, self.u_pressure)  # (mu_Low, mu_Med, mu_High) for other

        # Step 3-4: Rule evaluation and aggregation
        # Mamdani aggregation: for each rule, firing strength = AND (min) of
        # antecedent degrees, scaled by the (optimizable) rule weight; the
        # rule's consequent MF is clipped at that strength; all clipped
        # consequents are aggregated with max, then centroid-defuzzified.
        aggregate = np.zeros_like(self.u_green)  # start with zero everywhere on green universe
        any_fired = False  # track if at least one rule fired
        for idx, ((self_lvl, other_lvl), out_lvl) in enumerate(_RULE_TABLE):
            # Compute firing strength = min(antecedent degrees) * rule_weight
            # The AND operator is min (standard Mamdani)
            strength = (
                min(self_deg[self_lvl], other_deg[other_lvl])  # how strongly this rule's conditions match
                * self.rule_weights[idx]                         # scaled by the optimizable weight
            )
            if strength <= 0.0:
                continue  # skip rules that don't fire at all (saves computation)
            # Clip the output membership function at the firing strength
            # (Mamdani implication: truncate the consequent MF)
            clipped = np.fmin(strength, self.green_mfs[out_lvl])
            # Aggregate with max (combine outputs of all rules)
            aggregate = np.fmax(aggregate, clipped)
            any_fired = True

        # Step 5: Defuzzification — convert the aggregated fuzzy set to a crisp number
        if not any_fired or aggregate.sum() == 0.0:
            # Degenerate case (e.g. all rule weights collapsed to 0):
            # fall back to a neutral mid-range green time rather than
            # crashing the defuzzifier on an all-zero aggregate.
            return float((GREEN_MIN + GREEN_MAX) / 2.0)

        # Centroid defuzzification: find the "center of mass" of the aggregated output
        crisp = fuzz.defuzz(self.u_green, aggregate, "centroid")
        # Ensure the result stays within the allowed green time range
        return float(np.clip(crisp, GREEN_MIN, GREEN_MAX))

    def get_membership_functions(self):
        """
        Expose the pre-computed membership function arrays and universes
        for visualization purposes (used by plots.py to draw the fuzzy sets).
        
        Returns
        -------
        dict
            Dictionary containing universes and membership function arrays
            for all three fuzzy variables (pressure_self, pressure_other, green_time).
        """
        return {
            "pressure_universe": self.u_pressure,      # x-axis for pressure plots
            "green_universe": self.u_green,            # x-axis for green time plots
            "pressure_self_mfs": self.self_mfs,        # (low, med, high) MFs for self pressure
            "pressure_other_mfs": self.other_mfs,      # (low, med, high) MFs for other pressure
            "green_mfs": self.green_mfs,               # (short, medium, long) MFs for green time
        }