"""
simulation.py
==============

Discrete-time simulation of a two-road signalized intersection, as
described in the project brief (Section 3): vehicles arrive stochastically
each time step, and depart when their road has a green light, at a fixed
saturation flow rate.

Simulation model
-----------------
Time is split into decision CYCLES. At the start of each cycle:
  1. The controller observes (queue1, queue2, arrival_rate1, arrival_rate2)
     and returns green durations (green1, green2) for the two roads.
  2. Road 1 is green for `green1` seconds while road 2 is red, then the
     reverse for `green2` seconds (a simple two-phase, non-overlapping
     signal plan -- no explicit yellow/all-red clearance interval is
     modeled, matching the abstraction level of the project brief).

During every 1-second sub-step within a phase:
  - New vehicles may arrive on EITHER road (arrivals do not stop just
    because a road has a red light), modeled as an independent Bernoulli/
    Poisson-thinned process per road using each road's arrival rate.
  - The road with the green light discharges vehicles from its queue at a
    fixed saturation flow rate (SATURATION_FLOW vehicles/second), capped by
    how many vehicles are actually waiting.
  - The road with the red light discharges nothing.

Metrics tracked (needed by cost.py):
  - total_wait_time_steps: running sum of (queue1 + queue2) sampled every
    second -> a standard discrete proxy for total vehicle-seconds of delay.
  - queue_samples: per-second queue lengths (for averaging / plotting).
  - stops: count of vehicles that arrived while their road's queue was
    already non-empty OR while their road had a red light -- i.e. vehicles
    that could not pass straight through, a standard "stop" proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SATURATION_FLOW = 0.6  # vehicles departing per second of green (per road)


@dataclass
class SimulationConfig:
    arrival_rate1: float = 0.35  # vehicles / second (Poisson mean), road 1
    arrival_rate2: float = 0.25  # vehicles / second (Poisson mean), road 2
    num_cycles: int = 60  # number of controller decision cycles per episode
    max_queue: float = 60.0  # hard cap to keep the simulation numerically bounded
    seed: Optional[int] = None

@dataclass
class EpisodeResult:
    avg_wait_time: float
    avg_queue_length: float
    total_stops: int
    total_vehicles: int

    queue1_history: np.ndarray
    queue2_history: np.ndarray

    green1_history: np.ndarray
    green2_history: np.ndarray

    light1_history: np.ndarray
    light2_history: np.ndarray
    phase_history: np.ndarray

    time_axis: np.ndarray


class TrafficIntersection:
    """Two-road, discrete-time intersection with a pluggable controller."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.reset()

    def reset(self) -> None:
        self.queue1 = 0.0
        self.queue2 = 0.0
        # Rolling estimate of arrival rate handed to the controller (a
        # simple exponential moving average over recent seconds, since a
        # real controller cannot know the *true* generative rate).
        self._ema1 = 0.0
        self._ema2 = 0.0
        self._ema_alpha = 0.2

    def _arrivals(self, rate: float) -> int:
        return int(self.rng.poisson(lam=max(rate, 0.0)))

    def _run_seconds(
        self, seconds: int, road1_green: bool, queue_log1: list, queue_log2: list
    ) -> tuple[int, int]:
        """Advance the simulation second-by-second for one signal phase.
        Returns (stops1, stops2) accumulated during this phase."""
        cfg = self.config
        stops1 = stops2 = 0

        for _ in range(seconds):
            a1 = self._arrivals(cfg.arrival_rate1)
            a2 = self._arrivals(cfg.arrival_rate2)

            # A vehicle "stops" if it arrives while its road is red, or
            # while its road is green but there's already a queue ahead of it.
            if a1 > 0:
                if (not road1_green) or self.queue1 > 0:
                    stops1 += a1
            if a2 > 0:
                if road1_green or self.queue2 > 0:
                    stops2 += a2

            self.queue1 = min(cfg.max_queue, self.queue1 + a1)
            self.queue2 = min(cfg.max_queue, self.queue2 + a2)

            if road1_green:
                depart = min(self.queue1, SATURATION_FLOW)
                self.queue1 -= depart
            else:
                depart = min(self.queue2, SATURATION_FLOW)
                self.queue2 -= depart

            self._ema1 = (1 - self._ema_alpha) * self._ema1 + self._ema_alpha * a1
            self._ema2 = (1 - self._ema_alpha) * self._ema2 + self._ema_alpha * a2

            queue_log1.append(self.queue1)
            queue_log2.append(self.queue2)

        return stops1, stops2

    def run_episode(self, controller) -> EpisodeResult:
        """
        Run one complete traffic simulation episode.
        """
    
        self.reset()
    
        # Queue history (one value every simulated second)
        q1_hist = []
        q2_hist = []
    
        # Green duration chosen by the controller (one value every cycle)
        g1_hist = []
        g2_hist = []
    
        # NEW: light state history (one value every simulated second)
        light1_hist = []
        light2_hist = []
    
        # NEW: phase history
        # 1 = Road 1 green
        # 2 = Road 2 green
        phase_hist = []
    
        total_stops = 0
        total_vehicles = 0
    
        for _ in range(self.config.num_cycles):
        
            # ---------------------------------------
            # Compute green duration for Road 1
            # ---------------------------------------
            green1 = controller.compute(
                self.queue1,
                self._ema1,
                self.queue2,
                self._ema2,
            )
    
            # ---------------------------------------
            # Compute green duration for Road 2
            # ---------------------------------------
            green2 = controller.compute(
                self.queue2,
                self._ema2,
                self.queue1,
                self._ema1,
            )
    
            # Store controller outputs
            g1_hist.append(green1)
            g2_hist.append(green2)
    
            # ---------------------------------------
            # Road 1 Green
            # ---------------------------------------
    
            duration1 = int(round(green1))
    
            s1, _ = self._run_seconds(
                duration1,
                road1_green=True,
                queue_log1=q1_hist,
                queue_log2=q2_hist,
            )
    
            # Save light state for every simulated second
            light1_hist.extend([1] * duration1)
            light2_hist.extend([0] * duration1)
            phase_hist.extend([1] * duration1)
    
            # ---------------------------------------
            # Road 2 Green
            # ---------------------------------------
    
            duration2 = int(round(green2))
    
            _, s2 = self._run_seconds(
                duration2,
                road1_green=False,
                queue_log1=q1_hist,
                queue_log2=q2_hist,
            )
    
            light1_hist.extend([0] * duration2)
            light2_hist.extend([1] * duration2)
            phase_hist.extend([2] * duration2)
    
            total_stops += s1 + s2
            total_vehicles += s1 + s2
    
        # Convert everything to numpy arrays
        q1_hist = np.array(q1_hist)
        q2_hist = np.array(q2_hist)
    
        avg_queue = float(np.mean(q1_hist + q2_hist)) if len(q1_hist) else 0.0
        avg_wait = avg_queue
    
        return EpisodeResult(
            avg_wait_time=avg_wait,
            avg_queue_length=avg_queue,
    
            total_stops=total_stops,
            total_vehicles=max(total_vehicles, 1),
    
            queue1_history=q1_hist,
            queue2_history=q2_hist,
    
            green1_history=np.array(g1_hist),
            green2_history=np.array(g2_hist),
    
            light1_history=np.array(light1_hist),
            light2_history=np.array(light2_hist),
            phase_history=np.array(phase_hist),
    
            time_axis=np.arange(len(q1_hist)),
        )
    