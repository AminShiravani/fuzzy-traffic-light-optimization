"""
simulation.py

This file contains the playback engine.

It does NOT perform any fuzzy logic.

It simply "plays" an already generated simulation.
"""

from dataclasses import dataclass


@dataclass
class FrameState:
    frame: int

    queue1: int
    queue2: int

    green1: float
    green2: float

    light1: int
    light2: int

    phase: int

    time: int


class SimulationPlayer:
    """
    Plays a recorded simulation.

    Think of it like a video player.

    JSON ----> SimulationPlayer ----> Current Frame
    """

    def __init__(self, simulation_data: dict):

        self.data = simulation_data

        self.queue1 = simulation_data["queue1_history"]
        self.queue2 = simulation_data["queue2_history"]

        self.green1 = simulation_data["green1_history"]
        self.green2 = simulation_data["green2_history"]

        self.light1 = simulation_data["light1_history"]
        self.light2 = simulation_data["light2_history"]
        self.phase = simulation_data["phase_history"]

        self.time_axis = simulation_data["time_axis"]

        self.total_frames = len(self.time_axis)

        self.current_frame = 0

    # --------------------------------------------------

    def restart(self):

        """
        Restart playback.
        """

        self.current_frame = 0

    # --------------------------------------------------

    def next_frame(self):

        """
        Advance one frame.
        """

        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1

    # --------------------------------------------------

    def previous_frame(self):

        """
        Move one frame backwards.
        """

        if self.current_frame > 0:
            self.current_frame -= 1

    # --------------------------------------------------

    def is_finished(self):

        """
        Returns True if playback reached the end.
        """

        return self.current_frame >= self.total_frames - 1

    # --------------------------------------------------

    def get_state(self) -> FrameState:
        
        i = self.current_frame
        green_index = min(i, len(self.green1) - 1)
    
        return FrameState(
            frame=i,
    
            queue1=self.queue1[i],
            queue2=self.queue2[i],
    
            green1=self.green1[green_index],
            green2=self.green2[green_index],
    
            light1=self.light1[i],
            light2=self.light2[i],
    
            phase=self.phase[i],
    
            time=self.time_axis[i],
        )