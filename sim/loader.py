"""
loader.py

Reads a simulation JSON file and converts it into Python objects.

This module does NOT know anything about pygame.
Its only responsibility is loading data.
"""

import json
from pathlib import Path


def load_simulation(path: str | Path) -> dict:
    """
    Load a simulation JSON.

    Parameters
    ----------
    path : str | Path
        Path to baseline.json / pso.json / aco.json

    Returns
    -------
    dict
        Dictionary containing all simulation data.
    """

    path = Path(path)

    with path.open("r") as f:
        data = json.load(f)

    return data