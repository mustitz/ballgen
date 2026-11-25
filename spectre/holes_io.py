"""Load the hole ledger written by ``fold_spectre.py`` (spectre_out/holes.json)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class HoleRecord:
    N: int                      # tiles in the folded patch
    hole: int                   # hole index within that fold
    sides: int
    kind: str
    K: int                      # angle deficit, exact integer degrees
    closes_flat: bool
    corner_angles: List[int]    # tile angle at each hole corner (alpha = 360 - this)
    boundary_half_edges: List[int]   # half-edge id = 14 * local_tile + edge_index
    area_fraction: float        # estimate: fraction of the sphere
    area_sr: float
    centre: np.ndarray          # estimate: unit vector

    @property
    def alpha(self) -> List[int]:
        return [360 - t for t in self.corner_angles]


def load_holes(path: str = "spectre_out/holes.json") -> List[HoleRecord]:
    with open(path) as f:
        rows = json.load(f)
    return [HoleRecord(r["N"], r["hole"], r["sides"], r["kind"], r["angle_deficit_K_deg"], r["closes_flat"],
                       r["corner_angles_deg"], r["boundary_half_edges"], r["area_fraction"], r["area_sr"],
                       np.array(r["centre_xyz"])) for r in rows]
