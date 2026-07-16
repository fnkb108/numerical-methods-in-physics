from typing import List, Tuple
import numpy as np

from core.grid import Grid, GridConfig, update_E, update_H
from core.pml import PMLConfig, apply_pml
from core.sources import Source, Waveform
from models.results import Snapshot


def run_and_collect(
    N: int,
    dx: float,
    Q: float,
    num_steps: int,
    src_pos: int,
    waveform: Waveform,
    mode: str,
    use_pml: bool,
    pmlN: int,
    snap_interval: int,
) -> Tuple[np.ndarray, List[Snapshot]]:
    cfg = GridConfig(
        num_cells=N,
        dx=dx,
        courant=Q,
        c_speed=1.0,
    )

    grid = Grid(cfg)

    if use_pml:
        apply_pml(
            grid,
            PMLConfig(thickness=pmlN, order=3, delta=1e-6),
        )

    source = Source(index=src_pos, waveform=waveform, mode=mode)
    snapshots: List[Snapshot] = []
    x = np.arange(N) * dx

    for n in range(num_steps):
        update_H(grid)
        source.inject(grid, n)
        update_E(grid)

        if snap_interval > 0 and n % snap_interval == 0 and n != 0:
            snapshots.append(Snapshot(step=n, Ey=grid.Ey.copy()))

    return x, snapshots