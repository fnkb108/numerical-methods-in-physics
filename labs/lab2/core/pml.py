from dataclasses import dataclass
import numpy as np
from core.grid import Grid


@dataclass
class PMLConfig:
    thickness: int
    order: int
    delta: float


def apply_pml(grid: Grid, pml: PMLConfig):
    if pml.thickness <= 0:
        return

    N = pml.thickness
    m = pml.order
    dx = grid.cfg.dx
    L = N * dx
    eta1 = 1.0

    sigma_max = -(m + 1.0) * np.log(max(pml.delta, 1e-30)) / (eta1 * L)

    def grading(d: float) -> float:
        if d <= 0.0:
            return 0.0

        return sigma_max * (d / L) ** m

    nE = grid.nE
    nH = grid.nH

    for i in range(N):
        d = (N - i) * dx
        sig = grading(d)
        grid.sigma_e[i] = sig

    for i in range(N):
        d = (N - i - 0.5) * dx
        sig = grading(d)
        eps_local = grid.eps[min(i, nE - 1)]
        grid.sigma_m[i] = sig * grid.mu[i] / eps_local

    right_start_E = nE - N

    for i in range(right_start_E, nE):
        d = (i - right_start_E) * dx
        sig = grading(d)
        grid.sigma_e[i] = sig

    right_start_H = nH - N
    if right_start_H < 0:
        right_start_H = 0

    for i in range(right_start_H, nH):
        d = (i + 0.5 - (nE - N)) * dx
        if d < 0.0:
            d = 0.0

        sig = grading(d)
        eps_local = grid.eps[min(i, nE - 1)]
        grid.sigma_m[i] = sig * grid.mu[i] / eps_local

    grid.recompute_coefficients()