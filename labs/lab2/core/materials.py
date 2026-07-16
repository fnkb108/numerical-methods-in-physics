from dataclasses import dataclass
from typing import List
from grid import Grid


@dataclass
class Layer:
    x_start: float
    x_end: float
    eps_r: float
    mu_r: float = 1.0
    sigma: float = 0.0


def apply_layers(grid: Grid, layers: List[Layer]):
    grid.eps[:] = 1.0
    grid.mu[:] = 1.0
    grid.sigma_e[:] = 0.0

    for layer in layers:
        for i in range(grid.nE):
            x = grid.x_E(i)
            if layer.x_start <= x < layer.x_end:
                grid.eps[i] = layer.eps_r
                grid.sigma_e[i] = layer.sigma

        for i in range(grid.nH):
            x = grid.x_H(i)
            if layer.x_start <= x < layer.x_end:
                grid.mu[i] = layer.mu_r

    grid.recompute_coefficients()