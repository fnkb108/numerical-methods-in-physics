import numpy as np
from dataclasses import dataclass


@dataclass
class GridConfig:
    num_cells: int = 1000
    dx: float = 1.0
    courant: float = 0.5
    c_speed: float = 1.0

    @property
    def dt(self) -> float:
        return self.courant * self.dx

    @property
    def dt_phys(self) -> float:
        return self.courant * self.dx / self.c_speed


class Grid:
    def __init__(self, cfg: GridConfig):
        self.cfg = cfg
        self.nE = cfg.num_cells
        self.nH = cfg.num_cells - 1

        self.Ey = np.zeros(self.nE)
        self.Hz = np.zeros(self.nH)

        self.eps = np.ones(self.nE)
        self.mu = np.ones(self.nH)

        self.sigma_e = np.zeros(self.nE)
        self.sigma_m = np.zeros(self.nH)

        self.Ca = np.zeros(self.nE)
        self.Cb = np.zeros(self.nE)
        self.Da = np.zeros(self.nH)
        self.Db = np.zeros(self.nH)

        self.recompute_coefficients()

    def recompute_coefficients(self):
        dt = self.cfg.dt
        dx = self.cfg.dx

        for i in range(self.nE):
            denom = 2.0 * self.eps[i] + self.sigma_e[i] * dt
            self.Ca[i] = (2.0 * self.eps[i] - self.sigma_e[i] * dt) / denom
            self.Cb[i] = (2.0 * dt / dx) / denom

        for i in range(self.nH):
            denom = 2.0 * self.mu[i] + self.sigma_m[i] * dt
            self.Da[i] = (2.0 * self.mu[i] - self.sigma_m[i] * dt) / denom
            self.Db[i] = (2.0 * dt / dx) / denom

    def reset_fields(self):
        self.Ey.fill(0.0)
        self.Hz.fill(0.0)

    def x_E(self, i: int) -> float:
        return i * self.cfg.dx

    def x_H(self, i: int) -> float:
        return (i + 0.5) * self.cfg.dx


def update_H(grid: Grid):
    dE = grid.Ey[1:] - grid.Ey[:-1]
    grid.Hz = grid.Da * grid.Hz + grid.Db * dE


def update_E(grid: Grid):
    dH = np.zeros_like(grid.Ey)
    dH[1:-1] = grid.Hz[1:] - grid.Hz[:-1]
    dH[0] = grid.Hz[0]
    dH[-1] = -grid.Hz[-1]
    grid.Ey = grid.Ca * grid.Ey + grid.Cb * dH