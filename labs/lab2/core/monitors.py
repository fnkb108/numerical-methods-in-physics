import numpy as np
from dataclasses import dataclass
from grid import Grid


@dataclass
class DFTMonitor:
    index: int
    freqs: np.ndarray
    dt_phys: float

    def __post_init__(self):
        self.Edft = np.zeros(self.freqs.size, dtype=np.complex128)

    def accumulate_after_E(self, grid: Grid, n: int):
        t = (n + 0.5) * self.dt_phys
        e_val = grid.Ey[self.index]
        phase = np.exp(-1j * 2.0 * np.pi * self.freqs * t)
        self.Edft += e_val * phase


def freqs_for_range(lambda_min: float,
                    lambda_max: float,
                    N: int,
                    c: float) -> np.ndarray:
    f_max = c / lambda_min
    f_min = c / lambda_max
    return np.linspace(f_min, f_max, N)