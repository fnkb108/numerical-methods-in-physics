import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple
from grid import Grid


Waveform = Callable[[float], float]


class SourceMode:
    SOFT = "soft"
    HARD = "hard"
    CURRENT = "current"

@dataclass
class Source:
    index: int
    waveform: Waveform
    mode: str = SourceMode.SOFT

    def inject(self, grid: Grid, n: int):
        if self.waveform is None:
            return
        if not (0 <= self.index < grid.nE):
            raise IndexError("Source index outside E-grid")

        t = (n + 0.5) * grid.cfg.dt_phys
        val = self.waveform(t)

        if self.mode == SourceMode.SOFT:
            grid.Ey[self.index] += val
        elif self.mode == SourceMode.HARD:
            grid.Ey[self.index] = val
        elif self.mode == SourceMode.CURRENT:
            grid.Ey[self.index] -= grid.Cb[self.index] * grid.cfg.dx * val
        else:
            raise ValueError("Unknown source mode")

def make_cw(freq: float, width: float = 0.0, slowness: float = 3.0) -> Waveform:
    def f(t: float) -> float:
        envelope = 1.0
        if width > 0.0:
            envelope = 0.5 * (1.0 + np.tanh((t - width) / slowness))
        return envelope * np.sin(2.0 * np.pi * freq * t)
    return f


def make_gaussian_pulse(freq_center: float,
                        freq_width: float,
                        cutoff: float = 5.0) -> Waveform:
    w = 1.0 / freq_width
    t0 = cutoff * w

    def f(t: float) -> float:
        env = np.exp(-0.5 * ((t - t0) / w) ** 2)
        return env * np.sin(2.0 * np.pi * freq_center * t)
    return f


def make_gaussian_for_range(lambda_min: float,
                            lambda_max: float,
                            c_speed: float = 1.0,
                            cutoff: float = 6.0) -> Tuple[Waveform, float, float]:
    f_max = c_speed / lambda_min
    f_min = c_speed / lambda_max
    f_center = 0.5 * (f_max + f_min)
    f_width = 1.5 * (f_max - f_min)
    return make_gaussian_pulse(f_center, f_width, cutoff), f_center, f_width