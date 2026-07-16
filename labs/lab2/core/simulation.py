import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from grid import Grid, GridConfig, update_E, update_H
from materials import Layer, apply_layers
from monitors import DFTMonitor
from pml import PMLConfig, apply_pml
from sources import Waveform, Source, SourceMode


@dataclass
class SimulationConfig:
    num_cells: int
    dx: float
    courant: float
    c_speed: float
    num_steps: int
    pml_thickness: int
    pml_order: int = 3
    pml_delta: float = 1e-6

@dataclass
class RTResult:
    freqs: np.ndarray
    wavelengths: np.ndarray
    R: np.ndarray
    T: np.ndarray

class Simulation:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        gc = GridConfig(
            num_cells=cfg.num_cells,
            dx=cfg.dx,
            courant=cfg.courant,
            c_speed=cfg.c_speed
        )
        self.grid = Grid(gc)
        self.layers: List[Layer] = []
        self.source: Optional[Source] = None
        self.monitors: List[DFTMonitor] = []

    def add_layer(self, x_start: float, x_end: float,
                  eps_r: float, mu_r: float = 1.0, sigma: float = 0.0):
        self.layers.append(Layer(x_start, x_end, eps_r, mu_r, sigma))

    def set_source(self, index: int, wf: Waveform, mode: str = SourceMode.SOFT):
        self.source = Source(index=index, waveform=wf, mode=mode)

    def add_monitor(self, indexE: int, freqs: np.ndarray):
        self.monitors.append(
            DFTMonitor(index=indexE, freqs=freqs, dt_phys=self.grid.cfg.dt_phys)
        )


    def build(self):
        self.grid.eps[:] = 1.0
        self.grid.mu[:] = 1.0
        self.grid.sigma_e[:] = 0.0
        self.grid.sigma_m[:] = 0.0

        if self.layers:
            apply_layers(self.grid, self.layers)

        pml_cfg = PMLConfig(
            thickness=self.cfg.pml_thickness,
            order=self.cfg.pml_order,
            delta=self.cfg.pml_delta
        )
        apply_pml(self.grid, pml_cfg)


    def run(self, snapshot_interval: int = 0, callback=None):
        for n in range(self.cfg.num_steps):
            update_E(self.grid)

            if self.source is not None:
                self.source.inject(self.grid, n)

            for mon in self.monitors:
                mon.accumulate_after_E(self.grid, n)

            update_H(self.grid)

            if callback is not None and snapshot_interval > 0 and (n % snapshot_interval == 0):
                callback(n, self.grid)


    def run_normalization(self) -> List[DFTMonitor]:
        gc = GridConfig(
            num_cells=self.cfg.num_cells,
            dx=self.cfg.dx,
            courant=self.cfg.courant,
            c_speed=self.cfg.c_speed
        )
        norm_grid = Grid(gc)

        pml_cfg = PMLConfig(
            thickness=self.cfg.pml_thickness,
            order=self.cfg.pml_order,
            delta=self.cfg.pml_delta
        )
        apply_pml(norm_grid, pml_cfg)

        norm_monitors = [
            DFTMonitor(index=mon.index, freqs=mon.freqs.copy(), dt_phys=norm_grid.cfg.dt_phys)
            for mon in self.monitors
        ]

        norm_source = None
        if self.source is not None:
            norm_source = Source(
                index=self.source.index,
                waveform=self.source.waveform,
                mode=self.source.mode
            )

        for n in range(self.cfg.num_steps):
            update_E(norm_grid)

            if norm_source is not None:
                norm_source.inject(norm_grid, n)

            for mon in norm_monitors:
                mon.accumulate_after_E(norm_grid, n)

            update_H(norm_grid)

        return norm_monitors

    def compute_RT(self, norm_monitors: List[DFTMonitor]) -> RTResult:
        if len(self.monitors) < 2 or len(norm_monitors) < 2:
            raise RuntimeError("Need at least 2 monitors for reflection/transmission")

        Etotal_ref = self.monitors[0].Edft
        Etotal_trans = self.monitors[1].Edft
        Einc_ref = norm_monitors[0].Edft
        Einc_trans = norm_monitors[1].Edft

        freqs = self.monitors[0].freqs
        nf = freqs.size

        max_inc2 = np.max(np.abs(Einc_ref) ** 2)
        threshold = max_inc2 * 1e-4

        R = np.zeros(nf)
        T = np.zeros(nf)

        for k in range(nf):
            Einc2 = np.abs(Einc_ref[k]) ** 2
            if Einc2 > threshold:
                Eref = Etotal_ref[k] - Einc_ref[k]
                R[k] = np.abs(Eref) ** 2 / Einc2

                Einc_trans2 = np.abs(Einc_trans[k]) ** 2
                if Einc_trans2 > threshold:
                    T[k] = np.abs(Etotal_trans[k]) ** 2 / Einc_trans2
                else:
                    T[k] = np.abs(Etotal_trans[k]) ** 2 / Einc2
            else:
                R[k] = 0.0
                T[k] = 0.0

        wavelengths = self.cfg.c_speed / freqs
        return RTResult(freqs=freqs, wavelengths=wavelengths, R=R, T=T)
