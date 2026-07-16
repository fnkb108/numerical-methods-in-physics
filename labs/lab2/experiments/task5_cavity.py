import matplotlib.pyplot as plt
import numpy as np

from core.grid import Grid, GridConfig, update_E, update_H
from core.materials import Layer, apply_layers
from core.monitors import freqs_for_range
from core.pml import PMLConfig, apply_pml
from core.simulation import Simulation, SimulationConfig
from core.sources import (
    Source,
    SourceMode,
    make_cw,
    make_gaussian_for_range,
)
from models.results import CavityResult


def run_bragg_cavity(
    n_periods: int,
    cavity_nm: float,
    lambda_center_nm: float,
    label: str,
) -> CavityResult:
    c0 = 3e8
    lam_min = 300e-9
    lam_max = 900e-9

    n_SiO2 = 1.45
    n_TiO2 = 2.28
    eps_SiO2 = n_SiO2 ** 2
    eps_TiO2 = n_TiO2 ** 2

    lam_c = lambda_center_nm * 1e-9
    L_SiO2 = lam_c / (4.0 * n_SiO2)
    L_TiO2 = lam_c / (4.0 * n_TiO2)
    L_cavity = cavity_nm * 1e-9
    period = L_SiO2 + L_TiO2

    dx = 5e-9
    pmlN = 40
    Q = 0.5

    struct_len = 2.0 * n_periods * period + L_cavity
    struct_cells = int(np.ceil(struct_len / dx)) + 8

    left_gap_cells = 120
    right_gap_cells = 120

    src_pos = pmlN + 30
    struct_start = pmlN + left_gap_cells
    struct_end = struct_start + struct_cells

    mon_ref = struct_start - 30
    mon_trans = struct_end + 30

    N_total = struct_end + right_gap_cells + pmlN
    num_steps = 160000

    freqs = freqs_for_range(lam_min, lam_max, 800, c0)

    cfg = SimulationConfig(
        num_cells=N_total,
        dx=dx,
        courant=Q,
        c_speed=c0,
        num_steps=num_steps,
        pml_thickness=pmlN,
        pml_order=3,
        pml_delta=1e-8,
    )

    sim = Simulation(cfg)

    x = struct_start * dx

    for _ in range(n_periods):
        sim.add_layer(x, x + L_TiO2, eps_r=eps_TiO2)
        x += L_TiO2

        sim.add_layer(x, x + L_SiO2, eps_r=eps_SiO2)
        x += L_SiO2

    x += L_cavity

    for _ in range(n_periods):
        sim.add_layer(x, x + L_SiO2, eps_r=eps_SiO2)
        x += L_SiO2

        sim.add_layer(x, x + L_TiO2, eps_r=eps_TiO2)
        x += L_TiO2

    wf_range, _, _ = make_gaussian_for_range(
        lam_min,
        lam_max,
        c_speed=c0,
        cutoff=6.0,
    )

    sim.set_source(src_pos, wf_range, SourceMode.SOFT)
    sim.add_monitor(mon_ref, freqs)
    sim.add_monitor(mon_trans, freqs)

    sim.build()
    sim.run()

    norm = sim.run_normalization()
    rt = sim.compute_RT(norm)

    lam_nm = rt.wavelengths * 1e9

    return CavityResult(
        lam_nm=lam_nm,
        R=rt.R,
        T=rt.T,
        RTsum=rt.R + rt.T,
        label=label,
    )


def run_pure_phc(
    total_periods: int,
    lambda_center_nm: float,
) -> CavityResult:
    c0 = 3e8
    lam_min = 300e-9
    lam_max = 900e-9

    n_SiO2 = 1.45
    n_TiO2 = 2.28
    eps_SiO2 = n_SiO2 ** 2
    eps_TiO2 = n_TiO2 ** 2

    lam_c = lambda_center_nm * 1e-9
    L_SiO2 = lam_c / (4.0 * n_SiO2)
    L_TiO2 = lam_c / (4.0 * n_TiO2)
    period = L_SiO2 + L_TiO2

    dx = 5e-9
    pmlN = 40
    Q = 0.5

    struct_len = total_periods * period
    struct_cells = int(np.ceil(struct_len / dx)) + 8

    left_gap_cells = 120
    right_gap_cells = 120

    srcpos = pmlN + 30
    struct_start = pmlN + left_gap_cells
    struct_end = struct_start + struct_cells

    mon_ref = struct_start - 30
    mon_trans = struct_end + 30

    N_total = struct_end + right_gap_cells + pmlN
    num_steps = 120000

    freqs = freqs_for_range(lam_min, lam_max, 800, c0)

    cfg = SimulationConfig(
        num_cells=N_total,
        dx=dx,
        courant=Q,
        c_speed=c0,
        num_steps=num_steps,
        pml_thickness=pmlN,
        pml_order=3,
        pml_delta=1e-8,
    )

    sim = Simulation(cfg)

    x = struct_start * dx

    for _ in range(total_periods):
        sim.add_layer(x, x + L_TiO2, eps_r=eps_TiO2)
        x += L_TiO2

        sim.add_layer(x, x + L_SiO2, eps_r=eps_SiO2)
        x += L_SiO2

    wf_range, _, _ = make_gaussian_for_range(
        lam_min,
        lam_max,
        c_speed=c0,
        cutoff=6.0,
    )

    sim.set_source(srcpos, wf_range, SourceMode.SOFT)
    sim.add_monitor(mon_ref, freqs)
    sim.add_monitor(mon_trans, freqs)

    sim.build()
    sim.run()

    norm = sim.run_normalization()
    rt = sim.compute_RT(norm)

    lam_nm = rt.wavelengths * 1e9

    return CavityResult(
        lam_nm=lam_nm,
        R=rt.R,
        T=rt.T,
        RTsum=rt.R + rt.T,
        label=f"Чистый ФК, {total_periods} периодов",
    )


def run_field_snapshots_gaussian(
    n_periods: int,
    cavity_nm: float,
    lambda_center_nm: float,
):
    c0 = 3e8
    lam_min = 300e-9
    lam_max = 900e-9

    n_SiO2 = 1.45
    n_TiO2 = 2.28
    eps_SiO2 = n_SiO2 ** 2
    eps_TiO2 = n_TiO2 ** 2

    lam_c = lambda_center_nm * 1e-9
    L_SiO2 = lam_c / (4.0 * n_SiO2)
    L_TiO2 = lam_c / (4.0 * n_TiO2)
    L_cavity = cavity_nm * 1e-9
    period = L_SiO2 + L_TiO2

    dx = 5e-9
    pmlN = 40
    Q = 0.5

    struct_len = 2.0 * n_periods * period + L_cavity
    struct_cells = int(np.ceil(struct_len / dx)) + 8

    left_gap_cells = 120
    right_gap_cells = 120

    src_pos = pmlN + 30
    struct_start = pmlN + left_gap_cells
    struct_end = struct_start + struct_cells
    N_total = struct_end + right_gap_cells + pmlN

    gc = GridConfig(
        num_cells=N_total,
        dx=dx,
        courant=Q,
        c_speed=c0,
    )

    grid = Grid(gc)

    layers = []
    x = struct_start * dx

    for _ in range(n_periods):
        layers.append(Layer(x, x + L_TiO2, eps_r=eps_TiO2))
        x += L_TiO2

        layers.append(Layer(x, x + L_SiO2, eps_r=eps_SiO2))
        x += L_SiO2

    cavity_x0 = x
    x += L_cavity
    cavity_x1 = x

    for _ in range(n_periods):
        layers.append(Layer(x, x + L_SiO2, eps_r=eps_SiO2))
        x += L_SiO2

        layers.append(Layer(x, x + L_TiO2, eps_r=eps_TiO2))
        x += L_TiO2

    apply_layers(grid, layers)
    apply_pml(
        grid,
        PMLConfig(thickness=pmlN, order=3, delta=1e-8),
    )

    wf_range, _, _ = make_gaussian_for_range(
        lam_min,
        lam_max,
        c_speed=c0,
        cutoff=6.0,
    )

    source = Source(
        index=src_pos,
        waveform=wf_range,
        mode=SourceMode.SOFT,
    )

    num_steps = 90000
    snap_times = [3000, 8000, 15000, 30000, 50000, 80000]

    snapEy = []
    snaplabels = []

    x_nm = np.array(
        [grid.x_E(i) for i in range(grid.nE)]
    ) * 1e9

    snap_idx = 0

    for n in range(num_steps):
        update_H(grid)
        source.inject(grid, n)
        update_E(grid)

        if (
            snap_idx < len(snap_times)
            and n == snap_times[snap_idx]
        ):
            snapEy.append(grid.Ey.copy())
            snaplabels.append(n)
            snap_idx += 1

    plt.figure(figsize=(12, 6))

    for s, lab in zip(snapEy, snaplabels):
        plt.plot(
            x_nm,
            s,
            label=f"шаг {lab}",
            linewidth=1.2,
        )

    plt.axvline(
        cavity_x0 * 1e9,
        linestyle="--",
        color="gray",
        label="границы полости",
    )

    plt.axvline(
        cavity_x1 * 1e9,
        linestyle="--",
        color="gray",
    )

    plt.xlabel("Координата x, нм")
    plt.ylabel("Поле E_y")
    plt.title(
        "Эволюция поля гауссовского импульса "
        "в брэгговской микрополости"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_field_snapshots_cw(
    n_periods: int,
    cavity_nm: float,
    lambda_center_nm: float,
):
    c0 = 3e8

    n_SiO2 = 1.45
    n_TiO2 = 2.28
    eps_SiO2 = n_SiO2 ** 2
    eps_TiO2 = n_TiO2 ** 2

    lam_c = lambda_center_nm * 1e-9
    f0 = c0 / lam_c

    L_SiO2 = lam_c / (4.0 * n_SiO2)
    L_TiO2 = lam_c / (4.0 * n_TiO2)
    L_cavity = cavity_nm * 1e-9
    period = L_SiO2 + L_TiO2

    dx = 5e-9
    pmlN = 40
    Q = 0.5

    struct_len = 2.0 * n_periods * period + L_cavity
    struct_cells = int(np.ceil(struct_len / dx)) + 8

    left_gap_cells = 120
    right_gap_cells = 120

    src_pos = pmlN + 30
    struct_start = pmlN + left_gap_cells
    struct_end = struct_start + struct_cells
    N_total = struct_end + right_gap_cells + pmlN

    gc = GridConfig(
        num_cells=N_total,
        dx=dx,
        courant=Q,
        c_speed=c0,
    )

    grid = Grid(gc)

    layers = []
    x = struct_start * dx

    for _ in range(n_periods):
        layers.append(Layer(x, x + L_TiO2, eps_r=eps_TiO2))
        x += L_TiO2

        layers.append(Layer(x, x + L_SiO2, eps_r=eps_SiO2))
        x += L_SiO2

    cavity_x0 = x
    x += L_cavity
    cavity_x1 = x

    for _ in range(n_periods):
        layers.append(Layer(x, x + L_SiO2, eps_r=eps_SiO2))
        x += L_SiO2

        layers.append(Layer(x, x + L_TiO2, eps_r=eps_TiO2))
        x += L_TiO2

    apply_layers(grid, layers)

    apply_pml(
        grid,
        PMLConfig(thickness=pmlN, order=3, delta=1e-8),
    )

    wf_cw = make_cw(
        freq=f0,
        width=20.0 / f0,
        slowness=4.0,
    )

    source = Source(
        index=src_pos,
        waveform=wf_cw,
        mode=SourceMode.SOFT,
    )

    num_steps = 120000
    snap_times = [80000, 90000, 100000, 110000]

    snapEy = []
    snaplabels = []

    x_nm = np.array(
        [grid.x_E(i) for i in range(grid.nE)]
    ) * 1e9

    snap_idx = 0

    for n in range(num_steps):
        update_H(grid)
        source.inject(grid, n)
        update_E(grid)

        if (
            snap_idx < len(snap_times)
            and n == snap_times[snap_idx]
        ):
            snapEy.append(grid.Ey.copy())
            snaplabels.append(n)
            snap_idx += 1

    plt.figure(figsize=(12, 6))

    for s, lab in zip(snapEy, snaplabels):
        plt.plot(
            x_nm,
            s,
            label=f"шаг {lab}",
            linewidth=1.3,
        )

    plt.axvline(
        cavity_x0 * 1e9,
        linestyle="--",
        color="gray",
        label="границы полости",
    )

    plt.axvline(
        cavity_x1 * 1e9,
        linestyle="--",
        color="gray",
    )

    plt.xlabel("Координата x, нм")
    plt.ylabel("Поле E_y")
    plt.title(
        "Локализация монохроматического поля "
        "в брэгговской микрополости"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def task5():
    print("Task 5: Bragg Microcavity")

    lam_center = 650.0
    cavity_nm = lam_center / 2.0

    nper_list = [4, 8]

    for npers in nper_list:
        r = run_bragg_cavity(
            npers,
            cavity_nm,
            lam_center,
            f"{npers} периодов с каждой стороны",
        )

        plt.figure(figsize=(10, 6))

        plt.plot(
            r.lam_nm,
            r.R,
            label="R (отражение)",
            linewidth=2,
        )

        plt.plot(
            r.lam_nm,
            r.T,
            label="T (прохождение)",
            linewidth=2,
        )

        plt.plot(
            r.lam_nm,
            r.RTsum,
            "--",
            label="R+T",
            linewidth=1.2,
            color="gray",
        )

        plt.xlabel("Длина волны, нм")
        plt.ylabel("Коэффициенты R, T")
        plt.title(f"Брэгговская микрополость, {r.label}")
        plt.legend()
        plt.xlim(300.0, 900.0)
        plt.ylim(0.0, 1.1)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    nper = 8

    rcav = run_bragg_cavity(
        nper,
        cavity_nm,
        lam_center,
        f"Полость {nper}/{nper}",
    )

    rpure = run_pure_phc(2 * nper, lam_center)

    plt.figure(figsize=(10, 6))

    plt.plot(
        rcav.lam_nm,
        rcav.T,
        label="T брэгговской микрополости",
        linewidth=2,
    )

    plt.plot(
        rpure.lam_nm,
        rpure.T,
        "--",
        label="T чистого фотонного кристалла",
        linewidth=2,
    )

    plt.xlabel("Длина волны, нм")
    plt.ylabel("Коэффициент прохождения T")
    plt.title(
        "Сравнение передачи: брэгговская микрополость "
        "и чистый ФК"
    )
    plt.legend()
    plt.xlim(300.0, 900.0)
    plt.ylim(0.0, 1.1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    run_field_snapshots_gaussian(
        8,
        cavity_nm,
        lam_center,
    )

    run_field_snapshots_cw(
        8,
        cavity_nm,
        lam_center,
    )