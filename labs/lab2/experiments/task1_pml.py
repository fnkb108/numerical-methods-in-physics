import matplotlib.pyplot as plt
import numpy as np

from core.grid import Grid, GridConfig, update_E, update_H
from core.monitors import DFTMonitor
from core.pml import PMLConfig, apply_pml
from core.sources import Source, SourceMode, Waveform, make_gaussian_pulse
from models.results import PMLResult


def run_once(
    N: int,
    dx: float,
    Q: float,
    num_steps: int,
    src_pos: int,
    mon_pos: int,
    waveform: Waveform,
    pml_thickness: int,
    pml_order: int,
    pml_delta: float,
    freqs: np.ndarray,
) -> np.ndarray:
    cfg = GridConfig(num_cells=N, dx=dx, courant=Q, c_speed=1.0)
    grid = Grid(cfg)

    apply_pml(
        grid,
        PMLConfig(
            thickness=pml_thickness,
            order=pml_order,
            delta=pml_delta,
        ),
    )

    source = Source(
        index=src_pos,
        waveform=waveform,
        mode=SourceMode.SOFT,
    )

    monitor = DFTMonitor(
        index=mon_pos,
        freqs=freqs,
        dt_phys=cfg.dt_phys,
    )

    for n in range(num_steps):
        update_H(grid)
        source.inject(grid, n)
        update_E(grid)
        monitor.accumulate_after_E(grid, n)

    return monitor.Edft


def run_pml_experiment(
    pml_thickness: int,
    pml_order: int,
    pml_delta: float = 1e-6,
) -> PMLResult:
    N = 800
    dx = 1.0
    Q = 0.5
    num_steps = 8000

    f_min = 0.01
    f_max = 0.5
    f_center = 0.22
    f_width = 1.5

    src_pos = pml_thickness + 30
    mon_pos = N // 2

    nf = 300
    freqs = np.zeros(nf)

    for i in range(nf):
        freqs[i] = f_min + (f_max - f_min) * i / (nf - 1.0)

    waveform = make_gaussian_pulse(
        f_center,
        f_width,
        cutoff=6.0,
    )

    E_total = run_once(
        N,
        dx,
        Q,
        num_steps,
        src_pos,
        mon_pos,
        waveform,
        pml_thickness,
        pml_order,
        pml_delta,
        freqs,
    )

    ref_pml_thick = 100
    ref_pml_order = 3
    ref_pml_delta = 1e-12

    N_ref = N + 2 * (ref_pml_thick - pml_thickness)
    src_ref = ref_pml_thick + 30
    mon_ref = src_ref + (mon_pos - src_pos)

    E_inc = run_once(
        N_ref,
        dx,
        Q,
        num_steps,
        src_ref,
        mon_ref,
        waveform,
        ref_pml_thick,
        ref_pml_order,
        ref_pml_delta,
        freqs,
    )

    result = PMLResult(
        freqs=freqs.copy(),
        R=np.zeros(nf),
        R_max=0.0,
    )

    max_inc2 = 0.0

    for k in range(nf):
        max_inc2 = max(max_inc2, np.abs(E_inc[k]) ** 2)

    threshold = max_inc2 * 1e-2

    R_raw = np.zeros(nf)
    valid = np.zeros(nf, dtype=bool)

    for k in range(nf):
        inc2 = np.abs(E_inc[k]) ** 2

        if inc2 > threshold:
            E_ref = E_total[k] - E_inc[k]
            R_raw[k] = min(np.abs(E_ref) ** 2 / inc2, 1.0)
            valid[k] = True

    half_window = 2

    for k in range(nf):
        if not valid[k]:
            result.R[k] = 0.0
            continue

        s = 0.0
        cnt = 0
        k1 = max(0, k - half_window)
        k2 = min(nf - 1, k + half_window)

        for j in range(k1, k2 + 1):
            if valid[j]:
                s += R_raw[j]
                cnt += 1

        result.R[k] = (s / cnt) if cnt > 0 else 0.0

        if 0.05 <= freqs[k] <= 0.30:
            result.R_max = max(result.R_max, result.R[k])

    return result


def task1():
    fixed_width = 60
    orders = [0, 2, 3]

    names = {
        0: "постоянный профиль",
        2: "квадратичный профиль",
        3: "кубический профиль",
    }

    results = []

    for ord_ in orders:
        print(
            f"Running PML experiment: "
            f"order={ord_}, width={fixed_width}"
        )

        res = run_pml_experiment(fixed_width, ord_)

        print("Rmax =", res.R_max)
        results.append(res)

    plt.figure(figsize=(10, 6))

    fmin_plot = 0.03
    fmax_plot = 0.333

    for res, ord_ in zip(results, orders):
        freqs_plot = res.freqs.copy()
        R_plot = res.R.copy()

        mask = (
            (freqs_plot >= fmin_plot)
            & (freqs_plot <= fmax_plot)
        )

        R_plot[~mask] = np.nan

        plt.semilogy(
            freqs_plot,
            np.maximum(R_plot, 1e-7),
            label=names[ord_],
            linewidth=2,
        )

    plt.xlabel("Частота")
    plt.ylabel("Коэффициент отражения R(f)")
    plt.title(
        "Коэффициент отражения в зависимости от частоты, "
        f"ширина профиля PML = {fixed_width} ячеек"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    widths = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70]
    rmax_curves = {ord_: [] for ord_ in orders}

    for ord_ in orders:
        for w in widths:
            print(f"order={ord_}, width={w}")

            res = run_pml_experiment(w, ord_)
            rmax_curves[ord_].append(max(res.R_max, 1e-9))

            print(" Rmax =", res.R_max)

    plt.figure(figsize=(10, 6))
    widths_arr = np.array(widths, dtype=float)

    for ord_ in orders:
        plt.semilogy(
            widths_arr,
            rmax_curves[ord_],
            marker="o",
            linewidth=2,
            label=names[ord_],
        )

    plt.xlabel("Ширина PML (ячейки)")
    plt.ylabel("Максимальный коэффициент отражения Rmax")
    plt.title(
        "Максимальный коэффициент отражения "
        "в зависимости от ширины PML"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()