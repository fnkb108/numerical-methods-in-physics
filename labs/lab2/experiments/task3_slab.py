import matplotlib.pyplot as plt
import numpy as np

from core.monitors import freqs_for_range
from core.simulation import Simulation, SimulationConfig
from core.sources import SourceMode, make_gaussian_for_range
from models.results import SlabResult
from core.optics import fabry_perot


def run_slab(L_nm: float, num_steps: int) -> SlabResult:
    c0 = 3e8
    lambda_min = 300e-9
    lambda_max = 900e-9

    n_quartz = 1.45
    eps_quartz = n_quartz ** 2

    L = L_nm * 1e-9

    dx = 5e-9
    pmlN = 100
    N_domain = 400
    N_total = N_domain + 2 * pmlN
    Q = 0.5

    src_pos = pmlN + 60
    mon_ref = pmlN + 120
    center = pmlN + N_domain // 2

    slab_start_i = center - int(L / (2.0 * dx))
    slab_end_i = center + int(L / (2.0 * dx))

    slab_start_x = slab_start_i * dx
    slab_end_x = slab_end_i * dx

    mon_trans = slab_end_i + 60

    freqs = freqs_for_range(
        lambda_min,
        lambda_max,
        500,
        c0,
    )

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

    sim.add_layer(
        slab_start_x,
        slab_end_x,
        eps_r=eps_quartz,
    )

    wf_range, _, _ = make_gaussian_for_range(
        lambda_min,
        lambda_max,
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

    lam = rt.wavelengths
    lam_nm = lam * 1e9

    R_th = np.zeros_like(lam_nm)
    T_th = np.zeros_like(lam_nm)

    for k in range(lam.size):
        fp = fabry_perot(n_quartz, L, lam[k])
        R_th[k] = fp.R
        T_th[k] = fp.T

    return SlabResult(
        lam_nm=lam_nm,
        R=rt.R,
        T=rt.T,
        R_th=R_th,
        T_th=T_th,
        L_nm=L_nm,
        steps=num_steps,
    )


def task3():
    r = run_slab(200.0, 30000)

    plt.figure(figsize=(10, 6))
    plt.plot(
        r.lam_nm,
        r.R,
        label="Численный коэффициент отражения R",
        linewidth=2,
    )
    plt.plot(
        r.lam_nm,
        r.R_th,
        "--",
        label="Теоретический R",
        linewidth=2,
    )
    plt.plot(
        r.lam_nm,
        r.T,
        label="Численный коэффициент прохождения T",
        linewidth=2,
    )
    plt.plot(
        r.lam_nm,
        r.T_th,
        "--",
        label="Теоретический T",
        linewidth=2,
    )

    plt.xlabel("Длина волны (нм)")
    plt.ylabel("Коэффициент")
    plt.title(
        "Отражение от диэлектрической пластины "
        "кварца длиной 200 нм"
    )
    plt.legend()
    plt.xlim(300.0, 900.0)
    plt.ylim(0.0, 1.1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    thicknesses = [100.0, 200.0, 500.0]

    plt.figure(figsize=(10, 6))

    for L_nm in thicknesses:
        rr = run_slab(L_nm, 30000)

        plt.plot(
            rr.lam_nm,
            rr.R,
            label=f"L={int(L_nm)} нм",
            linewidth=1.5,
        )

    plt.xlabel("Длина волны (нм)")
    plt.ylabel("R")
    plt.title(
        "Отражение от кварцевой пластины "
        "для разных значений L"
    )
    plt.legend()
    plt.xlim(300.0, 900.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    step_counts = [1500, 2000, 5000]

    plt.figure(figsize=(10, 6))

    for steps in step_counts:
        rr = run_slab(200.0, steps)

        plt.plot(
            rr.lam_nm,
            rr.R,
            label=f"{steps} шагов",
            linewidth=1.5,
        )

    rref = run_slab(200.0, 30000)

    plt.plot(
        rref.lam_nm,
        rref.R_th,
        "--",
        label="Теория (Фабри-Перо)",
        linewidth=2,
        color="black",
    )

    plt.xlabel("Длина волны (нм)")
    plt.ylabel("R")
    plt.title(
        "Зависимость результатов от времени моделирования"
    )
    plt.legend()
    plt.xlim(300.0, 900.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()