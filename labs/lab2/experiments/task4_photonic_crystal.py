import matplotlib.pyplot as plt
import numpy as np

from core.monitors import freqs_for_range
from core.simulation import Simulation, SimulationConfig
from core.sources import SourceMode, make_gaussian_for_range
from models.results import PhCResult


def run_photonic_crystal(
    num_periods: int,
    L_SiO2_nm: float,
    L_TiO2_nm: float,
    label: str,
    num_steps: int = 120000,
) -> PhCResult:
    print(
        f"PhC: {num_periods} периодов, "
        f"SiO2={L_SiO2_nm} нм, TiO2={L_TiO2_nm} нм"
    )

    c0 = 3e8
    lambda_min = 300e-9
    lambda_max = 900e-9

    n_SiO2 = 1.45
    n_TiO2 = 2.28

    eps_SiO2 = n_SiO2 ** 2
    eps_TiO2 = n_TiO2 ** 2

    L_SiO2 = L_SiO2_nm * 1e-9
    L_TiO2 = L_TiO2_nm * 1e-9
    period = L_SiO2 + L_TiO2

    dx = 5e-9
    Q = 0.5
    pmlN = 40

    left_gap_cells = 140
    right_gap_cells = 140
    src_pos = pmlN + 35

    struct_len = num_periods * period
    struct_cells = int(np.ceil(struct_len / dx))

    struct_start = pmlN + left_gap_cells
    struct_end = struct_start + struct_cells

    mon_ref = src_pos + (struct_start - src_pos) // 2
    mon_trans = struct_end + 60

    N_total = mon_trans + right_gap_cells + pmlN

    freqs = freqs_for_range(
        lambda_min,
        lambda_max,
        700,
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

    x = struct_start * dx

    for _ in range(num_periods):
        sim.add_layer(
            x,
            x + L_SiO2,
            eps_r=eps_SiO2,
        )

        x += L_SiO2

        sim.add_layer(
            x,
            x + L_TiO2,
            eps_r=eps_TiO2,
        )

        x += L_TiO2

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

    res = PhCResult(
        lam_nm=np.zeros(freqs.size),
        R=np.zeros(freqs.size),
        T=np.zeros(freqs.size),
        periods=num_periods,
        label=label,
    )

    for k in range(freqs.size):
        res.lam_nm[k] = rt.wavelengths[k] * 1e9
        res.R[k] = max(0.0, rt.R[k])
        res.T[k] = max(0.0, rt.T[k])

    return res


def task4():
    print("Task 4: Photonic Crystal")

    L_SiO2_opt_nm = 100.0
    L_TiO2_opt_nm = 55.0

    periods_list = [5, 10, 15]

    print("1) Одинаковая толщина слоев 100 нм")

    results_phys = []

    for n_pers in periods_list:
        results_phys.append(
            run_photonic_crystal(
                n_pers,
                100.0,
                100.0,
                f"{n_pers} периодов",
            )
        )

    plt.figure(figsize=(10, 6))

    for r in results_phys:
        plt.plot(
            r.lam_nm,
            r.R,
            label=f"R, {r.label}",
            linewidth=1.8,
        )

    plt.xlabel("Длина волны, нм")
    plt.ylabel("Коэффициент отражения R")
    plt.title(
        "Спектр отражения для фотонного кристалла "
        "SiO₂ 100 нм / TiO₂ 100 нм"
    )
    plt.legend()
    plt.xlim(300.0, 900.0)
    plt.ylim(0.0, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    r15 = results_phys[-1]
    RTsum = r15.R + r15.T

    plt.figure(figsize=(10, 6))
    plt.plot(
        r15.lam_nm,
        r15.R,
        label="R (отражение)",
        linewidth=2.0,
    )
    plt.plot(
        r15.lam_nm,
        r15.T,
        label="T (прохождение)",
        linewidth=2.0,
    )
    plt.plot(
        r15.lam_nm,
        RTsum,
        "--",
        label="R + T",
        linewidth=1.2,
        color="gray",
    )

    plt.xlabel("Длина волны, нм")
    plt.ylabel("Коэффициенты R, T")
    plt.title("SiO₂ 100 нм / TiO₂ 100 нм, 15 периодов")
    plt.legend()
    plt.xlim(300.0, 900.0)
    plt.ylim(0.0, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("2) Одинаковая оптическая толщина")

    results_opt = []

    for n_pers in periods_list:
        results_opt.append(
            run_photonic_crystal(
                n_pers,
                L_SiO2_opt_nm,
                L_TiO2_opt_nm,
                f"{n_pers} периодов",
            )
        )

    plt.figure(figsize=(10, 6))

    for r in results_opt:
        plt.plot(
            r.lam_nm,
            r.R,
            label=f"R, {r.label}",
            linewidth=1.8,
        )

    plt.xlabel("Длина волны, нм")
    plt.ylabel("Коэффициент отражения R")
    plt.title(
        "Фотонный кристалл с одинаковой оптической "
        "толщиной слоёв (SiO₂ 100 нм / TiO₂ 55 нм)"
    )
    plt.legend()
    plt.xlim(300.0, 900.0)
    plt.ylim(0.0, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    r15_opt = results_opt[-1]
    RTsum_opt = r15_opt.R + r15_opt.T

    plt.figure(figsize=(10, 6))
    plt.plot(
        r15_opt.lam_nm,
        r15_opt.R,
        label="R (отражение)",
        linewidth=2.0,
    )
    plt.plot(
        r15_opt.lam_nm,
        r15_opt.T,
        label="T (прохождение)",
        linewidth=2.0,
    )
    plt.plot(
        r15_opt.lam_nm,
        RTsum_opt,
        "--",
        label="R + T",
        linewidth=1.2,
        color="gray",
    )

    plt.xlabel("Длина волны, нм")
    plt.ylabel("Коэффициенты R, T")
    plt.title("SiO₂ 100 нм / TiO₂ 55 нм, 15 периодов")
    plt.legend()
    plt.xlim(300.0, 900.0)
    plt.ylim(0.0, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(
        "3) Сравнение: одинаковая физическая "
        "и оптическая толщина"
    )

    rphys15 = run_photonic_crystal(
        15,
        100.0,
        100.0,
        "100/100 нм",
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        rphys15.lam_nm,
        rphys15.R,
        label="R, SiO₂/TiO₂ = 100/100 нм",
        linewidth=2.0,
    )
    plt.plot(
        r15_opt.lam_nm,
        r15_opt.R,
        label="R, SiO₂/TiO₂ = 100/55 нм",
        linewidth=2.0,
    )

    plt.xlabel("Длина волны (нм)")
    plt.ylabel("Коэффициент отражения R")
    plt.title(
        "Сравнение одинаковой физической и оптической "
        "толщины для 15 периодов"
    )
    plt.legend()
    plt.xlim(300.0, 900.0)
    plt.ylim(0.0, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.show()