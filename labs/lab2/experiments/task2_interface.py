import matplotlib.pyplot as plt
import numpy as np

from core.monitors import freqs_for_range
from core.simulation import Simulation, SimulationConfig
from core.sources import SourceMode, make_gaussian_for_range


def task2():
    c0 = 3e8
    lambda_min = 300e-9
    lambda_max = 900e-9

    n_quartz = 1.45
    eps_quartz = n_quartz ** 2

    dx = 10e-9
    pmlN = 100
    N_domain = 400
    N_total = N_domain + 2 * pmlN
    Q = 0.5
    num_steps = 30000

    src_pos = pmlN + 50
    mon_ref = pmlN + 100
    interface_pos = pmlN + N_domain // 2
    mon_trans = interface_pos + 50

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

    x_interface = interface_pos * dx
    x_end = N_total * dx

    sim.add_layer(
        x_interface,
        x_end,
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

    R_theory = ((1.0 - n_quartz) / (1.0 + n_quartz)) ** 2
    T_theory = 1.0 - R_theory

    lam_nm = rt.wavelengths * 1e9
    R_th = np.full_like(lam_nm, R_theory)
    T_th = np.full_like(lam_nm, T_theory)

    plt.figure(figsize=(10, 6))

    plt.plot(
        lam_nm,
        rt.R,
        label="Численный коэффициент отражения R",
        linewidth=2,
    )

    plt.plot(
        lam_nm,
        R_th,
        "--",
        label=f"Теоретический R = {R_theory:.3f}",
        linewidth=2,
        color="red",
    )

    plt.plot(
        lam_nm,
        rt.T * n_quartz,
        label="Численный коэффициент прохождения T",
        linewidth=2,
    )

    plt.plot(
        lam_nm,
        T_th,
        "--",
        label=f"Теоретический T = {T_theory:.3f}",
        linewidth=2,
        color="green",
    )

    plt.xlabel("Длина волны (нм)")
    plt.ylabel("Коэффициенты")
    plt.title(
        "Отражение и прохождение для бесконечного "
        "диэлектрика (кварца с n=1.45)"
    )
    plt.legend()
    plt.xlim(300.0, 900.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()