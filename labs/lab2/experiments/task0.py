import numpy as np
import matplotlib.pyplot as plt

from core.sources import SourceMode, make_cw, make_gaussian_pulse
from experiments.common import run_and_collect
from visual.plotting import plot_snapshots


def task0():
    N = 500
    dx = 1.0
    Q = 0.5
    x = np.arange(N) * dx

    pulse_gauss = make_gaussian_pulse(
        freq_center=0.05,
        freq_width=0.03,
        cutoff=5.0,
    )

    pulse_cw = make_cw(
        freq=0.05,
        width=100.0,
        slowness=10.0,
    )

    x1, snaps1 = run_and_collect(
        N=N,
        dx=dx,
        Q=Q,
        num_steps=1200,
        src_pos=100,
        waveform=pulse_gauss,
        mode=SourceMode.SOFT,
        use_pml=False,
        pmlN=0,
        snap_interval=300,
    )

    plot_snapshots(x1, snaps1, "Gaussian pulse, soft source")

    x2, snaps2 = run_and_collect(
        N=N,
        dx=dx,
        Q=Q,
        num_steps=600 + 1,
        src_pos=250,
        waveform=pulse_cw,
        mode=SourceMode.SOFT,
        use_pml=False,
        pmlN=0,
        snap_interval=200,
    )

    plot_snapshots(x2, snaps2, "CW source, soft source")

    x3, snaps3 = run_and_collect(
        N=N,
        dx=dx,
        Q=Q,
        num_steps=1200,
        src_pos=100,
        waveform=pulse_gauss,
        mode=SourceMode.CURRENT,
        use_pml=False,
        pmlN=0,
        snap_interval=300,
    )

    plot_snapshots(x3, snaps3, "Gaussian pulse, current source")

    x4, snaps4 = run_and_collect(
        N=N,
        dx=dx,
        Q=Q,
        num_steps=600 + 1,
        src_pos=250,
        waveform=pulse_cw,
        mode=SourceMode.CURRENT,
        use_pml=False,
        pmlN=0,
        snap_interval=200,
    )

    plot_snapshots(x4, snaps4, "CW source, current source")

    snap_time = 1000

    _, snaps_soft = run_and_collect(
        N=N,
        dx=dx,
        Q=Q,
        num_steps=snap_time + 1,
        src_pos=250,
        waveform=pulse_gauss,
        mode=SourceMode.SOFT,
        use_pml=False,
        pmlN=0,
        snap_interval=snap_time,
    )

    _, snaps_cur = run_and_collect(
        N=N,
        dx=dx,
        Q=Q,
        num_steps=snap_time + 1,
        src_pos=250,
        waveform=pulse_gauss,
        mode=SourceMode.CURRENT,
        use_pml=False,
        pmlN=0,
        snap_interval=snap_time,
    )

    if snaps_soft and snaps_cur:
        plt.figure(figsize=(10, 5))
        plt.plot(x, snaps_soft[0].Ey, label="Soft source", linewidth=2.0)
        plt.plot(
            x,
            snaps_cur[0].Ey,
            "--",
            label="Current source",
            linewidth=1.5,
        )
        plt.xlabel("x (cells)")
        plt.ylabel("E_y")
        plt.title(
            f"Gaussian soft and current sources, "
            f"step {snaps_soft[0].step}"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()