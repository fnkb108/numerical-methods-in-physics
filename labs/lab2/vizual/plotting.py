from typing import List

import matplotlib.pyplot as plt
import numpy as np

from models.results import Snapshot


def plot_snapshots(
    x: np.ndarray,
    snaps: List[Snapshot],
    title: str,
):
    plt.figure(figsize=(10, 5))

    for s in snaps:
        plt.plot(x, s.Ey, label=f"шаг {s.step}")

    plt.xlabel("x (ячеек)")
    plt.ylabel("E_y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()