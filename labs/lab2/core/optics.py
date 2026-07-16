import numpy as np

from models.results import FPResult


def fabry_perot(n: float, L: float, Lambda: float) -> FPResult:
    r12 = (1.0 - n) / (1.0 + n)
    delta = 2.0 * np.pi * n * L / Lambda
    r12sq = r12 ** 2

    denom = 1.0 + r12sq ** 2 - 2.0 * r12sq * np.cos(2.0 * delta)

    R = 2.0 * r12sq * (1.0 - np.cos(2.0 * delta)) / denom
    T = (1.0 - r12sq) ** 2 / denom

    return FPResult(R=R, T=T)