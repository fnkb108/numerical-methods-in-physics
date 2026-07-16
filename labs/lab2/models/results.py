from dataclasses import dataclass
import numpy as np


@dataclass
class RTResult:
    freqs: np.ndarray
    wavelengths: np.ndarray
    R: np.ndarray
    T: np.ndarray


@dataclass
class Snapshot:
    step: int
    Ey: np.ndarray


@dataclass
class PMLResult:
    freqs: np.ndarray
    R: np.ndarray
    R_max: float


@dataclass
class FPResult:
    R: float
    T: float


@dataclass
class SlabResult:
    lam_nm: np.ndarray
    R: np.ndarray
    T: np.ndarray
    R_th: np.ndarray
    T_th: np.ndarray
    L_nm: float
    steps: int


@dataclass
class PhCResult:
    lam_nm: np.ndarray
    R: np.ndarray
    T: np.ndarray
    periods: int
    label: str


@dataclass
class CavityResult:
    lam_nm: np.ndarray
    R: np.ndarray
    T: np.ndarray
    RTsum: np.ndarray
    label: str