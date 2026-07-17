import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class FlowParameters:
    rho: float = 1.0
    mu: float = 0.01
    h: float = 0.5
    dpdx: float = -0.04
    L: float = 20.0

    @property
    def nu(self) -> float:
        return self.mu / self.rho

    def u_analytical(self, y: np.ndarray) -> np.ndarray:
        return (self.h ** 2 * (-self.dpdx)) / (2.0 * self.mu) * (1.0 - (y / self.h) ** 2)

    @property
    def u_max(self) -> float:
        return (self.h ** 2 * (-self.dpdx)) / (2.0 * self.mu)

    @property
    def u_mean(self) -> float:
        return (2.0 / 3.0) * self.u_max

    @property
    def Re(self) -> float:
        return self.u_mean * (2.0 * self.h) / self.nu


class StaggeredGrid:

    def __init__(self, NI: int, NJ: int, Lx: float, Ly: float):
        self.NI = NI
        self.NJ = NJ
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / NI
        self.dy = Ly / NJ

        self.xP = np.array([(I - 0.5) * self.dx for I in range(NI + 2)])
        self.yP = np.array([(J - 0.5) * self.dy for J in range(NJ + 2)])

        self.y_phys = self.yP[1:NJ + 1] - Ly / 2.0

    def shape(self) -> Tuple[int, int]:
        return (self.NI + 2, self.NJ + 2)


class BoundaryConditions:

    def __init__(self, grid: StaggeredGrid, params: FlowParameters):
        self.grid = grid
        self.params = params

        NJ = grid.NJ
        yc = grid.yP[1:NJ + 1] - grid.Ly / 2.0

        self.u_inlet = params.u_analytical(yc)

    def apply_u_inlet(self, u: np.ndarray) -> None:
        u[1, 1:self.grid.NJ + 1] = self.u_inlet

    def apply_u_outlet(self, u: np.ndarray) -> None:
        NI = self.grid.NI
        u[NI + 1, 1:self.grid.NJ + 1] = u[NI, 1:self.grid.NJ + 1]

    def apply_v_walls(self, v: np.ndarray) -> None:
        NJ = self.grid.NJ
        v[:, 1] = 0.0
        v[:, NJ + 1] = 0.0

    def apply_all(self, u: np.ndarray, v: np.ndarray) -> None:
        self.apply_u_inlet(u)
        self.apply_u_outlet(u)
        self.apply_v_walls(v)


class TDMA:

    @staticmethod
    def solve(aW: np.ndarray, aP: np.ndarray, aE: np.ndarray,
              b: np.ndarray) -> np.ndarray:
        n = len(aP)
        alpha = np.zeros(n)
        beta = np.zeros(n)
        phi = np.zeros(n)

        denom = aP[0] if abs(aP[0]) > 1e-300 else 1e-300
        alpha[0] = aE[0] / denom
        beta[0] = b[0] / denom

        for i in range(1, n):
            denom = aP[i] - aW[i] * alpha[i - 1]
            if abs(denom) < 1e-300:
                denom = 1e-300
            alpha[i] = aE[i] / denom
            beta[i] = (b[i] + aW[i] * beta[i - 1]) / denom

        phi[-1] = beta[-1]
        for i in range(n - 2, -1, -1):
            phi[i] = alpha[i] * phi[i + 1] + beta[i]

        return phi

    @staticmethod
    def sweep(aP: np.ndarray, aW: np.ndarray, aE: np.ndarray,
              aS: np.ndarray, aN: np.ndarray, b: np.ndarray,
              phi: np.ndarray,
              i1: int, i2: int, j1: int, j2: int,
              n_sweeps: int = 3) -> np.ndarray:

        phi = phi.copy()

        for _ in range(n_sweeps):

            for J in range(j1, j2 + 1):
                rhs = (b[i1:i2 + 1, J]
                       + aS[i1:i2 + 1, J] * phi[i1:i2 + 1, J - 1]
                       + aN[i1:i2 + 1, J] * phi[i1:i2 + 1, J + 1])
                phi[i1:i2 + 1, J] = TDMA.solve(
                    aW[i1:i2 + 1, J], aP[i1:i2 + 1, J], aE[i1:i2 + 1, J], rhs)

            for i in range(i1, i2 + 1):
                rhs = (b[i, j1:j2 + 1]
                       + aW[i, j1:j2 + 1] * phi[i - 1, j1:j2 + 1]
                       + aE[i, j1:j2 + 1] * phi[i + 1, j1:j2 + 1])
                phi[i, j1:j2 + 1] = TDMA.solve(
                    aS[i, j1:j2 + 1], aP[i, j1:j2 + 1], aN[i, j1:j2 + 1], rhs)

        return phi


class FVMDiscretization:

    def __init__(self, grid: StaggeredGrid, params: FlowParameters):
        self.g = grid
        self.p = params

    def build_u(self, u: np.ndarray, v: np.ndarray, p_field: np.ndarray,
                alpha_u: float) -> Tuple[np.ndarray, ...]:

        NI, NJ = self.g.NI, self.g.NJ
        dx, dy = self.g.dx, self.g.dy
        rho, mu = self.p.rho, self.p.mu
        Ax = dy
        Ay = dx
        sh = self.g.shape()

        aW = np.zeros(sh);
        aE = np.zeros(sh)
        aS = np.zeros(sh);
        aN = np.zeros(sh)
        b = np.zeros(sh);
        aP = np.zeros(sh);
        d = np.zeros(sh)

        for i in range(2, NI + 1):
            for J in range(1, NJ + 1):

                Fw = rho * 0.5 * (u[i, J] + u[i - 1, J]) * Ax
                Fe = rho * 0.5 * (u[i + 1, J] + u[i, J]) * Ax

                Fs = rho * 0.5 * (v[i, J] + v[i - 1, J]) * Ay
                Fn = rho * 0.5 * (v[i, J + 1] + v[i - 1, J + 1]) * Ay

                Dw = mu * Ax / dx
                De = mu * Ax / dx
                Ds = mu * Ay / dy
                Dn = mu * Ay / dy

                _aW = max(Fw, Dw + Fw / 2.0, 0.0)
                _aE = max(-Fe, De - Fe / 2.0, 0.0)
                _aS = max(Fs, Ds + Fs / 2.0, 0.0)
                _aN = max(-Fn, Dn - Fn / 2.0, 0.0)

                Sp = 0.0
                if J == 1:   Sp -= mu * Ax / (dy / 2.0)
                if J == NJ:  Sp -= mu * Ax / (dy / 2.0)

                _aP_nom = _aW + _aE + _aS + _aN + (Fe - Fw) - Sp

                _b = ((p_field[i - 1, J] - p_field[i, J]) * Ax
                      + (1.0 - alpha_u) / alpha_u * _aP_nom * u[i, J])

                _aP_rel = _aP_nom / alpha_u

                aW[i, J] = _aW;
                aE[i, J] = _aE
                aS[i, J] = _aS;
                aN[i, J] = _aN
                aP[i, J] = _aP_rel
                b[i, J] = _b

                d[i, J] = Ax / _aP_rel if _aP_rel > 1e-300 else 0.0

        for J in range(1, NJ + 1):
            aP[1, J] = 1.0
            b[1, J] = u[1, J]
            d[1, J] = 0.0

        return aP, aW, aE, aS, aN, b, d

    def build_v(self, u: np.ndarray, v: np.ndarray, p_field: np.ndarray,
                alpha_v: float) -> Tuple[np.ndarray, ...]:

        NI, NJ = self.g.NI, self.g.NJ
        dx, dy = self.g.dx, self.g.dy
        rho, mu = self.p.rho, self.p.mu
        Ax = dy
        Ay = dx
        sh = self.g.shape()

        aW = np.zeros(sh);
        aE = np.zeros(sh)
        aS = np.zeros(sh);
        aN = np.zeros(sh)
        b = np.zeros(sh);
        aP = np.zeros(sh);
        d = np.zeros(sh)

        for I in range(1, NI + 1):
            for j in range(2, NJ + 1):
                J = j

                Fw = rho * 0.5 * (u[I, J] + u[I, J - 1]) * Ax
                Fe = rho * 0.5 * (u[I + 1, J] + u[I + 1, J - 1]) * Ax
                Fs = rho * 0.5 * (v[I, j - 1] + v[I, j]) * Ay
                Fn = rho * 0.5 * (v[I, j] + v[I, j + 1]) * Ay

                Dw = mu * Ax / dx
                De = mu * Ax / dx
                Ds = mu * Ay / dy
                Dn = mu * Ay / dy

                _aW = max(Fw, Dw + Fw / 2.0, 0.0)
                _aE = max(-Fe, De - Fe / 2.0, 0.0)
                _aS = max(Fs, Ds + Fs / 2.0, 0.0)
                _aN = max(-Fn, Dn - Fn / 2.0, 0.0)

                _aP_nom = _aW + _aE + _aS + _aN + (Fe - Fw)

                _b = ((p_field[I, J - 1] - p_field[I, J]) * Ay
                      + (1.0 - alpha_v) / alpha_v * _aP_nom * v[I, j])

                _aP_rel = _aP_nom / alpha_v

                aW[I, j] = _aW;
                aE[I, j] = _aE
                aS[I, j] = _aS;
                aN[I, j] = _aN
                aP[I, j] = _aP_rel
                b[I, j] = _b
                d[I, j] = Ay / _aP_rel if _aP_rel > 1e-300 else 0.0

        return aP, aW, aE, aS, aN, b, d


class SIMPLESolver:

    def __init__(self, grid: StaggeredGrid, params: FlowParameters,
                 bc: BoundaryConditions,
                 alpha_u: float = 0.5,
                 alpha_v: float = 0.5,
                 alpha_p: float = 0.3):
        self.g = grid
        self.params = params
        self.bc = bc
        self.disc = FVMDiscretization(grid, params)
        self.alpha_u = alpha_u
        self.alpha_v = alpha_v
        self.alpha_p = alpha_p

        sh = grid.shape()
        self.u = np.zeros(sh)
        self.v = np.zeros(sh)
        self.pf = np.zeros(sh)

        self.u[1:, 1:grid.NJ + 1] = params.u_mean

        bc.apply_u_inlet(self.u)
        bc.apply_u_outlet(self.u)
        bc.apply_v_walls(self.v)

    def _pressure_correction(self,
                             u_star: np.ndarray,
                             v_star: np.ndarray,
                             d_u: np.ndarray,
                             d_v: np.ndarray) -> np.ndarray:

        NI, NJ = self.g.NI, self.g.NJ
        rho = self.params.rho
        Ax = self.g.dy
        Ay = self.g.dx
        sh = self.g.shape()

        aE = np.zeros(sh);
        aW = np.zeros(sh)
        aN = np.zeros(sh);
        aS = np.zeros(sh)
        b = np.zeros(sh);
        aP = np.zeros(sh)

        for I in range(1, NI + 1):
            for J in range(1, NJ + 1):

                _aE = rho * d_u[I + 1, J] * Ax
                _aW = rho * d_u[I, J] * Ax
                _aN = rho * d_v[I, J + 1] * Ay
                _aS = rho * d_v[I, J] * Ay

                if I == 1:
                    _aW = 0.0

                if I == NI:
                    _aE = 0.0

                _aP = _aE + _aW + _aN + _aS

                _b = (rho * (u_star[I, J] - u_star[I + 1, J]) * Ax
                      + rho * (v_star[I, J] - v_star[I, J + 1]) * Ay)

                aE[I, J] = _aE;
                aW[I, J] = _aW
                aN[I, J] = _aN;
                aS[I, J] = _aS
                aP[I, J] = _aP;
                b[I, J] = _b

        I_ref = NI
        J_ref = NJ // 2 + 1
        aP[I_ref, J_ref] += 1.0e30

        p_prime = TDMA.sweep(aP, aW, aE, aS, aN, b,
                             np.zeros(sh),
                             i1=1, i2=NI, j1=1, j2=NJ, n_sweeps=15)
        return p_prime

    def iterate(self, n_iter: int = 3000, tol: float = 1e-6,
                verbose: bool = True) -> list:

        NI, NJ = self.g.NI, self.g.NJ
        Ax = self.g.dy
        Ay = self.g.dx
        rho = self.params.rho
        residuals = []

        for it in range(n_iter):

            aP_u, aW_u, aE_u, aS_u, aN_u, b_u, d_u = self.disc.build_u(
                self.u, self.v, self.pf, self.alpha_u)
            u_star = TDMA.sweep(aP_u, aW_u, aE_u, aS_u, aN_u, b_u,
                                self.u.copy(),
                                i1=1, i2=NI, j1=1, j2=NJ, n_sweeps=5)

            self.bc.apply_u_inlet(u_star)
            self.bc.apply_u_outlet(u_star)

            aP_v, aW_v, aE_v, aS_v, aN_v, b_v, d_v = self.disc.build_v(
                u_star, self.v, self.pf, self.alpha_v)
            v_star = TDMA.sweep(aP_v, aW_v, aE_v, aS_v, aN_v, b_v,
                                self.v.copy(),
                                i1=1, i2=NI, j1=2, j2=NJ, n_sweeps=5)
            self.bc.apply_v_walls(v_star)

            p_prime = self._pressure_correction(u_star, v_star, d_u, d_v)

            for i in range(2, NI + 1):
                for J in range(1, NJ + 1):
                    self.u[i, J] = (u_star[i, J]
                                    + d_u[i, J] * (p_prime[i - 1, J] - p_prime[i, J]))
            self.bc.apply_u_inlet(self.u)
            self.bc.apply_u_outlet(self.u)

            for I in range(1, NI + 1):
                for j in range(2, NJ + 1):
                    self.v[I, j] = (v_star[I, j]
                                    + d_v[I, j] * (p_prime[I, j - 1] - p_prime[I, j]))
            self.bc.apply_v_walls(self.v)

            self.pf += self.alpha_p * p_prime

            res = 0.0
            for I in range(1, NI + 1):
                for J in range(1, NJ + 1):
                    div = (rho * (self.u[I + 1, J] - self.u[I, J]) * Ax
                           + rho * (self.v[I, J + 1] - self.v[I, J]) * Ay)
                    res += abs(div)
            residuals.append(res)

            if verbose and it % 200 == 0:
                print(f"  Iter {it:5d}:  residual = {res:.4e}")
            if res < tol:
                if verbose:
                    print(f"  Сошлось на итерации {it},  residual = {res:.4e}")
                break

        return residuals

    def u_profile_at(self, x_frac: float) -> Tuple[np.ndarray, np.ndarray]:

        NI, NJ = self.g.NI, self.g.NJ
        i_mid = max(1, min(NI, round(x_frac * NI)))
        y = self.g.y_phys
        uc = 0.5 * (self.u[i_mid, 1:NJ + 1] + self.u[i_mid + 1, 1:NJ + 1])
        return y, uc

    def pressure_along_axis(self) -> Tuple[np.ndarray, np.ndarray]:

        NI, NJ = self.g.NI, self.g.NJ
        J_mid = NJ // 2 + 1
        x = self.g.xP[1:NI + 1]
        return x, self.pf[1:NI + 1, J_mid]

    def y_plus(self) -> float:

        NI = self.g.NI
        yP = self.g.dy / 2.0
        u_P = np.mean(self.u[2:NI + 1, 1])
        tau_w = self.params.mu * u_P / yP
        u_tau = np.sqrt(abs(tau_w) / self.params.rho)
        return u_tau * yP / self.params.nu


class PoiseuilleProblem:

    def __init__(self,
                 NI: int = 40, NJ: int = 20,
                 rho: float = 1.0, mu: float = 0.01,
                 h: float = 0.5, dpdx: float = -0.04, L: float = 20.0,
                 alpha_u: float = 0.5, alpha_v: float = 0.5, alpha_p: float = 0.3):
        self.params = FlowParameters(rho=rho, mu=mu, h=h, dpdx=dpdx, L=L)
        self.grid = StaggeredGrid(NI, NJ, L, 2.0 * h)
        self.bc = BoundaryConditions(self.grid, self.params)
        self.solver = SIMPLESolver(self.grid, self.params, self.bc,
                                   alpha_u, alpha_v, alpha_p)

    def solve(self, n_iter: int = 3000, tol: float = 1e-6,
              verbose: bool = True) -> list:
        return self.solver.iterate(n_iter, tol, verbose)

    def print_params(self) -> None:
        pr = self.params
        g = self.grid
        print(f"  rho={pr.rho}, mu={pr.mu}, h={pr.h}, dpdx={pr.dpdx}, L={pr.L}")
        print(f"  u_max  = {pr.u_max:.4f}  (ф.2)")
        print(f"  u_mean = {pr.u_mean:.4f}  (ф.3)")
        print(f"  Re     = {pr.Re:.2f}      (ф.4)")
        print(f"  Сетка {g.NI}x{g.NJ}: dx={g.dx:.4f}, dy={g.dy:.4f}")
        y_wall = g.dy / 2.0
        print(f"  Аналит. y+ = {y_wall / g.dy:.4f}  (< 11.63 - лам. подслой)")

    def print_results(self) -> None:
        pr = self.params
        sv = self.solver
        NI = self.grid.NI
        u_num_max = np.max(sv.u[1:NI + 2, 1:self.grid.NJ + 1])
        err = abs(u_num_max - pr.u_max) / pr.u_max * 100
        print(f"  Re          = {pr.Re:.2f}")
        print(f"  u_max числ. = {u_num_max:.5f}")
        print(f"  u_max анал. = {pr.u_max:.5f}")
        print(f"  Ошибка      = {err:.2f} %")
        print(f"  y+          = {sv.y_plus():.4f}")


if __name__ == "__main__":
    prob = PoiseuilleProblem(
        NI=40, NJ=20,
        rho=1.0, mu=0.01, h=0.5, dpdx=-0.04, L=20.0,
        alpha_u=0.5, alpha_v=0.5, alpha_p=0.3
    )

    print("\nПараметры задачи:")
    prob.print_params()

    print(f"\n  SIMPLE (au=av={prob.solver.alpha_u}, ap={prob.solver.alpha_p})...")
    tolerance = 1e-9
    residuals = prob.solve(n_iter=5000, tol=tolerance, verbose=True)
    converged = (residuals[-1] < 1e-6)

    print(f"\n  Итераций: {len(residuals)}, сошлось: {converged}")
    print("\nРезультаты:")
    prob.print_results()

    y_an = np.linspace(-prob.params.h, prob.params.h, 300)
    u_an = prob.params.u_analytical(y_an)

    y01, u01 = prob.solver.u_profile_at(x_frac=0.1)
    y05, u05 = prob.solver.u_profile_at(x_frac=0.5)
    y09, u09 = prob.solver.u_profile_at(x_frac=0.9)

    x_p, p_axis = prob.solver.pressure_along_axis()
    p_an_line = p_axis[0] + prob.params.dpdx * (x_p - x_p[0])

    Re_str = f"Re={prob.params.Re:.0f}"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(u_an, y_an, 'r--', lw=2.0, label="Аналитика (ф.1)")
    ax.plot(u01, y01, 'C0-', lw=2.0, label="SIMPLE x/L=0.1")
    ax.plot(u05, y05, 'C1-', lw=2.0, label="SIMPLE x/L=0.5")
    ax.plot(u09, y09, 'C2-', lw=2.0, label="SIMPLE x/L=0.9")
    ax.axhline(-prob.params.h, color='k', lw=2.5)
    ax.axhline(prob.params.h, color='k', lw=2.5)
    ax.set_xlabel("u [м/с]", fontsize=12)
    ax.set_ylabel("y [м]", fontsize=12)
    ax.set_title("Профиль u(y)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9);
    ax.grid(True, alpha=0.4)

    ax = axes[1]
    ax.plot(x_p, p_axis, 'b-', lw=2.5, label="SIMPLE")
    ax.plot(x_p, p_an_line, 'r--', lw=2.0, label="Аналит. p(x)")
    ax.set_xlabel("x [м]", fontsize=12)
    ax.set_ylabel("p [Па]", fontsize=12)
    ax.set_title("Давление вдоль оси", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9);
    ax.grid(True, alpha=0.4)

    ax = axes[2]
    ax.semilogy(residuals, 'b-', lw=1.5)
    ax.axhline(tolerance, color='r', ls='--', label="tol=1e-6")
    ax.set_xlabel("Итерация", fontsize=12)
    ax.set_ylabel("sum|div u|", fontsize=12)
    ax.set_title("Сходимость SIMPLE", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9);
    ax.grid(True, which='both', alpha=0.4)

    NI = prob.grid.NI;
    NJ = prob.grid.NJ
    u_max_num = np.max(prob.solver.u[1:NI + 2, 1:NJ + 1])
    err_pct = abs(u_max_num - prob.params.u_max) / prob.params.u_max * 100

    plt.suptitle(
        f"Задание 1 — Пуазейль: {Re_str}, сетка {NI}×{NJ}, "
        f"ошибка u_max={err_pct:.2f}%",
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for y_num, u_num, label in [(y01, u01, "x/L=0.1"),
                                (y05, u05, "x/L=0.5"),
                                (y09, u09, "x/L=0.9")]:
        u_an_num = prob.params.u_analytical(y_num)
        ax2.semilogy(y_num, np.abs(u_num - u_an_num) + 1e-12, marker='o',
                     ms=4, lw=1.5, label=label)
    ax2.set_xlabel("y [М]", fontsize=12)
    ax2.set_ylabel("|u_num - u_an| [м/с]", fontsize=12)
    ax2.set_title("Абс. ошибка профиля скорости (log)", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10);
    ax2.grid(True, which='both', alpha=0.4)
    plt.tight_layout()
    plt.show()