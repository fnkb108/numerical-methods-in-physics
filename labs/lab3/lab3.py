import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import os

class BoundaryType:
    DIRICHLET = 1
    OUTLET = 2

class ConvectionScheme:
    UPWIND = 1
    TVD = 2

class TVDLimiterType:
    VAN_ALBADA = 1
    MINMOD = 2
    SUPERBEE = 3
    QUICK = 4

class LinearSolverType:
    SPARSE_LU = 1
    BICGSTAB = 2

@dataclass
class BoundaryCondition:
    type: int = BoundaryType.DIRICHLET
    value: object = field(default_factory=lambda: (lambda _: 0.0))

    @staticmethod
    def dirichlet(c):
        if callable(c):
            return BoundaryCondition(BoundaryType.DIRICHLET, c)
        return BoundaryCondition(BoundaryType.DIRICHLET, lambda _: float(c))

    @staticmethod
    def outlet():
        return BoundaryCondition(BoundaryType.OUTLET, lambda _: 0.0)

class Mesh:
    def __init__(self, Nx, Ny, Lx, Ly):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / Nx
        self.dy = Ly / Ny
    def cell_count(self): return self.Nx * self.Ny
    def id(self, i, j): return j * self.Nx + i
    def inside(self, i, j): return 0 <= i < self.Nx and 0 <= j < self.Ny
    def has_west(self, i): return i > 0
    def has_east(self, i): return i < self.Nx - 1
    def has_south(self, j): return j > 0
    def has_north(self, j): return j < self.Ny - 1
    def xc(self, i): return (i + 0.5) * self.dx
    def yc(self, j): return (j + 0.5) * self.dy

class ScalarField:
    def __init__(self, mesh, v=0.0):
        self.mesh = mesh
        self.data = np.full(mesh.cell_count(), v, dtype=float)
    def __call__(self, i, j): return self.data[self.mesh.id(i, j)]
    def set(self, i, j, v): self.data[self.mesh.id(i, j)] = v
    def copy(self):
        sf = ScalarField(self.mesh)
        sf.data = self.data.copy()
        return sf
    def max_abs_diff(self, other): return float(np.max(np.abs(self.data - other.data)))
    def to_2d(self): return self.data.reshape(self.mesh.Ny, self.mesh.Nx)

class LinearSystem:
    def __init__(self, mesh):
        self.mesh = mesh
        n = mesh.cell_count()
        self.aP = np.zeros(n)
        self.aW = np.zeros(n)
        self.aE = np.zeros(n)
        self.aS = np.zeros(n)
        self.aN = np.zeros(n)
        self.b  = np.zeros(n)

class TVDLimiter:
    @staticmethod
    def eval(t, r, beta=1.5):
        if not np.isfinite(r): return 0.0
        if t == TVDLimiterType.VAN_ALBADA:
            return 0.0 if r <= 0 else (r*r + r) / (r*r + 1)
        if t == TVDLimiterType.MINMOD:
            return max(0.0, min(1.0, r))
        if t == TVDLimiterType.SUPERBEE:
            return max(0.0, max(min(2*r, 1.0), min(r, 2.0)))
        if t == TVDLimiterType.QUICK:
            return max(0.0, min(2*r, (3+r)/4, 2.0))
        return 0.0

EPS = 1e-14

@dataclass
class ConvectionDiffusionProblem:
    mesh: object
    rho: float = 1.0
    u: float = 0.0
    v: float = 0.0
    Gamma: float = 1.0
    Su: float = 0.0
    Sp: float = 0.0
    left:   object = field(default_factory=lambda: BoundaryCondition.dirichlet(0.0))
    right:  object = field(default_factory=BoundaryCondition.outlet)
    bottom: object = field(default_factory=lambda: BoundaryCondition.dirichlet(0.0))
    top:    object = field(default_factory=BoundaryCondition.outlet)
    scheme: int = ConvectionScheme.TVD
    limiter: int = TVDLimiterType.VAN_ALBADA
    sweby_beta: float = 1.5

    def validate(self):
        if self.rho <= 0:  raise ValueError("rho>0")
        if self.Gamma < 0: raise ValueError("Gamma>=0")

    def Fw(self): return self.rho * self.u * self.mesh.dy
    def Fe(self): return self.rho * self.u * self.mesh.dy
    def Fs(self): return self.rho * self.v * self.mesh.dx
    def Fn(self): return self.rho * self.v * self.mesh.dx

    def Dw(self): return self.Gamma * self.mesh.dy / self.mesh.dx
    def De(self): return self.Gamma * self.mesh.dy / self.mesh.dx
    def Ds(self): return self.Gamma * self.mesh.dx / self.mesh.dy
    def Dn(self): return self.Gamma * self.mesh.dx / self.mesh.dy

    def Dwb(self): return self.Gamma * self.mesh.dy / (0.5 * self.mesh.dx)
    def Deb(self): return self.Gamma * self.mesh.dy / (0.5 * self.mesh.dx)
    def Dsb(self): return self.Gamma * self.mesh.dx / (0.5 * self.mesh.dy)
    def Dnb(self): return self.Gamma * self.mesh.dx / (0.5 * self.mesh.dy)
    def cell_volume(self): return self.mesh.dx * self.mesh.dy

    def Pex(self):
        if abs(self.Gamma) < EPS: return np.inf
        return self.rho * self.u * self.mesh.dx / self.Gamma

    def Pey(self):
        if abs(self.Gamma) < EPS: return np.inf
        return self.rho * self.v * self.mesh.dy / self.Gamma

def ghost_west(pr, phi, j):
    bc  = pr.left.value(pr.mesh.yc(j))
    inn = phi(0, j)
    return (2*bc - inn) if pr.left.type == BoundaryType.DIRICHLET else inn

def ghost_east(pr, phi, j):
    i   = pr.mesh.Nx - 1
    bc  = pr.right.value(pr.mesh.yc(j))
    inn = phi(i, j)
    return (2*bc - inn) if pr.right.type == BoundaryType.DIRICHLET else inn

def ghost_south(pr, phi, i):
    bc  = pr.bottom.value(pr.mesh.xc(i))
    inn = phi(i, 0)
    return (2*bc - inn) if pr.bottom.type == BoundaryType.DIRICHLET else inn

def ghost_north(pr, phi, i):
    j   = pr.mesh.Ny - 1
    bc  = pr.top.value(pr.mesh.xc(i))
    inn = phi(i, j)
    return (2*bc - inn) if pr.top.type == BoundaryType.DIRICHLET else inn

def sample(pr, phi, i, j):
    m = pr.mesh
    if m.inside(i, j):                       return phi(i, j)
    if i == -1   and 0 <= j < m.Ny:          return ghost_west(pr, phi, j)
    if i == m.Nx and 0 <= j < m.Ny:          return ghost_east(pr, phi, j)
    if j == -1   and 0 <= i < m.Nx:          return ghost_south(pr, phi, i)
    if j == m.Ny and 0 <= i < m.Nx:          return ghost_north(pr, phi, i)
    raise RuntimeError("Only 1 ghost layer")

def psi(pr, r): return TVDLimiter.eval(pr.limiter, r, pr.sweby_beta)
def ratio(n, d): return 0.0 if abs(d) < EPS else n / d

def east_corr(pr, phi, i, j):
    if not pr.mesh.has_east(i) or abs(pr.Fe()) < EPS: return 0.0
    if pr.Fe() > 0:
        C = phi(i, j);   D = phi(i+1, j); U = sample(pr, phi, i-1, j)
    else:
        C = phi(i+1, j); D = phi(i, j);   U = sample(pr, phi, i+2, j)
    d = D - C
    return 0.5 * psi(pr, ratio(C - U, d)) * d

def west_corr(pr, phi, i, j):
    if not pr.mesh.has_west(i) or abs(pr.Fw()) < EPS: return 0.0
    if pr.Fw() > 0:
        C = phi(i-1, j); D = phi(i, j);   U = sample(pr, phi, i-2, j)
    else:
        C = phi(i, j);   D = phi(i-1, j); U = sample(pr, phi, i+1, j)
    d = D - C
    return 0.5 * psi(pr, ratio(C - U, d)) * d

def north_corr(pr, phi, i, j):
    if not pr.mesh.has_north(j) or abs(pr.Fn()) < EPS: return 0.0
    if pr.Fn() > 0:
        C = phi(i, j);   D = phi(i, j+1); U = sample(pr, phi, i, j-1)
    else:
        C = phi(i, j+1); D = phi(i, j);   U = sample(pr, phi, i, j+2)
    d = D - C
    return 0.5 * psi(pr, ratio(C - U, d)) * d

def south_corr(pr, phi, i, j):
    if not pr.mesh.has_south(j) or abs(pr.Fs()) < EPS: return 0.0
    if pr.Fs() > 0:
        C = phi(i, j-1); D = phi(i, j);   U = sample(pr, phi, i, j-2)
    else:
        C = phi(i, j);   D = phi(i, j-1); U = sample(pr, phi, i, j+1)
    d = D - C
    return 0.5 * psi(pr, ratio(C - U, d)) * d

def build_system(pr, phi_lag):
    pr.validate()
    m   = pr.mesh
    sys = LinearSystem(m)
    Fw = pr.Fw(); Fe = pr.Fe(); Fs = pr.Fs(); Fn = pr.Fn()
    Dw = pr.Dw(); De = pr.De(); Ds = pr.Ds(); Dn = pr.Dn()
    V  = pr.cell_volume()
    for j in range(m.Ny):
        for i in range(m.Nx):
            p = m.id(i, j)
            aW = aE = aS = aN = 0.0
            rhs = pr.Su * V
            if m.has_west(i):
                aW = Dw + max(Fw, 0)
            else:
                y = m.yc(j)
                if pr.left.type == BoundaryType.DIRICHLET:
                    pb = pr.left.value(y)
                    sys.aP[p] += pr.Dwb()
                    rhs       += pr.Dwb() * pb
                    if Fw > 0:
                        sys.aP[p] += Fw
                        rhs       += Fw * pb
            if m.has_east(i):
                aE = De + max(-Fe, 0)
            else:
                y = m.yc(j)
                if pr.right.type == BoundaryType.DIRICHLET:
                    pb = pr.right.value(y)
                    sys.aP[p] += pr.Deb()
                    rhs       += pr.Deb() * pb
                    if Fe < 0:
                        sys.aP[p] += (-Fe)
                        rhs       += (-Fe) * pb
            if m.has_south(j):
                aS = Ds + max(Fs, 0)
            else:
                x = m.xc(i)
                if pr.bottom.type == BoundaryType.DIRICHLET:
                    pb = pr.bottom.value(x)
                    sys.aP[p] += pr.Dsb()
                    rhs       += pr.Dsb() * pb
                    if Fs > 0:
                        sys.aP[p] += Fs
                        rhs       += Fs * pb
            if m.has_north(j):
                aN = Dn + max(-Fn, 0)
            else:
                x = m.xc(i)
                if pr.top.type == BoundaryType.DIRICHLET:
                    pb = pr.top.value(x)
                    sys.aP[p] += pr.Dnb()
                    rhs       += pr.Dnb() * pb
                    if Fn < 0:
                        sys.aP[p] += (-Fn)
                        rhs       += (-Fn) * pb

            if pr.scheme == ConvectionScheme.TVD:
                rhs -= Fe * east_corr(pr, phi_lag, i, j)
                rhs += Fw * west_corr(pr, phi_lag, i, j)
                rhs -= Fn * north_corr(pr, phi_lag, i, j)
                rhs += Fs * south_corr(pr, phi_lag, i, j)

            sys.aP[p] += aW + aE + aS + aN + (Fe - Fw) + (Fn - Fs) - pr.Sp * V
            sys.aW[p]  = aW
            sys.aE[p]  = aE
            sys.aS[p]  = aS
            sys.aN[p]  = aN
            sys.b[p]   = rhs
    return sys

@dataclass
class SolverResult:
    iterations:  int   = 0
    residual:    float = 0.0
    max_delta:   float = 0.0
    converged:   bool  = False
    solver_name: str   = ""

def linear_solve(lsys, phi, solver_type=LinearSolverType.SPARSE_LU, tol=1e-12, maxiter=5000):
    m = lsys.mesh
    n = m.cell_count()
    res = SolverResult()
    rows, cols, vals = [], [], []
    for j in range(m.Ny):
        for i in range(m.Nx):
            p = m.id(i, j)
            if abs(lsys.aP[p]) < 1e-14:
                raise RuntimeError(f"Zero diag ({i},{j})")
            rows.append(p); cols.append(p);          vals.append( lsys.aP[p])
            if m.has_west(i)  and abs(lsys.aW[p]) > 1e-14:
                rows.append(p); cols.append(m.id(i-1,j)); vals.append(-lsys.aW[p])
            if m.has_east(i)  and abs(lsys.aE[p]) > 1e-14:
                rows.append(p); cols.append(m.id(i+1,j)); vals.append(-lsys.aE[p])
            if m.has_south(j) and abs(lsys.aS[p]) > 1e-14:
                rows.append(p); cols.append(m.id(i,j-1)); vals.append(-lsys.aS[p])
            if m.has_north(j) and abs(lsys.aN[p]) > 1e-14:
                rows.append(p); cols.append(m.id(i,j+1)); vals.append(-lsys.aN[p])
    A  = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    b  = lsys.b.copy()
    x0 = phi.data.copy()
    if solver_type == LinearSolverType.SPARSE_LU:
        lu = spla.splu(A.tocsc())
        x  = lu.solve(b)
        res.converged = True; res.solver_name = "SparseLU"; res.iterations = 1
    else:
        x, info = spla.bicgstab(A, b, x0=x0, maxiter=maxiter, rtol=tol)
        res.converged = (info == 0); res.solver_name = "BiCGSTAB"; res.iterations = maxiter
    bn = max(np.linalg.norm(b), 1e-14)
    res.residual  = float(np.linalg.norm(A @ x - b) / bn)
    res.max_delta = float(np.max(np.abs(x - x0)))
    phi.data[:] = x
    return res

@dataclass
class NonlinearSolveResult:
    outer_iterations: int   = 0
    max_delta:        float = 0.0
    converged:        bool  = False
    linear_result:    object = field(default_factory=SolverResult)
    history:          list  = field(default_factory=list)

def outer_solve(pr, phi, max_outer=1000, outer_tol=1e-3):
    res = NonlinearSolveResult()
    if pr.scheme == ConvectionScheme.UPWIND:
        old = phi.copy()
        lsys = build_system(pr, old)
        res.linear_result  = linear_solve(lsys, phi)
        res.max_delta      = phi.max_abs_diff(old)
        res.history.append(res.max_delta)
        res.outer_iterations = 1
        res.converged = res.linear_result.converged
        return res
    for it in range(1, max_outer + 1):
        lag  = phi.copy()
        lsys = build_system(pr, lag)
        res.linear_result  = linear_solve(lsys, phi)
        res.max_delta      = phi.max_abs_diff(lag)
        res.history.append(res.max_delta)
        res.outer_iterations = it
        res.converged = res.linear_result.converged and res.max_delta <= outer_tol
        if res.converged:
            return res
    return res

def sample_bilinear(phi, x, y):
    m  = phi.mesh
    x  = np.clip(x, m.xc(0), m.xc(m.Nx-1))
    y  = np.clip(y, m.yc(0), m.yc(m.Ny-1))
    i0 = int(np.clip(int((x - m.xc(0)) / m.dx), 0, m.Nx-2))
    j0 = int(np.clip(int((y - m.yc(0)) / m.dy), 0, m.Ny-2))
    i1 = i0+1; j1 = j0+1
    x0 = m.xc(i0); x1 = m.xc(i1)
    y0 = m.yc(j0); y1 = m.yc(j1)
    tx = 0.0 if abs(x1-x0) < 1e-14 else (x-x0)/(x1-x0)
    ty = 0.0 if abs(y1-y0) < 1e-14 else (y-y0)/(y1-y0)
    return (1-ty)*((1-tx)*phi(i0,j0)+tx*phi(i1,j0)) + ty*((1-tx)*phi(i0,j1)+tx*phi(i1,j1))

def sample_segment(phi, x0, y0, x1, y1, pts=401):
    t  = np.linspace(0, 1, pts)
    xs = (1-t)*x0 + t*x1
    ys = (1-t)*y0 + t*y1
    return t, np.array([sample_bilinear(phi, xi, yi) for xi, yi in zip(xs, ys)])

def lim_name(lim):
    return {TVDLimiterType.VAN_ALBADA: "VanAlbada",
            TVDLimiterType.MINMOD:     "MinMod",
            TVDLimiterType.SUPERBEE:   "Superbee",
            TVDLimiterType.QUICK:      "QUICK"}.get(lim, str(lim))

def make_pr(mesh, scheme, lim, gamma, left_bc, bottom_bc):
    return ConvectionDiffusionProblem(
        mesh=mesh, rho=1.0, u=2.0, v=2.0, Gamma=gamma,
        left=left_bc, bottom=bottom_bc,
        right=BoundaryCondition.outlet(), top=BoundaryCondition.outlet(),
        scheme=scheme, limiter=lim)

def run_pr(pr, v0=0.0):
    phi    = ScalarField(pr.mesh, v0)
    result = outer_solve(pr, phi)
    pex    = pr.Pex()
    pey    = pr.Pey()
    sn     = "Upwind" if pr.scheme == ConvectionScheme.UPWIND else "TVD-" + lim_name(pr.limiter)
    print(f"  {sn:<20} G={pr.Gamma:.0e}  Pex={pex:.2f}  Pey={pey:.2f}  "
          f"it={result.outer_iterations}  d={result.max_delta:.1e}  "
          f"r={result.linear_result.residual:.1e}  "
          f"{'OK' if result.converged else 'FAIL'}")
    return phi, result

def plot_heatmap(phi, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    m  = phi.mesh
    xs = np.array([m.xc(i) for i in range(m.Nx)])
    ys = np.array([m.yc(j) for j in range(m.Ny)])
    im = ax.contourf(xs, ys, phi.to_2d(), levels=50, cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()

def plot_curves(x, ys, labels, title, xl, yl, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for y, lb in zip(ys, labels):
        ax.plot(x, y, lw=2, label=lb)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
    ax.legend(); ax.grid(True)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()

def task1(Nx=80):
    os.makedirs("output", exist_ok=True)
    print("\n=== Task 1 ===")
    mesh = Mesh(Nx, Nx, 1.0, 1.0)
    lbc  = BoundaryCondition.dirichlet(1.0)
    bbc  = BoundaryCondition.dirichlet(0.0)
    X0, Y0, X1, Y1 = 0.0, 1.0, 1.0, 0.0
    N = 401
    def ex1(x, y): return 1.0 if y > x else (0.0 if y < x else 0.5)
    t_ = np.linspace(0, 1, N)
    exact = np.array([ex1((1-t)*X0+t*X1, (1-t)*Y0+t*Y1) for t in t_])

    lims = [TVDLimiterType.VAN_ALBADA, TVDLimiterType.MINMOD,
            TVDLimiterType.SUPERBEE,   TVDLimiterType.QUICK]

    phi_up, _ = run_pr(make_pr(mesh, ConvectionScheme.UPWIND, TVDLimiterType.VAN_ALBADA, 0, lbc, bbc))
    t_seg, c_up = sample_segment(phi_up, X0, Y0, X1, Y1, N)
    curves = [exact, c_up]; labels = ["Exact", "Upwind"]; phi_vl = None
    for lim in lims:
        phi_t, _ = run_pr(make_pr(mesh, ConvectionScheme.TVD, lim, 0, lbc, bbc))
        _, c = sample_segment(phi_t, X0, Y0, X1, Y1, N)
        curves.append(c); labels.append(lim_name(lim))
        if lim == TVDLimiterType.VAN_ALBADA: phi_vl = phi_t
    plot_curves(t_seg, curves, labels, "Task 1: схемы, Gamma=0", "t", "phi", "output/task1_comparison.png")
    plot_heatmap(phi_vl, "Task 1: phi(x,y) TVD-VanAlbada Gamma=0", "output/task1_heatmap_G0.png")

    phi_g1, _ = run_pr(make_pr(mesh, ConvectionScheme.TVD, TVDLimiterType.VAN_ALBADA, 1e-1, lbc, bbc))
    phi_g2, _ = run_pr(make_pr(mesh, ConvectionScheme.TVD, TVDLimiterType.VAN_ALBADA, 1e-2, lbc, bbc))
    phi_g3, _ = run_pr(make_pr(mesh, ConvectionScheme.TVD, TVDLimiterType.VAN_ALBADA, 1e-3, lbc, bbc))
    phi_g4, _ = run_pr(make_pr(mesh, ConvectionScheme.TVD, TVDLimiterType.VAN_ALBADA, 1e-4, lbc, bbc))
    _, c0 = sample_segment(phi_vl, X0, Y0, X1, Y1, N)
    _, c1 = sample_segment(phi_g1, X0, Y0, X1, Y1, N)
    _, c2 = sample_segment(phi_g2, X0, Y0, X1, Y1, N)
    _, c3 = sample_segment(phi_g3, X0, Y0, X1, Y1, N)
    _, c4 = sample_segment(phi_g4, X0, Y0, X1, Y1, N)
    plot_curves(t_seg, [c0,c1,c2,c3,c4], ["G=0","G=1e-1","G=1e-2","G=1e-3","G=1e-4"],
                "Task 1: влияние диффузии", "t", "phi", "output/task1_diffusion.png")
    plot_heatmap(phi_g1, "Task 1: phi Gamma=1e-1", "output/task1_heatmap_G1e-1.png")
    plot_heatmap(phi_g2, "Task 1: phi Gamma=1e-2", "output/task1_heatmap_G1e-2.png")
    plot_heatmap(phi_g3, "Task 1: phi Gamma=1e-3", "output/task1_heatmap_G1e-3.png")
    plot_heatmap(phi_g4, "Task 1: phi Gamma=1e-4", "output/task1_heatmap_G1e-4.png")

def task2(Nx=80):
    os.makedirs("output", exist_ok=True)
    print("\n=== Task 2 ===")
    mesh = Mesh(Nx, Nx, 1.0, 1.0)
    lbc  = BoundaryCondition.dirichlet(lambda y: 1.0 if y <= 0.5 else 0.0)
    bbc  = BoundaryCondition.dirichlet(0.0)
    X0, Y0, X1, Y1 = 0.0, 1.0, 1.0, 0.0
    N = 401
    def ex2(x, y): return 0.0 if y <= x else (1.0 if (y - x) <= 0.5 else 0.0)
    t_ = np.linspace(0, 1, N)
    exact = np.array([ex2((1-t)*X0+t*X1, (1-t)*Y0+t*Y1) for t in t_])

    lims = [TVDLimiterType.VAN_ALBADA, TVDLimiterType.MINMOD,
            TVDLimiterType.SUPERBEE,   TVDLimiterType.QUICK]

    phi_up, _ = run_pr(make_pr(mesh, ConvectionScheme.UPWIND, TVDLimiterType.VAN_ALBADA, 0, lbc, bbc))
    t_seg, c_up = sample_segment(phi_up, X0, Y0, X1, Y1, N)
    curves = [exact, c_up]; labels = ["Exact", "Upwind"]; phi_vl = None
    for lim in lims:
        phi_t, _ = run_pr(make_pr(mesh, ConvectionScheme.TVD, lim, 0, lbc, bbc))
        _, c = sample_segment(phi_t, X0, Y0, X1, Y1, N)
        curves.append(c); labels.append(lim_name(lim))
        if lim == TVDLimiterType.VAN_ALBADA: phi_vl = phi_t
    plot_curves(t_seg, curves, labels, "Task 2: схемы, Gamma=0", "t", "phi", "output/task2_comparison.png")
    plot_heatmap(phi_vl, "Task 2: phi(x,y) TVD-VanAlbada Gamma=0", "output/task2_heatmap_G0.png")

    phi_g1, _ = run_pr(make_pr(mesh, ConvectionScheme.TVD, TVDLimiterType.VAN_ALBADA, 1e-1, lbc, bbc))
    phi_g2, _ = run_pr(make_pr(mesh, ConvectionScheme.TVD, TVDLimiterType.VAN_ALBADA, 1e-2, lbc, bbc))
    phi_g3, _ = run_pr(make_pr(mesh, ConvectionScheme.TVD, TVDLimiterType.VAN_ALBADA, 1e-3, lbc, bbc))
    phi_g4, _ = run_pr(make_pr(mesh, ConvectionScheme.TVD, TVDLimiterType.VAN_ALBADA, 1e-4, lbc, bbc))
    _, c0 = sample_segment(phi_vl, X0, Y0, X1, Y1, N)
    _, c1 = sample_segment(phi_g1, X0, Y0, X1, Y1, N)
    _, c2 = sample_segment(phi_g2, X0, Y0, X1, Y1, N)
    _, c3 = sample_segment(phi_g3, X0, Y0, X1, Y1, N)
    _, c4 = sample_segment(phi_g4, X0, Y0, X1, Y1, N)
    plot_curves(t_seg, [c0,c1,c2,c3,c4], ["G=0","G=1e-1","G=1e-2","G=1e-3","G=1e-4"],
                "Task 2: влияние диффузии", "t", "phi", "output/task2_diffusion.png")
    plot_heatmap(phi_g1, "Task 2: phi Gamma=1e-1", "output/task2_heatmap_G1e-1.png")
    plot_heatmap(phi_g2, "Task 2: phi Gamma=1e-2", "output/task2_heatmap_G1e-2.png")
    plot_heatmap(phi_g3, "Task 2: phi Gamma=1e-3", "output/task2_heatmap_G1e-3.png")
    plot_heatmap(phi_g4, "Task 2: phi Gamma=1e-4", "output/task2_heatmap_G1e-4.png")

if __name__ == "__main__":
    task1(Nx=80)
    task2(Nx=80)
    print("\nAll tasks complete.")