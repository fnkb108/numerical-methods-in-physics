import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

L = 2
q = 1
theta = 1
kappa = 1

nx = 5
nz = 10

hx = 1 / nx
hz = L / nz
r = np.array([hx*i for i in range(nx+1)])
z = np.array([hz*i for i in range(nz+1)])

p = np.array([[r[i], z[j]] for j in range(nz + 1) for i in range(nx + 1)]).T
N = p.shape[1]

idx = lambda i, j: j * (nx + 1) + i

triangles = []
for j in range(nz):
    for i in range(nx):
        n0 = idx(i, j)
        n1 = idx(i + 1, j)
        n2 = idx(i, j + 1)
        n3 = idx(i + 1, j + 1)
        triangles.append([n0, n1, n2])
        triangles.append([n1, n3, n2])

t_elem = np.array(triangles)

nb_tri = [[] for _ in range(N)]
for element, tri in enumerate(t_elem):
    for v in tri:
        nb_tri[v].append(element)

nb_count = np.array([len(lst) for lst in nb_tri], dtype=int)
nb_data = -np.ones((N, nb_count.max()), dtype=int)
for i, lst in enumerate(nb_tri):
    nb_data[i, :len(lst)] = lst

is_wall = np.zeros(N, dtype=int)

rr = p[0, :]
zz = p[1, :]

tol = 1e-12
r_min, r_max = rr.min(), rr.max()
z_min, z_max = zz.min(), zz.max()

for k in range(N):
    if abs(rr[k] - r_min) < tol:
        is_wall[k] = 1  # r=0, dT/dr=0
    if abs(zz[k] - z_min) < tol:
        is_wall[k] = 1  # z=0, dT/dz=-q
    if abs(rr[k] - r_max) < tol:
        is_wall[k] = 2  # r=1, T=0
    if abs(zz[k] - z_max) < tol:
        is_wall[k] = 3  # z=L, T=theta


def neumann_flux(node_id):
    if abs(rr[node_id] - r_min) < tol:
        return 0
    if abs(zz[node_id] - z_min) < tol:
        return q
    return 0


A = np.zeros((N, N))
b = np.zeros(N)

for i in range(N):

    if is_wall[i] == 2:
        A[i, i] = 1
        b[i] = 0
        continue

    if is_wall[i] == 3:
        A[i, i] = 1
        b[i] = theta
        continue

    for j in range(nb_count[i]):
        nel = nb_data[i, j]
        if nel < 0:
            continue

        n0, n1, n2 = t_elem[nel]

        if i == n1:
            n0, n1, n2 = n1, n2, n0
        elif i == n2:
            n0, n1, n2 = n2, n0, n1

        x0, y0 = p[0, n0], p[1, n0]
        x1, y1 = p[0, n1], p[1, n1]
        x2, y2 = p[0, n2], p[1, n2]

        x10 = x1 - x0
        y01 = y0 - y1
        x21 = x2 - x1
        y12 = y1 - y2
        x02 = x0 - x2
        y20 = y2 - y0

        Delta = x10 * y20 - x02 * y01
        area = 0.5 * abs(Delta)

        r_c = (x0 + x1 + x2) / 3
        coef = 0.5 * kappa * r_c / Delta

        a00 = coef * (y12 * y12 + x21 * x21)
        a01 = coef * (y12 * y20 + x21 * x02)
        a02 = coef * (y12 * y01 + x21 * x10)

        A[i, i] += a00
        A[i, n1] += a01
        A[i, n2] += a02

        if is_wall[n0] == 1:
            qn0 = neumann_flux(n0)

            if is_wall[n1] == 1:
                edge_len = np.hypot(x1 - x0, y1 - y0)
                r_edge = 0.5 * (x0 + x1)
                b[i] += qn0 * r_edge * edge_len / 2.0

            if is_wall[n2] == 1:
                edge_len = np.hypot(x2 - x0, y2 - y0)
                r_edge = 0.5 * (x0 + x2)
                b[i] += qn0 * r_edge * edge_len / 2.0

T = np.linalg.solve(A, b)

triang = mtri.Triangulation(p[0, :], p[1, :], t_elem)

fig, ax = plt.subplots(figsize=(6, 8))
ax.set_aspect('equal')
tpc = ax.tripcolor(triang, T, cmap='inferno')
ax.set_xlabel('r')
ax.set_ylabel('z')
ax.set_title(f'T(r, z), θ={theta}, q={q}')
plt.colorbar(tpc, ax=ax, label='T')
plt.tight_layout()
plt.show()