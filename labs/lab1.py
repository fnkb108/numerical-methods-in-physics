import numpy as np
import matplotlib.pyplot as plt

L = 2
q = 1
theta = 1

nx = 10
nz = 20

hx = 2 / (2 * nx - 1)
hz = L * 2 / (2 * nz - 1)

r = np.array([(-0.5 + i) * hx for i in range(nx + 1)])
z = np.array([(-0.5 + i) * hz for i in range(nz + 1)])

ni = nx - 1
nj = nz - 1
N = ni * nj

idx = lambda i, j: i * nj + j

A = np.zeros((N, N))
b = np.zeros(N)

for i_ in range(ni):
    i = i_ + 1

    ri = r[i]
    ri_plus = ri + hx / 2
    ri_minus = ri - hx / 2

    for j_ in range(nj):
        j = j_ + 1
        k = idx(i_, j_)
        cc = 0

        c_p = ri_plus / (ri * hx * hx)
        c_m = ri_minus / (ri * hx * hx)
        cc -= (c_p + c_m)

        if i + 1 <= nx - 1:
            A[k, idx(i_ + 1, j_)] += c_p
        elif i + 1 == nx:
            pass

        if i - 1 >= 1:
            A[k, idx(i_ - 1, j_)] += c_m
        elif i - 1 == 0:
            cc += c_m

        c_z = 1 / (hz * hz)
        cc -= 2 * c_z

        if j + 1 <= nz - 1:
            A[k, idx(i_, j_ + 1)] += c_z
        elif j + 1 == nz:
            b[k] -= c_z * theta

        if j - 1 >= 1:
            A[k, idx(i_, j_ - 1)] += c_z
        elif j - 1 == 0:
            cc += c_z
            b[k] -= c_z * hz * q

        A[k, k] += cc

T_vec = np.linalg.solve(A, b)
T_full = np.zeros((nx, nz))


for i_ in range(ni):
    for j_ in range(nj):
        T_full[i_ + 1, j_ + 1] = T_vec[idx(i_, j_)]

T_full[0, :] = T_full[1, :]
T_full[:, 0] = T_full[:, 1] + hz * q

r_vis = np.maximum(r, 0)
z_vis = np.maximum(z, 0)
Rm, Zm = np.meshgrid(r_vis, z_vis, indexing='ij')

fig, ax = plt.subplots(figsize=(6, 8))
cax = ax.imshow(T_full.T, origin='lower', aspect = 'auto',
                extent=[r_vis[0], r_vis[-1], z_vis[0], z_vis[-1]],
                cmap='inferno')
fig.colorbar(cax, ax=ax, label='T')
ax.set_xlabel('r')
ax.set_ylabel('z')
ax.set_title(f'T(r, z), θ={theta}, q={q}')
plt.tight_layout()
plt.savefig('lab1.png', dpi=150)
plt.show()
