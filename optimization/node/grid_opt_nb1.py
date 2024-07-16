import math as fm

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from experiments.optimization.node.grid_optimizer import fourth_order_error_kz, rational_approx_error, _OptTable, \
    get_optimal_grid
from propagators._utils import pade_propagator_coefs


k_z_grid = np.linspace(0, 2*fm.pi, 1000)
dz_grid = np.linspace(0.001, 10, 10000)
k_z_grid_m, dz_grid_m = np.meshgrid(k_z_grid, dz_grid)
t = fourth_order_error_kz(k_z_grid_m, dz_grid_m)

plt.imshow(np.log10(t), extent=[k_z_grid[0], k_z_grid[-1], dz_grid[-1], dz_grid[0]], norm=Normalize(-9, -1), aspect='auto')
plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()
plt.yscale("log")
plt.show()

t2 = fourth_order_error_kz(0.1, dz_grid)
plt.plot(dz_grid, t2)
plt.yscale("log")
plt.show()


x_grid = np.linspace(0, 1.5, 10000)
f = x_grid - np.sin(x_grid) * (1 + 2/3*np.sin(x_grid/2)**2)
plt.plot(x_grid, f)
plt.show()


def optimal_beta(kz_max, k_min, k_max):
    return fm.sqrt((k_min**2 + k_max**2 - kz_max**2)/2)


pade_order = (7, 8)
beta = 0.1#optimal_beta(2*fm.pi*fm.sin(fm.radians(1)), 2*fm.pi, 2*fm.pi)
dx = 100
coefs, c0 = pade_propagator_coefs(pade_order=pade_order, beta=beta, dx=dx, a0=0)

xis = np.linspace(-10, 10, 1000)
err = rational_approx_error(beta * dx, xis, coefs, c0)

plt.plot(xis, 10*np.log10(err))
plt.grid()
plt.show()


c0 = 1500
f = 50
k0 = 2*fm.pi*f/c0
kz_max = k0*fm.sin(fm.radians(10))
eps_x_max = 1e-2
x_max = 20000
beta, dx, dz = get_optimal_grid(kz_max, k0, k0, eps_x_max / x_max)
print(f'beta: {beta}, dx: {dx}, dz: {dz}')
