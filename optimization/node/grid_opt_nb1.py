import cmath as cm

import math as fm
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from propagators._utils import pade_propagator_coefs


def second_difference_disp_rel(k_z: complex, dz: float, z=0):
    return np.exp(1j*k_z*z) * (np.exp(-1j*k_z*dz) - 2 + np.exp(1j*k_z*dz))


def fourth_difference_disp_rel(k_z: complex, dz: float, z=0):
    return np.exp(1j*k_z*z) * (np.exp(-1j*k_z*dz) - 2 + np.exp(1j*k_z*dz))**2


def second_order_error_kz(k_z: float, dz: float):
    d = 1 / dz**2 * second_difference_disp_rel(k_z, dz)
    return abs(d - (-k_z**2))


def fourth_order_error_kz(k_z: float, dz: float):
    d = 1 / dz**2 * (second_difference_disp_rel(k_z, dz) - 1/12 * fourth_difference_disp_rel(k_z, dz))
    return abs(d - (-k_z**2))


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


def rational_approx_error(beta_dx: float, xi: np.array, rational_coefs, c0=1.0+0j):
    product = c0
    xi = xi + 0j
    for a_i, b_i in rational_coefs:
        product *= (1 + a_i * xi) / (1 + b_i * xi)

    ex = np.exp(1j * beta_dx * (np.sqrt(1.0 + xi) - 1.0))
    return abs(ex - product).real


def optimal_beta(kz_max, k_min, k_max):
    return fm.sqrt((k_min**2 + k_max**2 - kz_max**2)/2)


pade_order = (7, 8)
beta = 0.1#optimal_beta(2*fm.pi*fm.sin(fm.radians(1)), 2*fm.pi, 2*fm.pi)
dx = 100
coefs, c0 = pade_propagator_coefs(pade_order=pade_order, beta=beta, dx=dx, a0=0)

xis = np.linspace(-10, 10, 1000)
err = rational_approx_error(beta*dx, xis, coefs, c0)

plt.plot(xis, 10*np.log10(err))
plt.grid()
plt.show()


def _decreasing_func_binary_search(f, x_min, x_max, val, rel_prec):
    x_mid = (x_min + x_max) / 2
    f_mid = f(x_mid)
    if abs(f_mid - val) / val < rel_prec:
        return x_mid
    if f_mid > val:
        return _decreasing_func_binary_search(f, x_mid, x_max, val, rel_prec)
    else:
        return _decreasing_func_binary_search(f, x_min, x_mid, val, rel_prec)


def _increasing_func_binary_search(f, x_min, x_max, val, rel_prec):
    x_mid = (x_min + x_max) / 2
    f_mid = f(x_mid)
    if abs(f_mid - val) / val < rel_prec:
        return x_mid
    if f_mid > val:
        return _increasing_func_binary_search(f, x_min, x_mid, val, rel_prec)
    else:
        return _increasing_func_binary_search(f, x_mid, x_max, val, rel_prec)


def func_binary_search(f, x_min, x_max, val, rel_prec):
    f_min, f_max = f(x_min), f(x_max)
    if f_min > f_max:
        if val > f_min or val < f_max:
            return fm.nan
        return _decreasing_func_binary_search(f, x_min, x_max, val, rel_prec)
    else:
        if val > f_max or val < f_min:
            return fm.nan
        return _increasing_func_binary_search(f, x_min, x_max, val, rel_prec)


class OptTable:

    def __init__(self, order=(7, 8)):
        self.order = order
        self.t_betas = np.concatenate((
            np.linspace(0.1, 1, 10),
            np.linspace(2, 10, 9),
            np.linspace(20, 100, 9),
            np.linspace(200, 1000, 9),
            np.linspace(2000, 10000, 9),
            np.linspace(20000, 50000, 4)
        ))
        self.precs = np.array([1E-10, 1E-9, 1E-8, 1E-7, 1E-6, 1E-5,])
        self.xi_a = np.empty(shape=(len(self.t_betas), len(self.precs)), dtype=float)
        self.xi_b = np.empty(shape=(len(self.t_betas), len(self.precs)), dtype=float)

    def compute(self):
        for i_t_beta, t_beta in enumerate(self.t_betas):
                coefs, _ = pade_propagator_coefs(pade_order=self.order, beta=t_beta, dx=1)
                for i_prec, prec in enumerate(self.precs):
                    self.xi_a[i_t_beta, i_prec] = func_binary_search(
                        lambda xi: rational_approx_error(t_beta, xi, coefs), -1, 0, prec, 1e-3)
                    self.xi_b[i_t_beta, i_prec] = func_binary_search(
                        lambda xi: rational_approx_error(t_beta, xi, coefs), 0, 10, prec, 1e-3)
                    print(f't_beta={t_beta}, prec={prec}, [{self.xi_a[i_t_beta, i_prec]}, {self.xi_b[i_t_beta, i_prec]}]')

    @classmethod
    def load(cls):
        pass

    @classmethod
    def save(cls):
        pass

    def get_optimal(self, kz_max, k_min, k_max, required_prec) -> (float, float, float):
        a = k_min**2 - kz_max**2
        b = k_max**2
        res = fm.nan, 0.0, fm.nan
        for i_t_beta, t_beta in enumerate(self.t_betas):
            for i_prec, prec in enumerate(self.precs):
                r = fm.sqrt(a / (self.xi_a[i_t_beta, i_prec] + 1))
                l = fm.sqrt(b / (self.xi_b[i_t_beta, i_prec] + 1))
                if l > r:
                    continue
                beta = l
                dx = t_beta / beta
                if prec / dx > required_prec:
                    continue
                if dx > res[1]:
                    res = (beta, dx, prec)
        return res


table = OptTable()
table.compute()

beta, dx, prec = table.get_optimal(2*fm.pi*fm.sin(fm.radians(1)), 2*fm.pi, 2*fm.pi, 1e-3)
