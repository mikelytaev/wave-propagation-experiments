import numpy as np
from matplotlib.colors import Normalize
from scipy.stats import norm

from rwp.environment import surface_duct
import matplotlib.pyplot as plt
import math as fm


def plot_profile_prop_density(profile_func, generator):
    n = 1000
    x_n = 1000
    z_grid_m = np.linspace(0, 300, 1000)
    pic = np.zeros((x_n, len(z_grid_m))) + 1e-16
    vals = [generator() for _ in range(0, n)]
    m_min, m_max = fm.inf, 0
    for val in vals:
        m_profile = profile_func(z_grid_m, val)
        m_min, m_max = min(m_min, min(m_profile)), max(m_max, max(m_profile))
        for z_i in range(0, len(z_grid_m)):
            x_i = round(x_n * (m_profile[z_i] - m_min) / (m_max - m_min))
            if x_i >= x_n:
                continue
            pic[x_i, z_i] += 1

    pic /= len(vals)

    plt.figure(figsize=(3, 5))
    extent = [m_min, m_max, 0, z_grid_m[-1]]
    plt.imshow((pic.T[::-1, :]), norm=Normalize(0, 0.04), extent=extent, aspect='auto', cmap=plt.get_cmap('binary'))
    plt.colorbar(location='right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()





expected_height = 100
expected_strength = 20
n = 1000
m_0 = 320

plot_profile_prop_density(lambda z, v: surface_duct(height_m=v, strength=expected_strength, z_grid_m=z, m_0=m_0),
                          generator=lambda : np.random.normal(expected_height, 5))
