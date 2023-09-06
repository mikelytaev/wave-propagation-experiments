import numpy as np
from matplotlib.colors import Normalize
from scipy.stats import norm

from rwp.environment import surface_duct, evaporation_duct
import matplotlib.pyplot as plt
import math as fm


def plot_profile_prop_density(profile_func, generator, ax, max_height_m=300, m_bounds=None):
    n = 1000
    x_n = 1000
    z_grid_m = np.linspace(0, max_height_m, 1000)
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

    if m_bounds:
        m_min, m_max = m_bounds[0], m_bounds[1]
    extent = [m_min, m_max, 0, z_grid_m[-1]]
    im = ax.imshow((pic.T[::-1, :]), norm=Normalize(0, 0.02), extent=extent, aspect='auto', cmap=plt.get_cmap('binary'))
    #plt.colorbar(location='right')
    ax.grid(True)
    #plt.tight_layout()
    #plt.show()
    return im




if __name__ == "__main__":
    n = 1000
    m_0 = 320
    f, ax = plt.subplots(2, 4, figsize=(9, 6), constrained_layout=True)
    duct = lambda z, v: evaporation_duct(height=v, z_grid_m=z, m_0=m_0)
    plot_profile_prop_density(duct,
                              generator=lambda : np.random.normal(10, 0.001),
                              max_height_m=200, ax=ax[0, 0], m_bounds=[290, 390]
                              )
    plot_profile_prop_density(duct,
                              generator=lambda: np.random.normal(10, 1),
                              max_height_m=200, ax=ax[0, 1], m_bounds=[290, 390]
                              )
    im = plot_profile_prop_density(duct,
                              generator=lambda: np.random.normal(10, 5),
                              max_height_m=200, ax=ax[0, 2], m_bounds=[290, 390]
                              )
    im = plot_profile_prop_density(duct,
                                   generator=lambda: np.random.normal(10, 10),
                                   max_height_m=200, ax=ax[0, 3], m_bounds=[290, 390]
                                   )

    plot_profile_prop_density(duct,
                              generator=lambda: np.random.normal(20, 0.001),
                              max_height_m=200, ax=ax[1, 0], m_bounds=[290, 390]
                              )
    plot_profile_prop_density(duct,
                              generator=lambda: np.random.normal(20, 1),
                              max_height_m=200, ax=ax[1, 1], m_bounds=[290, 390]
                              )
    im = plot_profile_prop_density(duct,
                                   generator=lambda: np.random.normal(20, 5),
                                   max_height_m=200, ax=ax[1, 2], m_bounds=[290, 390]
                                   )
    im = plot_profile_prop_density(duct,
                                   generator=lambda: np.random.normal(20, 10),
                                   max_height_m=200, ax=ax[1, 3], m_bounds=[290, 390]
                                   )
    ax[0, 0].set_ylabel("Высота, м")
    ax[1, 0].set_ylabel("Высота, м")
    ax[1, 0].set_xlabel("M профиль, M-ед.")
    ax[1, 1].set_xlabel("M профиль, M-ед.")
    ax[1, 2].set_xlabel("M профиль, M-ед.")
    ax[1, 3].set_xlabel("M профиль, M-ед.")
    f.colorbar(im, ax=ax[:], fraction=0.046*2/3, location='bottom')
    plt.savefig('ducts.eps')