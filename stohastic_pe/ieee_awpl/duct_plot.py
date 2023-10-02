import numpy as np
from matplotlib.colors import Normalize
import matplotlib
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
    f, ax = plt.subplots(2, 6, figsize=(9, 6), constrained_layout=True)
    duct = lambda z, v: evaporation_duct(height=v, z_grid_m=z, m_0=m_0)

    plt.rcParams['font.size'] = '13'

    plot_profile_prop_density(duct,
                              generator=lambda : np.random.normal(10, 0.001),
                              max_height_m=200, ax=ax[0, 0], m_bounds=[290, 390]
                              )
    plot_profile_prop_density(duct,
                              generator=lambda: np.random.normal(10,0.5),
                              max_height_m=200, ax=ax[0, 1], m_bounds=[290, 390]
                              )
    im = plot_profile_prop_density(duct,
                              generator=lambda: np.random.normal(10, 1),
                              max_height_m=200, ax=ax[0, 2], m_bounds=[290, 390]
                              )
    im = plot_profile_prop_density(duct,
                                   generator=lambda: np.random.normal(10, 2),
                                   max_height_m=200, ax=ax[0, 3], m_bounds=[290, 390]
                                   )
    im = plot_profile_prop_density(duct,
                                   generator=lambda: np.random.normal(10, 3),
                                   max_height_m=200, ax=ax[0, 4], m_bounds=[290, 390]
                                   )
    im = plot_profile_prop_density(duct,
                                   generator=lambda: np.random.normal(10, 4),
                                   max_height_m=200, ax=ax[0, 5], m_bounds=[290, 390]
                                   )

    plot_profile_prop_density(duct,
                              generator=lambda: np.random.normal(20, 0.001),
                              max_height_m=200, ax=ax[1, 0], m_bounds=[290, 390]
                              )
    plot_profile_prop_density(duct,
                              generator=lambda: np.random.normal(20, 0.5),
                              max_height_m=200, ax=ax[1, 1], m_bounds=[290, 390]
                              )
    im = plot_profile_prop_density(duct,
                                   generator=lambda: np.random.normal(20, 1),
                                   max_height_m=200, ax=ax[1, 2], m_bounds=[290, 390]
                                   )
    im = plot_profile_prop_density(duct,
                                   generator=lambda: np.random.normal(20, 2),
                                   max_height_m=200, ax=ax[1, 3], m_bounds=[290, 390]
                                   )
    im = plot_profile_prop_density(duct,
                                   generator=lambda: np.random.normal(20, 3),
                                   max_height_m=200, ax=ax[1, 4], m_bounds=[290, 390]
                                   )
    im = plot_profile_prop_density(duct,
                                   generator=lambda: np.random.normal(20, 4),
                                   max_height_m=200, ax=ax[1, 5], m_bounds=[290, 390]
                                   )
    ax[0, 0].set_ylabel("Height (m)", fontsize=13)
    ax[1, 0].set_ylabel("Height (m)", fontsize=13)
    ax[1, 0].set_xlabel("M-profile", fontsize=13)
    ax[1, 1].set_xlabel("M-profile", fontsize=13)
    ax[1, 2].set_xlabel("M-profile", fontsize=13)
    ax[1, 3].set_xlabel("M-profile", fontsize=13)
    ax[1, 4].set_xlabel("M-profile", fontsize=13)
    ax[1, 5].set_xlabel("M-profile", fontsize=13)

    ax[0, 0].set_title("h=10, σ=0")
    ax[0, 1].set_title("h=10, σ=0.5")
    ax[0, 2].set_title("h=10, σ=1")
    ax[0, 3].set_title("h=10, σ=2")
    ax[0, 4].set_title("h=10, σ=3")
    ax[0, 5].set_title("h=10, σ=4")
    ax[1, 0].set_title("h=20, σ=0")
    ax[1, 1].set_title("h=20, σ=0.5")
    ax[1, 2].set_title("h=20, σ=1")
    ax[1, 3].set_title("h=20, σ=2")
    ax[1, 4].set_title("h=20, σ=3")
    ax[1, 5].set_title("h=20, σ=4")

    ax[0, 1].set_yticklabels([])
    ax[0, 2].set_yticklabels([])
    ax[0, 3].set_yticklabels([])
    ax[0, 4].set_yticklabels([])
    ax[0, 5].set_yticklabels([])
    ax[1, 1].set_yticklabels([])
    ax[1, 2].set_yticklabels([])
    ax[1, 3].set_yticklabels([])
    ax[1, 4].set_yticklabels([])
    ax[1, 5].set_yticklabels([])
    ax[0, 0].set_xticklabels([])
    ax[0, 1].set_xticklabels([])
    ax[0, 2].set_xticklabels([])
    ax[0, 3].set_xticklabels([])
    ax[0, 4].set_xticklabels([])
    ax[0, 5].set_xticklabels([])

    for a in ax[0,:]:
        for label in (a.get_xticklabels() + a.get_yticklabels()):
            label.set_fontsize(13)
    for a in ax[1,:]:
        for label in (a.get_xticklabels() + a.get_yticklabels()):
            label.set_fontsize(13)

    f.colorbar(im, ax=ax[:], fraction=0.046*2/3, location='bottom')
    #plt.show()
    plt.savefig('ducts.eps')