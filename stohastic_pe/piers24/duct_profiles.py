import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from rwp.environment import RandomProfile, RandomSurfaceDuct, RandomTrilinearDuct
import math as fm
from scipy.stats import norm


def plot_profile_prop_density(M_profile: RandomProfile, ax, max_height_m=300, m_bounds=None):
    n = 300
    x_n = 200
    z_grid_m = np.linspace(0, max_height_m, 1000)
    pic = np.zeros((x_n, len(z_grid_m))) + 1e-16
    m_min, m_max = fm.inf, 0
    for _ in range(0, n):
        m_profile_grid = M_profile.get_sample()(0, z_grid_m)
        m_min, m_max = min(m_min, min(m_profile_grid)), max(m_max, max(m_profile_grid))
        for z_i in range(0, len(z_grid_m)):
            x_i = round(x_n * (m_profile_grid[z_i] - m_min) / (m_max - m_min))
            if x_i >= x_n:
                continue
            pic[x_i, z_i] += 1

    pic /= n

    if m_bounds:
        m_min, m_max = m_bounds[0], m_bounds[1]
    extent = [m_min, m_max, 0, z_grid_m[-1]]
    im = ax.imshow((pic.T[::-1, :]), norm=Normalize(0, 0.1), extent=extent, aspect='auto', cmap=plt.get_cmap('binary'))
    ax.grid(True)
    return im


sd0 = RandomSurfaceDuct(
        height=norm(loc=200, scale=0.00005),
        m0=norm(loc=350, scale=0.00005),
        m1=norm(loc=300, scale=0.00005),
        slope=norm(loc=0.15, scale=0.00005)
    )

sd1 = RandomSurfaceDuct(
        height=norm(loc=200, scale=1),
        m0=norm(loc=350, scale=1),
        m1=norm(loc=300, scale=1),
        slope=norm(loc=0.15, scale=0.005)
    )

sd2 = RandomSurfaceDuct(
        height=norm(loc=200, scale=2),
        m0=norm(loc=350, scale=2),
        m1=norm(loc=300, scale=2),
        slope=norm(loc=0.15, scale=0.005)
    )

sd3 = RandomSurfaceDuct(
        height=norm(loc=200, scale=5),
        m0=norm(loc=350, scale=5),
        m1=norm(loc=300, scale=5),
        slope=norm(loc=0.15, scale=0.005)
    )

plt.rcParams['font.size'] = '8'
f, ax = plt.subplots(1, 4, figsize=(6, 2.5), constrained_layout=True)
plot_profile_prop_density(sd0, ax[0])
plot_profile_prop_density(sd1, ax[1])
plot_profile_prop_density(sd2, ax[2])
im = plot_profile_prop_density(sd3, ax[3])

ax[0].set_ylabel("Height (m)", fontsize=8)
ax[0].set_title("σ=0 M-unit", fontsize=8)
ax[1].set_title("σ=1 M-unit", fontsize=8)
ax[2].set_title("σ=2 M-units", fontsize=8)
ax[3].set_title("σ=5 M-units", fontsize=8)
ax[0].set_xlabel("M-profile", fontsize=8)
ax[1].set_xlabel("M-profile", fontsize=8)
ax[2].set_xlabel("M-profile", fontsize=8)
ax[3].set_xlabel("M-profile", fontsize=8)

for a in ax[:]:
    for label in (a.get_xticklabels() + a.get_yticklabels()):
        label.set_fontsize(8)

f.colorbar(im, ax=ax[:], fraction=0.046*2/3, location='bottom')
plt.savefig("surface_duct_profile.eps")

sbd = RandomTrilinearDuct(
        z1=norm(loc=100, scale=3),
        z2=norm(loc=200, scale=3),
        m0=norm(loc=300, scale=3),
        m1=norm(loc=320, scale=3),
        m2=norm(loc=280, scale=3),
        slope=norm(loc=0.15, scale=0.005)
    )

ed = M_profile=RandomTrilinearDuct(
        z1=norm(loc=50, scale=3),
        z2=norm(loc=150, scale=3),
        m0=norm(loc=300, scale=3),
        m1=norm(loc=330, scale=3),
        m2=norm(loc=310, scale=3),
        slope=norm(loc=0.15, scale=0.005)
    )


plt.rcParams['font.size'] = '8'
f, ax = plt.subplots(1, 2, figsize=(3, 2.5), constrained_layout=True)
plot_profile_prop_density(sbd, ax[0])
im = plot_profile_prop_density(ed, ax[1])

ax[0].set_ylabel("Height (m)", fontsize=8)
ax[0].set_title("σ=3 M-units", fontsize=8)
ax[1].set_title("σ=3 M-units", fontsize=8)
ax[0].set_xlabel("M-profile", fontsize=8)
ax[1].set_xlabel("M-profile", fontsize=8)

for a in ax[:]:
    for label in (a.get_xticklabels() + a.get_yticklabels()):
        label.set_fontsize(8)

f.colorbar(im, ax=ax[:], fraction=0.046*2/3, location='bottom')
plt.savefig("triliear_duct_profile.eps")
