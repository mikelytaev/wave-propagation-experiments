import numpy as np
from matplotlib.colors import Normalize
import matplotlib
from scipy.stats import norm

from experiments.stohastic_pe.jeet.utils import get_modified_evaporation_duct
from rwp.environment import surface_duct, evaporation_duct
import matplotlib.pyplot as plt
import math as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def plot_profile_prop_density(mean_height, sigma_height, ax, max_height_m=300, m_bounds=None):
    z_grid_m = np.linspace(0, max_height_m, 1000)

    for h in np.linspace(max(0, mean_height - 3*sigma_height), mean_height + 3*sigma_height, 150):
        ax.plot(get_modified_evaporation_duct(h, 99)(0, z_grid_m), z_grid_m, color='lightgray')

    for h in np.linspace(max(0, mean_height - sigma_height), mean_height + sigma_height, 30):
        ax.plot(get_modified_evaporation_duct(h, 99)(0, z_grid_m), z_grid_m, color='gray')

    ax.plot(get_modified_evaporation_duct(mean_height, 99)(0, z_grid_m), z_grid_m, color='black')
    ax.set_xlim([320, 338])
    ax.set_ylim([z_grid_m[0], z_grid_m[-1]])

    ax.grid(True)





n = 1000
m_0 = 320
f, ax = plt.subplots(2, 5, figsize=(9, 6.2))

plt.rcParams['font.size'] = '13'

plot_profile_prop_density(mean_height=10, sigma_height=0.5,
                          max_height_m=100, ax=ax[0, 0], m_bounds=[290, 390]
                          )
plot_profile_prop_density(10, 1,
                          max_height_m=100, ax=ax[0, 1], m_bounds=[290, 390]
                          )
plot_profile_prop_density(10, 2,
                               max_height_m=100, ax=ax[0, 2], m_bounds=[290, 390]
                               )
plot_profile_prop_density(10, 3,
                               max_height_m=100, ax=ax[0, 3], m_bounds=[290, 390]
                               )
plot_profile_prop_density(10, 4,
                               max_height_m=100, ax=ax[0, 4], m_bounds=[290, 390]
                               )

plot_profile_prop_density(20, 0.5,
                          max_height_m=100, ax=ax[1, 0], m_bounds=[290, 390]
                          )
plot_profile_prop_density(20, 1,
                               max_height_m=100, ax=ax[1, 1], m_bounds=[290, 390]
                               )
plot_profile_prop_density(20, 2,
                               max_height_m=100, ax=ax[1, 2], m_bounds=[290, 390]
                               )
plot_profile_prop_density(20, 3,
                               max_height_m=100, ax=ax[1, 3], m_bounds=[290, 390]
                               )
plot_profile_prop_density(20, 4,
                               max_height_m=100, ax=ax[1, 4], m_bounds=[290, 390]
                               )
ax[0, 0].set_ylabel("Height (m)", fontsize=13)
ax[1, 0].set_ylabel("Height (m)", fontsize=13)
ax[1, 0].set_xlabel("M-profile", fontsize=13)
ax[1, 1].set_xlabel("M-profile", fontsize=13)
ax[1, 2].set_xlabel("M-profile", fontsize=13)
ax[1, 3].set_xlabel("M-profile", fontsize=13)
ax[1, 4].set_xlabel("M-profile", fontsize=13)


ax[0, 0].set_title("h=10, σ=0.5")
ax[0, 1].set_title("h=10, σ=1")
ax[0, 2].set_title("h=10, σ=2")
ax[0, 3].set_title("h=10, σ=3")
ax[0, 4].set_title("h=10, σ=4")

ax[1, 0].set_title("h=20, σ=0.5")
ax[1, 1].set_title("h=20, σ=1")
ax[1, 2].set_title("h=20, σ=2")
ax[1, 3].set_title("h=20, σ=3")
ax[1, 4].set_title("h=20, σ=4")

ax[0, 1].set_yticklabels([])
ax[0, 2].set_yticklabels([])
ax[0, 3].set_yticklabels([])
ax[0, 4].set_yticklabels([])
ax[1, 1].set_yticklabels([])
ax[1, 2].set_yticklabels([])
ax[1, 3].set_yticklabels([])
ax[1, 4].set_yticklabels([])
ax[0, 0].set_xticklabels([])
ax[0, 1].set_xticklabels([])
ax[0, 2].set_xticklabels([])
ax[0, 3].set_xticklabels([])
ax[0, 4].set_xticklabels([])

for a in ax[0,:]:
    for label in (a.get_xticklabels() + a.get_yticklabels()):
        label.set_fontsize(13)
for a in ax[1,:]:
    for label in (a.get_xticklabels() + a.get_yticklabels()):
        label.set_fontsize(13)

legend_elements = [Line2D([0], [0], color='black', lw=1, label='Mean duct'),
                   Line2D([0], [0], color='gray', lw=8, label='±σ'),
                   Line2D([0], [0], color='lightgray', lw=8, label='±3σ')]


f.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.00), ncol=3)
f.tight_layout()

plt.show()
plt.savefig('ducts.eps')