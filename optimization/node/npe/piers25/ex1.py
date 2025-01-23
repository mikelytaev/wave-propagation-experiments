import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

from experiments.optimization.node.npe.common import surface_based_duct_N, \
    evaporation_duct_N, elevated_evaporation_duct_N
from matplotlib.lines import Line2D

jax.config.update("jax_enable_x64", True)

from ex_evaporation import evaporation_duct_opt_res
from ex_surface_based import surface_based_opt_res
from ex_mixed import mixed_opt_res
from ex_terrain import terrain_opt_res


z_grid_o = jnp.linspace(0, 200, 250)

f, ax = plt.subplots(1, 4, figsize=(6, 3.4), constrained_layout=True)
ax[0].plot(evaporation_duct_N.M_Profile(z_grid_o), z_grid_o, color='blue', label="Original")
ax[0].plot(evaporation_duct_opt_res.res_profile.M_Profile(z_grid_o), z_grid_o, color='red', label="Inverted")
ax[0].set_title("(a)")
ax[0].set_xlabel("M-profile")
ax[0].set_ylabel("Height (m)")
ax[0].set_ylim([z_grid_o[0], z_grid_o[-1]])
ax[0].grid(True)

ax[1].plot(surface_based_duct_N.M_Profile(z_grid_o), z_grid_o, color='blue', label="Original")
ax[1].plot(surface_based_opt_res.res_profile.M_Profile(z_grid_o), z_grid_o, color='red', label="Inverted")
ax[1].set_title("(b)")
ax[1].set_xlabel("M-profile")
ax[1].set_yticklabels([])
ax[1].set_ylim([z_grid_o[0], z_grid_o[-1]])
ax[1].grid(True)

ax[2].plot(elevated_evaporation_duct_N.M_Profile(z_grid_o), z_grid_o, color='blue', label="Original")
ax[2].plot(mixed_opt_res.res_profile.M_Profile(z_grid_o), z_grid_o, color='red', label="Inverted")
ax[2].set_title("(c)")
ax[2].set_xlabel("M-profile")
ax[2].set_yticklabels([])
ax[2].set_ylim([z_grid_o[0], z_grid_o[-1]])
ax[2].grid(True)

ax[3].plot(surface_based_duct_N.M_Profile(z_grid_o), z_grid_o, color='blue', label="Original")
ax[3].plot(terrain_opt_res.res_profile.M_Profile(z_grid_o), z_grid_o, color='red', label="Inverted")
ax[3].set_title("(d)")
ax[3].set_xlabel("M-profile")
ax[3].set_yticklabels([])
ax[3].set_ylim([z_grid_o[0], z_grid_o[-1]])
ax[3].grid(True)

legend_elements = [Line2D([0], [0], color='blue', lw=1, label='Original'),
                   Line2D([0], [0], color='red', lw=1, label='Inverted')]
f.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.00), ncol=2)
f.tight_layout()
plt.show()
