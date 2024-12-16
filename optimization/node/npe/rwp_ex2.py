import jax

import matplotlib.pyplot as plt

from experimental.rwp_jax import RWPComputationalParams
import jax.numpy as jnp

from experiments.optimization.node.flax.utils import MLPNProfileModel
from experiments.optimization.node.npe.common import adam, RWPModel, add_noise, surface_based_duct_N, surface_duct_N, \
    elevated_duct_N, surface_based_duct2_N
from matplotlib.lines import Line2D

jax.config.update("jax_enable_x64", True)

model = RWPModel(params=RWPComputationalParams(
        max_range_m=5000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*18,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170],
)


measure_surface_based_duct = model.apply_profile(surface_based_duct_N)
measure_surface_based_duct = add_noise(measure_surface_based_duct, 30)

surface_based_duct_opt_res_10_1 = adam(model, measure_surface_based_duct, ground_truth_profile=surface_based_duct_N, learning_rate=0.05, profile_model=MLPNProfileModel(z_max_m=100.0, layers=[10]*1))
surface_based_duct_opt_res_50_1 = adam(model, measure_surface_based_duct, ground_truth_profile=surface_based_duct_N, learning_rate=0.05, profile_model=MLPNProfileModel(z_max_m=100.0, layers=[50]*1))
surface_based_duct_opt_res_100_1 = adam(model, measure_surface_based_duct, ground_truth_profile=surface_based_duct_N, learning_rate=0.05, profile_model=MLPNProfileModel(z_max_m=100.0, layers=[100]*1))

surface_based_duct_opt_res_10_4 = adam(model, measure_surface_based_duct, ground_truth_profile=surface_based_duct_N, learning_rate=0.05, profile_model=MLPNProfileModel(z_max_m=100.0, layers=[10]*4))
surface_based_duct_opt_res_50_4 = adam(model, measure_surface_based_duct, ground_truth_profile=surface_based_duct_N, learning_rate=0.05, profile_model=MLPNProfileModel(z_max_m=100.0, layers=[50]*4))
surface_based_duct_opt_res_100_4 = adam(model, measure_surface_based_duct, ground_truth_profile=surface_based_duct_N, learning_rate=0.05, profile_model=MLPNProfileModel(z_max_m=100.0, layers=[100]*4))

surface_based_duct_opt_res_10_8 = adam(model, measure_surface_based_duct, ground_truth_profile=surface_based_duct_N, learning_rate=0.05, profile_model=MLPNProfileModel(z_max_m=100.0, layers=[10]*8))
surface_based_duct_opt_res_50_8 = adam(model, measure_surface_based_duct, ground_truth_profile=surface_based_duct_N, learning_rate=0.05, profile_model=MLPNProfileModel(z_max_m=100.0, layers=[50]*8))
surface_based_duct_opt_res_100_8 = adam(model, measure_surface_based_duct, ground_truth_profile=surface_based_duct_N, learning_rate=0.05, profile_model=MLPNProfileModel(z_max_m=100.0, layers=[100]*8))


f, ax = plt.subplots(1, 3, figsize=(6, 3.2), constrained_layout=True)
ylim = [0.03, 10]
ax[0].plot(surface_based_duct_opt_res_10_1.ground_truth_errors, label="l=1; w=10", color="blue", linestyle="-")
ax[0].plot(surface_based_duct_opt_res_50_1.ground_truth_errors, label="l=1; w=50", color="red", linestyle="-")
ax[0].plot(surface_based_duct_opt_res_100_1.ground_truth_errors, label="l=1; w=100", color="green", linestyle="-")
ax[0].set_xlim([0, 250])
ax[0].set_ylim(ylim)
ax[0].set_xlabel("Iteration number")
ax[0].set_ylabel("Rel. error")
ax[0].set_yscale("log")
ax[0].grid(True)
ax[0].legend()

ax[1].plot(surface_based_duct_opt_res_10_4.ground_truth_errors, label="l=4; w=10", color="blue", linestyle="-")
ax[1].plot(surface_based_duct_opt_res_50_4.ground_truth_errors, label="l=4; w=50", color="red", linestyle="-")
ax[1].plot(surface_based_duct_opt_res_100_4.ground_truth_errors, label="l=4; w=100", color="green", linestyle="-")
ax[1].set_xlim([0, 250])
ax[1].set_ylim(ylim)
ax[1].set_xlabel("Iteration number")
ax[1].set_yscale("log")
ax[1].set_yticklabels([])
ax[1].grid(True)
ax[1].legend()

ax[2].plot(surface_based_duct_opt_res_10_8.ground_truth_errors, label="l=8; w=10", color="blue", linestyle="-")
ax[2].plot(surface_based_duct_opt_res_50_8.ground_truth_errors, label="l=8; w=50", color="red", linestyle="-")
ax[2].plot(surface_based_duct_opt_res_100_8.ground_truth_errors, label="l=8; w=100", color="green", linestyle="-")
ax[2].set_xlim([0, 250])
ax[2].set_ylim(ylim)
ax[2].set_xlabel("Iteration number")
ax[2].set_yscale("log")
ax[2].set_yticklabels([])
ax[2].grid(True)
ax[2].legend()
plt.show()

z_grid_o = jnp.linspace(0, 250, 250)
plt.figure()
plt.plot(surface_based_duct_N(z_grid_o), z_grid_o)
plt.plot(surface_based_duct_opt_res_10_1.res_profile(z_grid_o), z_grid_o, color="blue", linestyle="-")
plt.plot(surface_based_duct_opt_res_50_1.res_profile(z_grid_o), z_grid_o, color="red", linestyle="-")
plt.plot(surface_based_duct_opt_res_100_1.res_profile(z_grid_o), z_grid_o, color="green", linestyle="-")
plt.show()

z_grid_o = jnp.linspace(0, 250, 250)
plt.figure()
plt.plot(surface_based_duct_N(z_grid_o), z_grid_o)
plt.plot(surface_based_duct_opt_res_10_4.res_profile(z_grid_o), z_grid_o, color="blue", linestyle="-")
plt.plot(surface_based_duct_opt_res_50_4.res_profile(z_grid_o), z_grid_o, color="red", linestyle="-")
plt.plot(surface_based_duct_opt_res_100_4.res_profile(z_grid_o), z_grid_o, color="green", linestyle="-")
plt.show()


z_grid_o = jnp.linspace(0, 150, 250)
f, ax = plt.subplots(1, 3, figsize=(6, 3.4), constrained_layout=True)
ax[0].set_title("1 layer")
ax[0].plot(surface_based_duct_N.M_Profile(z_grid_o), z_grid_o, color="black")
ax[0].plot(surface_based_duct_opt_res_10_1.res_profile.M_Profile(z_grid_o), z_grid_o, color="blue", linestyle="-")
ax[0].plot(surface_based_duct_opt_res_50_1.res_profile.M_Profile(z_grid_o), z_grid_o, color="red", linestyle="-")
ax[0].plot(surface_based_duct_opt_res_100_1.res_profile.M_Profile(z_grid_o), z_grid_o, color="green", linestyle="-")
ax[0].set_xlabel("M-profile")
ax[0].set_ylabel("Height (m)")
ax[0].set_ylim([z_grid_o[0], z_grid_o[-1]])
ax[0].grid(True)

ax[1].set_title("4 layers")
ax[1].plot(surface_based_duct_N.M_Profile(z_grid_o), z_grid_o, color="black")
ax[1].plot(surface_based_duct_opt_res_10_4.res_profile.M_Profile(z_grid_o), z_grid_o, color="blue", linestyle="-")
ax[1].plot(surface_based_duct_opt_res_50_4.res_profile.M_Profile(z_grid_o), z_grid_o, color="red", linestyle="-")
ax[1].plot(surface_based_duct_opt_res_100_4.res_profile.M_Profile(z_grid_o), z_grid_o, color="green", linestyle="-")
ax[1].set_xlabel("M-profile")
ax[1].set_yticklabels([])
ax[1].set_ylim([z_grid_o[0], z_grid_o[-1]])
ax[1].grid(True)

ax[2].set_title("8 layers")
ax[2].plot(surface_based_duct_N.M_Profile(z_grid_o), z_grid_o, color="black")
ax[2].plot(surface_based_duct_opt_res_10_8.res_profile.M_Profile(z_grid_o), z_grid_o, color="blue", linestyle="-")
ax[2].plot(surface_based_duct_opt_res_50_8.res_profile.M_Profile(z_grid_o), z_grid_o, color="red", linestyle="-")
ax[2].plot(surface_based_duct_opt_res_100_8.res_profile.M_Profile(z_grid_o), z_grid_o, color="green", linestyle="-")
ax[2].set_xlabel("M-profile")
ax[2].set_yticklabels([])
ax[2].set_ylim([z_grid_o[0], z_grid_o[-1]])
ax[2].grid(True)

legend_elements = [Line2D([0], [0], color='black', lw=1, label='Original'),
                   Line2D([0], [0], color='blue', lw=1, label='w=10'),
                   Line2D([0], [0], color='red', lw=1, label='w=50'),
                   Line2D([0], [0], color='green', lw=1, label='w=100')
                   ]
f.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.00), ncol=4)
f.tight_layout()

plt.show()