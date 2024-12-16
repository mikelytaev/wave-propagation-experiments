from cProfile import label

import jax

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.pyplot import title
from networkx.algorithms.bipartite import color

from experimental.rwp_jax import RWPGaussSourceModel, RWPComputationalParams, PiecewiseLinearNProfileModel
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

surface_based_duct_opt_res_1 = adam(model, measure_surface_based_duct, ground_truth_profile=surface_based_duct_N, gamma=0, learning_rate=0.05, profile_model=MLPNProfileModel(z_max_m=100.0, layers=[40]*1))
surface_based_duct_opt_res_2 = adam(model, measure_surface_based_duct, ground_truth_profile=surface_based_duct_N, learning_rate=0.05, profile_model=MLPNProfileModel(z_max_m=100.0, layers=[40]*1))
surface_based_duct_opt_res_3 = adam(model, measure_surface_based_duct, ground_truth_profile=surface_based_duct_N, gamma=0, learning_rate=0.05, profile_model=MLPNProfileModel(z_max_m=100.0, layers=[40]*5))
surface_based_duct_opt_res_4 = adam(model, measure_surface_based_duct, ground_truth_profile=surface_based_duct_N, learning_rate=0.05, profile_model=MLPNProfileModel(z_max_m=100.0, layers=[40]*5))


plt.figure(figsize=(6, 3.2))
plt.plot(surface_based_duct_opt_res_1.ground_truth_errors, label="1", color="blue", linestyle="-")
plt.plot(surface_based_duct_opt_res_2.ground_truth_errors, label="2", color="blue", linestyle="--")
plt.plot(surface_based_duct_opt_res_3.ground_truth_errors, label="3", color="red", linestyle="-")
plt.plot(surface_based_duct_opt_res_4.ground_truth_errors, label="4", color="red", linestyle="--")
plt.xlim([0, 250])
plt.xlabel("Iteration number")
plt.ylabel("Loss")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

z_grid_o = jnp.linspace(0, 250, 250)
plt.figure()
plt.plot(surface_based_duct_N(z_grid_o), z_grid_o)
plt.plot(surface_based_duct_opt_res_1.res_profile(z_grid_o), z_grid_o, color="blue", linestyle="-")
plt.plot(surface_based_duct_opt_res_2.res_profile(z_grid_o), z_grid_o, color="blue", linestyle="--")
plt.plot(surface_based_duct_opt_res_3.res_profile(z_grid_o), z_grid_o, color="red", linestyle="-")
plt.plot(surface_based_duct_opt_res_4.res_profile(z_grid_o), z_grid_o, color="red", linestyle="--")
plt.show()