from copy import deepcopy

import jax.numpy as jnp
import numpy as np
from jax.experimental.array_api import linspace
from matplotlib.colors import Normalize
from networkx.algorithms.bipartite import color

from experimental.helmholtz_jax import PiecewiseLinearWaveSpeedModel
from experiments.optimization.node.realtime.utils import get_field, realtime_inversion_model
from experimental.uwa_jax import UWAComputationalParams, uwa_get_model

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#logging.basicConfig(level=logging.DEBUG)

simulated_ssp_list = []
for lower in linspace(1500.0, 1510, 21):
    simulated_ssp_list += [
        PiecewiseLinearWaveSpeedModel(
                    z_grid_m=jnp.array([0.0, 200.0]),
                    sound_speed=jnp.array([1500.0, lower])
                )]

for upper in linspace(1500.0, 1510, 21):
    simulated_ssp_list += [
        PiecewiseLinearWaveSpeedModel(
            z_grid_m=jnp.array([0.0, 75, 200]),
            sound_speed=jnp.array([upper, (lower+1500.0)/2, lower])
        )]

ms = simulated_ssp_list[-1](75.0)
for middle in linspace(ms, 1510, 10):
    simulated_ssp_list += [
        PiecewiseLinearWaveSpeedModel(
            z_grid_m=jnp.array([0.0, 75, 200]),
            sound_speed=jnp.array([upper, middle, lower])
        )]

res_500_10_2 = realtime_inversion_model(500, 12000, simulated_ssp_list, snr=30, gamma=100, arrays_num=2)

# for i, simulated_ssp in enumerate(simulated_ssp_list):
#     res_200_5.inverted_ssp_list[i].sound_speed += (simulated_ssp.sound_speed[0] -
#                                                    res_200_5.inverted_ssp_list[i].sound_speed[0])
#     res_200_10.inverted_ssp_list[i].sound_speed += simulated_ssp.sound_speed[0] - \
#                                                   res_200_10.inverted_ssp_list[i].sound_speed[0]
#     res_500_5.inverted_ssp_list[i].sound_speed += simulated_ssp.sound_speed[0] - \
#                                                   res_500_5.inverted_ssp_list[i].sound_speed[0]
#     res_500_10.inverted_ssp_list[i].sound_speed += simulated_ssp.sound_speed[0] - \
#                                                   res_500_10.inverted_ssp_list[i].sound_speed[0]
#     res_500_5_cn.inverted_ssp_list[i].sound_speed += simulated_ssp.sound_speed[0] - \
#                                                    res_500_5_cn.inverted_ssp_list[i].sound_speed[0]





plt.figure(figsize=(10, 3.2), constrained_layout=True)
plt.plot(range(0, len(res_500_10_2.abs_error_list)), res_500_10_2.abs_error_list, label='f = 500 Hz, range = 10 km, Pade-[7/8]')
plt.legend()
plt.xlabel('Number of iteration')
plt.xticks(range(0, len(res_500_10_2.abs_error_list))[::2])
plt.xlim([0, len(res_500_10_2.abs_error_list)-1])
plt.ylim([0, 1])
plt.ylabel('||abs. error||')
plt.grid(True)
plt.show()
#plt.savefig('ex1_ssp_rel_error_norm.eps')
