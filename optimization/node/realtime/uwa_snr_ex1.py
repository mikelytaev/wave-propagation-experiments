from copy import deepcopy

import jax.numpy as jnp
import numpy as np
from jax.experimental.array_api import linspace
from matplotlib.colors import Normalize

from experimental.helmholtz_jax import PiecewiseLinearWaveSpeedModel
from experiments.optimization.node.realtime.utils import get_field, realtime_inversion_model
from experimental.uwa_jax import UWAComputationalParams, uwa_get_model

import matplotlib.pyplot as plt

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

res_200_5_snr60 = realtime_inversion_model(200, 5000, simulated_ssp_list, snr=60, gamma=1)
res_200_5_snr30 = realtime_inversion_model(200, 5000.1, simulated_ssp_list, snr=30, gamma=100)
res_200_5_snr20 = realtime_inversion_model(200, 5000.15, simulated_ssp_list, snr=20, gamma=300)
res_200_5_snr10 = realtime_inversion_model(200, 5000.25, simulated_ssp_list, snr=10, gamma=500)


f, ax = plt.subplots(6, 1, figsize=(10, 11), constrained_layout=True)
for i, simulated_ssp in enumerate(simulated_ssp_list):
    ax[0].plot(simulated_ssp.sound_speed[::-1] + i, simulated_ssp.z_grid_m[::-1])
ax[0].set_title('Original SSP profiles')
ax[0].set_xticklabels([])
ax[0].set_ylabel("Depth (m)")
ax[0].set_ylim([simulated_ssp.z_grid_m[-1], simulated_ssp.z_grid_m[0]])
ax[0].grid(True)

for i, inverted_ssp in enumerate(res_200_5_snr60.inverted_ssp_list):
    ax[1].plot(inverted_ssp.sound_speed[::-1] + i, inverted_ssp.z_grid_m[::-1])
ax[1].set_title('Inverted SSP profiles (f = 200 Hz, range = 5 km, Pade-[7/8], SNR=60 dB')
ax[1].set_xticklabels([])
ax[1].set_ylabel("Depth (m)")
ax[1].set_ylim([inverted_ssp.z_grid_m[-1], inverted_ssp.z_grid_m[0]])
ax[1].grid(True)

for i, inverted_ssp in enumerate(res_200_5_snr30.inverted_ssp_list):
    ax[2].plot(inverted_ssp.sound_speed[::-1] + i, inverted_ssp.z_grid_m[::-1])
ax[2].set_title('Inverted SSP profiles (f = 200 Hz, range = 5 km, Pade-[7/8], SNR=30 dB')
ax[2].set_xticklabels([])
ax[2].set_ylabel("Depth (m)")
ax[2].set_ylim([inverted_ssp.z_grid_m[-1], inverted_ssp.z_grid_m[0]])
ax[2].grid(True)

for i, inverted_ssp in enumerate(res_200_5_snr20.inverted_ssp_list):
    ax[3].plot(inverted_ssp.sound_speed[::-1] + i, inverted_ssp.z_grid_m[::-1])
ax[3].set_title('Inverted SSP profiles (f = 200 Hz, range = 5 km, Pade-[7/8], SNR=20 dB')
ax[3].set_xticklabels([])
ax[3].set_ylabel("Depth (m)")
ax[3].set_ylim([inverted_ssp.z_grid_m[-1], inverted_ssp.z_grid_m[0]])
ax[3].grid(True)
plt.show()


plt.show()
#plt.savefig('ex1_ssp_dynamics.eps')

plt.figure(figsize=(10, 3.2), constrained_layout=True)
plt.plot(range(0, len(res_200_5_snr10.abs_error_list)), res_200_5_snr10.abs_error_list, label='SNR = 10 dB')
plt.plot(range(0, len(res_200_5_snr20.abs_error_list)), res_200_5_snr20.abs_error_list, label='SNR = 20 dB')
plt.plot(range(0, len(res_200_5_snr30.abs_error_list)), res_200_5_snr30.abs_error_list, label='SNR = 30 dB')
plt.plot(range(0, len(res_200_5_snr60.abs_error_list)), res_200_5_snr60.abs_error_list, label='SNR = 60 dB')
plt.legend()
plt.xlabel('Time step number')
plt.xticks(range(0, len(res_200_5_snr30.abs_error_list))[::2])
plt.xlim([0, len(res_200_5_snr30.abs_error_list) - 1])
plt.ylim([0, 1])
plt.ylabel('||abs. error||')
plt.grid(True)
plt.show()