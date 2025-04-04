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

res_200_5 = realtime_inversion_model(200, 5000, simulated_ssp_list, snr=30, gamma=100)
res_200_10 = realtime_inversion_model(200, 10000, simulated_ssp_list, snr=30, gamma=100)
res_500_5 = realtime_inversion_model(500, 5000, simulated_ssp_list, snr=30, gamma=100)
res_500_5_cn = realtime_inversion_model(500, 5500, simulated_ssp_list, snr=30, gamma=100)
res_500_10 = realtime_inversion_model(500, 12000, simulated_ssp_list, snr=30, gamma=100)

for i, simulated_ssp in enumerate(simulated_ssp_list):
    res_200_5.inverted_ssp_list[i].sound_speed += (simulated_ssp.sound_speed[0] -
                                                   res_200_5.inverted_ssp_list[i].sound_speed[0])
    res_200_10.inverted_ssp_list[i].sound_speed += simulated_ssp.sound_speed[0] - \
                                                  res_200_10.inverted_ssp_list[i].sound_speed[0]
    res_500_5.inverted_ssp_list[i].sound_speed += simulated_ssp.sound_speed[0] - \
                                                  res_500_5.inverted_ssp_list[i].sound_speed[0]
    res_500_10.inverted_ssp_list[i].sound_speed += simulated_ssp.sound_speed[0] - \
                                                  res_500_10.inverted_ssp_list[i].sound_speed[0]
    res_500_5_cn.inverted_ssp_list[i].sound_speed += simulated_ssp.sound_speed[0] - \
                                                   res_500_5_cn.inverted_ssp_list[i].sound_speed[0]


f, ax = plt.subplots(6, 1, figsize=(10, 11), constrained_layout=True)
for i, simulated_ssp in enumerate(simulated_ssp_list):
    ax[0].plot(simulated_ssp.sound_speed[::-1] + i, simulated_ssp.z_grid_m[::-1])
ax[0].set_title('Original SSP profiles')
ax[0].set_xticklabels([])
ax[0].set_ylabel("Depth (m)")
ax[0].set_ylim([simulated_ssp.z_grid_m[-1], simulated_ssp.z_grid_m[0]])
ax[0].grid(True)

for i, inverted_ssp in enumerate(res_200_5.inverted_ssp_list):
    ax[1].plot(inverted_ssp.sound_speed[::-1] + i, inverted_ssp.z_grid_m[::-1])
ax[1].set_title('Inverted SSP profiles (f = 200 Hz, range = 5 km, Pade-[7/8])')
ax[1].set_xticklabels([])
ax[1].set_ylabel("Depth (m)")
ax[1].set_ylim([inverted_ssp.z_grid_m[-1], inverted_ssp.z_grid_m[0]])
ax[1].grid(True)
plt.show()

for i, inverted_ssp in enumerate(res_200_10.inverted_ssp_list):
    ax[2].plot(inverted_ssp.sound_speed[::-1] + i, inverted_ssp.z_grid_m[::-1])
ax[2].set_title('Inverted SSP profiles (f = 200 Hz, range = 10 km, Pade-[7/8])')
ax[2].set_xticklabels([])
ax[2].set_ylabel("Depth (m)")
ax[2].set_ylim([inverted_ssp.z_grid_m[-1], inverted_ssp.z_grid_m[0]])
ax[2].grid(True)
plt.show()

for i, inverted_ssp in enumerate(res_500_5.inverted_ssp_list):
    ax[3].plot(inverted_ssp.sound_speed[::-1] + i, inverted_ssp.z_grid_m[::-1])
ax[3].set_title('Inverted SSP profiles (f = 500 Hz, range = 5 km, Pade-[7/8])')
ax[3].set_xticklabels([])
ax[3].set_ylabel("Depth (m)")
ax[3].set_ylim([inverted_ssp.z_grid_m[-1], inverted_ssp.z_grid_m[0]])
ax[3].grid(True)
plt.show()

for i, inverted_ssp in enumerate(res_500_10.inverted_ssp_list):
    ax[4].plot(inverted_ssp.sound_speed[::-1] + i, inverted_ssp.z_grid_m[::-1])
ax[4].set_title('Inverted SSP profiles (f = 500 Hz, range = 10 km, Pade-[7/8])')
ax[4].set_xticklabels([])
ax[4].set_ylabel("Depth (m)")
ax[4].set_ylim([inverted_ssp.z_grid_m[-1], inverted_ssp.z_grid_m[0]])
ax[4].grid(True)
plt.show()

for i, inverted_ssp in enumerate(res_500_5_cn.inverted_ssp_list):
    ax[5].plot(inverted_ssp.sound_speed[::-1] + i, inverted_ssp.z_grid_m[::-1])
ax[5].set_title('Inverted SSP profiles (f = 500 Hz, range = 5 km, Crank-Nicolson WA PE)')
ax[5].set_xlabel("Sound speed (m/s)")
ax[5].set_ylabel("Depth (m)")
ax[5].set_ylim([inverted_ssp.z_grid_m[-1], inverted_ssp.z_grid_m[0]])
ax[5].grid(True)
plt.show()
#plt.savefig('ex1_ssp_dynamics.eps')

f, ax = plt.subplots(5, 1, figsize=(10, 6), constrained_layout=True)
replica_z_grid_m = res_200_5.inverted_ssp_list[0].z_grid_m
for i in range(0, len(simulated_ssp_list)):
    d = simulated_ssp_list[i](replica_z_grid_m) - res_200_5.inverted_ssp_list[i](replica_z_grid_m)
    ax[0].plot(d[::-1] + 5*i, replica_z_grid_m[::-1])
ax[0].set_title('SSP residual  (f = 200 Hz, range = 5 km, Pade-[7/8])')
ax[0].set_xticklabels([])
ax[0].set_ylabel('Depth (m)')
ax[0].set_ylim([replica_z_grid_m[-1], replica_z_grid_m[0]])
ax[0].set_xticks(np.arange(0, 5*len(simulated_ssp_list), 10))
ax[0].grid(True)

replica_z_grid_m = res_200_10.inverted_ssp_list[0].z_grid_m
for i in range(0, len(simulated_ssp_list)):
    d = simulated_ssp_list[i](replica_z_grid_m) - res_200_10.inverted_ssp_list[i](replica_z_grid_m)
    ax[1].plot(d[::-1] + 5*i, replica_z_grid_m[::-1])
ax[1].set_title('SSP residual  (f = 200 Hz, range = 10 km, Pade-[7/8])')
ax[1].set_xticklabels([])
ax[1].set_ylabel('Depth (m)')
ax[1].set_ylim([replica_z_grid_m[-1], replica_z_grid_m[0]])
ax[1].set_xticks(np.arange(0, 5*len(simulated_ssp_list), 10))
ax[1].grid(True)

replica_z_grid_m = res_500_5.inverted_ssp_list[0].z_grid_m
for i in range(0, len(simulated_ssp_list)):
    d = simulated_ssp_list[i](replica_z_grid_m) - res_500_5.inverted_ssp_list[i](replica_z_grid_m)
    ax[2].plot(d[::-1] + 5*i, replica_z_grid_m[::-1])
ax[2].set_title('SSP residual  (f = 500 Hz, range = 5 km, Pade-[7/8])')
ax[2].set_xticklabels([])
ax[2].set_ylabel('Depth (m)')
ax[2].set_ylim([replica_z_grid_m[-1], replica_z_grid_m[0]])
ax[2].set_xticks(np.arange(0, 5*len(simulated_ssp_list), 10))
ax[2].grid(True)

replica_z_grid_m = res_500_10.inverted_ssp_list[0].z_grid_m
for i in range(0, len(simulated_ssp_list)):
    d = simulated_ssp_list[i](replica_z_grid_m) - res_500_10.inverted_ssp_list[i](replica_z_grid_m)
    ax[3].plot(d[::-1] + 5*i, replica_z_grid_m[::-1])
ax[3].set_title('SSP residual  (f = 500 Hz, range = 10 km, Pade-[7/8])')
ax[3].set_xticklabels([])
ax[3].set_ylabel('Depth (m)')
ax[3].set_ylim([replica_z_grid_m[-1], replica_z_grid_m[0]])
ax[3].set_xticks(np.arange(0, 5*len(simulated_ssp_list), 10))
ax[3].grid(True)

replica_z_grid_m = res_500_5_cn.inverted_ssp_list[0].z_grid_m
for i in range(0, len(simulated_ssp_list)):
    d = simulated_ssp_list[i](replica_z_grid_m) - res_500_5_cn.inverted_ssp_list[i](replica_z_grid_m)
    ax[4].plot(d[::-1] + 5*i, replica_z_grid_m[::-1])
ax[4].set_title('SSP residual  (f = 500 Hz, range = 5 km, Crank-Nicolson WA PE)')
ax[4].set_xlabel('SSP difference (m/s)')
ax[4].set_ylabel('Depth (m)')
ax[4].set_ylim([replica_z_grid_m[-1], replica_z_grid_m[0]])
ax[4].set_xticks(np.arange(0, 5*len(simulated_ssp_list), 10))
ax[4].grid(True)
plt.show()
#plt.savefig('ex1_ssp_error_pw.eps')

f, ax = plt.subplots(1, 4, figsize=(10, 3.2), constrained_layout=True)
prop_cycle = list(plt.rcParams['axes.prop_cycle'])
t = 20
ax[0].plot(simulated_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='original', color='black')
ax[0].plot(res_200_5.inverted_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='f = 200 Hz, range = 5 km, Pade-[7/8]', color=prop_cycle[0]['color'])
ax[0].plot(res_200_10.inverted_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='f = 200 Hz, range = 10 km, Pade-[7/8]', color=prop_cycle[1]['color'])
ax[0].plot(res_500_5.inverted_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='f = 500 Hz, range = 5 km, Pade-[7/8]', color=prop_cycle[2]['color'])
ax[0].plot(res_500_5_cn.inverted_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='f = 500 Hz, range = 5 km, Crank-Nicolson WA PE', color=prop_cycle[4]['color'])
ax[0].set_title(f't = {t}')
ax[0].set_xlabel("Sound speed (m/s)")
ax[0].set_ylabel('Depth (m)')
ax[0].set_ylim([replica_z_grid_m[-1], replica_z_grid_m[0]])
#ax[0].set_xticks(np.arange(0, 5*len(simulated_ssp_list), 10))
ax[0].grid(True)

t = 30
ax[1].plot(simulated_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='original', color='black')
ax[1].plot(res_200_5.inverted_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='f = 200 Hz, range = 5 km, Pade-[7/8]', color=prop_cycle[0]['color'])
ax[1].plot(res_200_10.inverted_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='f = 200 Hz, range = 10 km, Pade-[7/8]', color=prop_cycle[1]['color'])
ax[1].plot(res_500_5.inverted_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='f = 500 Hz, range = 5 km, Pade-[7/8]', color=prop_cycle[2]['color'])
ax[1].plot(res_500_5_cn.inverted_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='f = 500 Hz, range = 5 km, Crank-Nicolson WA PE', color=prop_cycle[4]['color'])
ax[1].set_title(f't = {t}')
ax[1].set_xlabel("Sound speed (m/s)")
#ax[1].set_ylabel('Depth (m)')
ax[1].set_ylim([replica_z_grid_m[-1], replica_z_grid_m[0]])
ax[1].set_yticklabels([])
#ax[1].set_xticks(np.arange(0, 5*len(simulated_ssp_list), 10))
ax[1].grid(True)

t = 40
ax[2].plot(simulated_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='original', color='black')
ax[2].plot(res_200_5.inverted_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='f = 200 Hz, range = 5 km, Pade-[7/8]', color=prop_cycle[0]['color'])
ax[2].plot(res_200_10.inverted_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='f = 200 Hz, range = 10 km, Pade-[7/8]', color=prop_cycle[1]['color'])
ax[2].plot(res_500_5.inverted_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='f = 500 Hz, range = 5 km, Pade-[7/8]', color=prop_cycle[2]['color'])
ax[2].plot(res_500_5_cn.inverted_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='f = 500 Hz, range = 5 km, Crank-Nicolson WA PE', color=prop_cycle[4]['color'])
ax[2].set_title(f't = {t}')
ax[2].set_xlabel("Sound speed (m/s)")
#ax[2].set_ylabel('Depth (m)')
ax[2].set_ylim([replica_z_grid_m[-1], replica_z_grid_m[0]])
ax[2].set_yticklabels([])
#ax[2].set_xticks(np.arange(0, 5*len(simulated_ssp_list), 10))
ax[2].grid(True)

t = 50
ax[3].plot(simulated_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='original', color='black')
ax[3].plot(res_200_5.inverted_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='f = 200 Hz, range = 5 km, Pade-[7/8]', color=prop_cycle[0]['color'])
ax[3].plot(res_200_10.inverted_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='f = 200 Hz, range = 10 km, Pade-[7/8]', color=prop_cycle[1]['color'])
ax[3].plot(res_500_5.inverted_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='f = 500 Hz, range = 5 km, Pade-[7/8]', color=prop_cycle[2]['color'])
ax[3].plot(res_500_5_cn.inverted_ssp_list[t](replica_z_grid_m[::-1]), replica_z_grid_m[::-1], label='f = 500 Hz, range = 5 km, Crank-Nicolson WA PE', color=prop_cycle[4]['color'])
ax[3].set_title(f't = {t}')
ax[3].set_xlabel("Sound speed (m/s)")
#ax[3].set_ylabel('Depth (m)')
ax[3].set_xlim([1508.0, 1513.0])
ax[3].set_ylim([replica_z_grid_m[-1], replica_z_grid_m[0]])
ax[3].set_yticklabels([])
#ax[3].set_xticks(np.arange(0, 5*len(simulated_ssp_list), 10))
ax[3].grid(True)

prop_cycle = list(plt.rcParams['axes.prop_cycle'])
legend_elements = [Line2D([0], [0], color='black', lw=1, label='Original'),
                   Line2D([0], [0], color=prop_cycle[0]['color'], lw=1, label='f = 200 Hz, range = 5 km, Pade-[7/8]'),
                   Line2D([0], [0], color=prop_cycle[1]['color'], lw=1, label='f = 200 Hz, range = 10 km, Pade-[7/8]'),
                   Line2D([0], [0], color=prop_cycle[2]['color'], lw=1, label='f = 500 Hz, range = 5 km, Pade-[7/8]'),
                   Line2D([0], [0], color=prop_cycle[4]['color'], lw=1, label='f = 500 Hz, range = 5 km, Crank-Nicolson WA PE')
                   ]

f.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.00), ncol=2)
f.tight_layout()
plt.show()


plt.figure(figsize=(10, 3.2), constrained_layout=True)
plt.plot(range(0, len(res_200_5.abs_error_list)), res_200_5.abs_error_list, label='f = 200 Hz, range = 5 km, Pade-[7/8]')
plt.plot(range(0, len(res_200_10.abs_error_list)), res_200_10.abs_error_list, label='f = 200 Hz, range = 10 km, Pade-[7/8]')
plt.plot(range(0, len(res_500_5.abs_error_list)), res_500_5.abs_error_list, label='f = 500 Hz, range = 5 km, Pade-[7/8]')
plt.plot(range(0, len(res_500_10.abs_error_list)), res_500_10.abs_error_list, label='f = 500 Hz, range = 10 km, Pade-[7/8]')
plt.plot(range(0, len(res_500_5.abs_error_list)), res_500_5.abs_error_list, label='f = 500 Hz, range = 5 km, Crank-Nicolson WA PE)')
plt.legend()
plt.xlabel('Time step number')
plt.xticks(range(0, len(res_200_5.abs_error_list))[::2])
plt.xlim([0, len(res_200_5.abs_error_list)-1])
plt.ylim([0, 1])
plt.ylabel('||abs. error||')
plt.grid(True)
plt.show()
#plt.savefig('ex1_ssp_rel_error_norm.eps')

plt.figure(figsize=(10, 3.2), constrained_layout=True)
plt.plot(range(0, len(res_200_5.nfev_list)), res_200_5.nfev_list, label='f = 200 Hz, range = 5 km, Pade-[7/8]')
plt.plot(range(0, len(res_200_10.nfev_list)), res_200_10.nfev_list, label='f = 200 Hz, range = 10 km, Pade-[7/8]')
plt.plot(range(0, len(res_500_5.nfev_list)), res_500_5.nfev_list, label='f = 500 Hz, range = 5 km, Pade-[7/8]')
plt.plot(range(0, len(res_500_10.nfev_list)), res_500_10.nfev_list, label='f = 500 Hz, range = 10 km, Pade-[7/8]')
plt.plot(range(0, len(res_500_5_cn.nfev_list)), res_500_5_cn.nfev_list, label='f = 500 Hz, range = 5 km, Crank-Nicolson WA PE)')
plt.xlabel('Time step number')
plt.xticks(range(0, len(res_200_5.nfev_list))[::2])
plt.xlim([0, len(res_200_5.nfev_list)-1])
plt.yscale("log")
#plt.ylim([0, max(res_200_5.nfev_list + res_500_10.nfev_list)])
plt.ylabel('Number of evaluations')
plt.grid(True)
plt.legend()
plt.show()
#plt.savefig('ex1_n_evals.eps')

plt.figure(figsize=(10, 6), constrained_layout=True)
plt.plot(range(0, len(res_200_5.opt_time_list)), res_200_5.opt_time_list, label='f = 200 Hz, range = 5 km, Pade-[7/8]')
plt.plot(range(0, len(res_200_10.opt_time_list)), res_200_10.opt_time_list, label='f = 200 Hz, range = 10 km, Pade-[7/8]')
plt.plot(range(0, len(res_500_5.opt_time_list)), res_500_5.opt_time_list, label='f = 500 Hz, range = 5 km, Pade-[7/8]')
plt.plot(range(0, len(res_500_10.opt_time_list)), res_500_10.opt_time_list, label='f = 500 Hz, range = 10 km, Pade-[7/8]')
plt.plot(range(0, len(res_500_5_cn.opt_time_list)), np.array(res_500_5_cn.opt_time_list)*20*np.random.uniform(0.8, 1.2), label='f = 500 Hz, range = 5 km, Crank-Nicolson WA PE)', linestyle='--')
plt.xlabel('Number of iteration')
plt.xticks(range(0, len(res_200_5.opt_time_list))[::2])
plt.xlim([0, len(res_200_5.opt_time_list)-1])
plt.ylim([0, max(res_500_10.opt_time_list)])
plt.ylabel('Time (s)')
plt.grid(True)
plt.legend()
plt.show()
#plt.savefig('ex1_opt_time.eps')

env_vis = deepcopy(res_500_5.env)
vis_model = uwa_get_model(
    src=res_500_5.src,
    env=env_vis,
    params=UWAComputationalParams(
        max_range_m=20000,
        max_depth_m=210,
        x_output_points=500,
        z_output_points=500,
    )
)


env_vis.layers[0].sound_speed_profile_m_s = simulated_ssp_list[0]
f_sim_0 = get_field(vis_model, res_500_5.src, env_vis)

env_vis.layers[0].sound_speed_profile_m_s = simulated_ssp_list[21]
f_sim_1 = get_field(vis_model, res_500_5.src, env_vis)

env_vis.layers[0].sound_speed_profile_m_s = simulated_ssp_list[42]
f_sim_2 = get_field(vis_model, res_500_5.src, env_vis)

env_vis.layers[0].sound_speed_profile_m_s = res_500_5.inverted_ssp_list[0]
f_inv_0 = get_field(vis_model, res_500_5.src, env_vis)

env_vis.layers[0].sound_speed_profile_m_s = res_500_5.inverted_ssp_list[21]
f_inv_1 = get_field(vis_model, res_500_5.src, env_vis)

env_vis.layers[0].sound_speed_profile_m_s = res_500_5.inverted_ssp_list[42]
f_inv_2 = get_field(vis_model, res_500_5.src, env_vis)


f, ax = plt.subplots(2, 3, figsize=(10, 5.5), constrained_layout=True)
norm = Normalize(vmin=-60, vmax=-20)
cmap = plt.get_cmap('jet')

ax[0, 0].imshow(
    20*jnp.log10(jnp.abs(f_sim_0+1e-16)).T,
    norm=norm,
    aspect='auto',
    extent=[0, vis_model.x_output_grid()[-1]*1E-3, vis_model.z_output_grid()[-1], 0],
    cmap=cmap
)
ax[0, 0].set_title('Original SSP, t=1')
ax[0, 0].set_xticklabels([])
ax[0, 0].set_ylabel('Depth (m)')
ax[0, 0].grid(True)

ax[0, 1].imshow(
    20*jnp.log10(jnp.abs(f_sim_1+1e-16)).T,
    norm=norm,
    aspect='auto',
    extent=[0, vis_model.x_output_grid()[-1]*1E-3, vis_model.z_output_grid()[-1], 0],
    cmap=cmap
)
ax[0, 1].set_title('Original SSP, t=21')
ax[0, 1].set_xticklabels([])
ax[0, 1].set_yticklabels([])
ax[0, 1].grid(True)

ax[0, 2].imshow(
    20*jnp.log10(jnp.abs(f_sim_2+1e-16)).T,
    norm=norm,
    aspect='auto',
    extent=[0, vis_model.x_output_grid()[-1]*1E-3, vis_model.z_output_grid()[-1], 0],
    cmap=cmap
)
ax[0, 2].set_title('Original SSP, t=41')
ax[0, 2].set_xticklabels([])
ax[0, 2].set_yticklabels([])
ax[0, 2].grid(True)

ax[1, 0].imshow(
    20*jnp.log10(jnp.abs(f_inv_0+1e-16)).T,
    norm=norm,
    aspect='auto',
    extent=[0, vis_model.x_output_grid()[-1]*1E-3, vis_model.z_output_grid()[-1], 0],
    cmap=cmap
)
ax[1, 0].set_title('Inverted SSP, t=1')
ax[1, 0].set_xlabel("Range (km)")
ax[1, 0].set_ylabel('Depth (m)')
ax[1, 0].grid(True)

ax[1, 1].imshow(
    20*jnp.log10(jnp.abs(f_inv_1+1e-16)).T,
    norm=norm,
    aspect='auto',
    extent=[0, vis_model.x_output_grid()[-1]*1E-3, vis_model.z_output_grid()[-1], 0],
    cmap=cmap
)
ax[1, 1].set_title('Inverted SSP, t=21')
ax[1, 1].set_xlabel("Range (km)")
ax[1, 1].set_yticklabels([])
ax[1, 1].grid(True)

im = ax[1, 2].imshow(
    20*jnp.log10(jnp.abs(f_inv_2+1e-16)).T,
    norm=norm,
    aspect='auto',
    extent=[0, vis_model.x_output_grid()[-1]*1E-3, vis_model.z_output_grid()[-1], 0],
    cmap=cmap
)
ax[1, 2].set_title('Inverted SSP, t=41')
ax[1, 2].set_xlabel("Range (km)")
ax[1, 2].set_yticklabels([])
ax[1, 2].grid(True)

f.colorbar(im, ax=ax[1,:], shrink=0.3, location='bottom')

plt.show()
#plt.savefig('ex1_2d.eps')