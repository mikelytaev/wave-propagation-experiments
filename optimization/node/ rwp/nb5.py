from common import *
from matplotlib.colors import Normalize
from matplotlib.pyplot import figure

from experimental.rwp_jax import PiecewiseLinearNProfileModel, EmptyNProfileModel
import jax.numpy as jnp
from utils import *
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


inv_model = RWPModel(params=ComputationalParams(
        max_range_m=5000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*11,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)



inverted_profiles, inversion_time, nfev_list = realtime(inv_model, profiles, gamma=1e-2)

inv_model.env.N_profile = profiles[35]
plt.plot(inv_model.env.M_profile(z_grid), z_grid)
inv_model.env.N_profile = inverted_profiles[35]
plt.plot(inv_model.env.M_profile(z_grid), z_grid)
plt.show()

f, ax = plt.subplots(2, 1, figsize=(10, 4), constrained_layout=True)
for i, true_profile in enumerate(profiles):
    inv_model.env.N_profile = true_profile
    ax[0].plot(inv_model.env.M_profile(z_grid) + 5*i, z_grid)
ax[0].set_title('Original M-profiles')
ax[0].set_xticklabels([])
ax[0].set_ylabel("Height (m)")
ax[0].set_xlim([325, 554])
ax[0].set_ylim([z_grid[0], z_grid[-1]])
ax[0].grid(True)

for i, inverted_profile in enumerate(inverted_profiles):
    inv_model.env.N_profile = inverted_profile
    ax[1].plot(inv_model.env.M_profile(z_grid) + 5*i, z_grid)
ax[1].set_title('Inverted M-profiles')
ax[1].set_ylabel("Height (m)")
ax[1].set_xlabel("M-profile (M units)")
ax[1].set_xlim([325, 554])
ax[1].set_ylim([z_grid[0], z_grid[-1]])
ax[1].grid(True)
plt.show()

f, ax = plt.subplots(1, 4, figsize=(10, 3.2), constrained_layout=True)
inds = [1, 15, 25, 35]
ax[0].set_ylabel("Height (m)")
for i in range(4):
    ax[i].set_title(f't = {inds[i]}')
    inv_model.env.N_profile = profiles[inds[i]]
    ax[i].plot(inv_model.env.M_profile(z_grid), z_grid)
    inv_model.env.N_profile = inverted_profiles[inds[i]]
    ax[i].plot(inv_model.env.M_profile(z_grid), z_grid)
    ax[i].set_ylim([z_grid[0], z_grid[-1]])
    ax[i].set_xlabel("M-profile (M units)")
    ax[i].grid(True)
plt.show()

plot_rel_error(profiles, inverted_profiles, z_grid)

vis_model = RWPModel(params=ComputationalParams(
        max_range_m=50000,
        max_height_m=250,
        dx_m=100,
        dz_m=0.5
    ))
f, ax = plt.subplots(1, 3, figsize=(10, 2.2), constrained_layout=True)
f1 = vis_model.calc_field(profiles[10])
extent = (vis_model.fwd_model.x_output_grid()[0]*1e-3, vis_model.fwd_model.x_output_grid()[-1]*1e-3,
          vis_model.fwd_model.z_output_grid()[0], vis_model.fwd_model.z_output_grid()[-1])
norm = Normalize(-60, -10)
ax[0].imshow(20*jnp.log10(jnp.abs(f1+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
ax[0].grid(True)
ax[0].set_xlabel('Range (km)')
ax[0].set_ylabel('Height (km)')

f1_i = vis_model.calc_field(profiles[10])
norm = Normalize(-60, -10)
ax[1].imshow(20*jnp.log10(jnp.abs(f1+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
ax[1].grid(True)
ax[1].set_xlabel('Range (km)')
ax[1].set_yticklabels([])

f2 = vis_model.calc_field(EmptyNProfileModel())
norm = Normalize(-60, -10)
ax[2].imshow(20*jnp.log10(jnp.abs(f2+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('jet'))
ax[2].grid(True)
ax[2].set_xlabel('Range (km)')
ax[2].set_yticklabels([])
plt.show()

plt.figure(figsize=(6, 3.2))
extent = (vis_model.fwd_model.x_output_grid()[0], vis_model.fwd_model.x_output_grid()[-1],
          vis_model.fwd_model.z_output_grid()[0], vis_model.fwd_model.z_output_grid()[-1])
norm = Normalize(-50, 0)
plt.imshow(20*jnp.log10(jnp.abs(f1-f2+1e-16)).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('binary'))
plt.grid(True)
plt.xlabel('Range (km)')
plt.show()

plt.figure(figsize=(6, 3.2))
extent = (vis_model.fwd_model.x_output_grid()[0], vis_model.fwd_model.x_output_grid()[-1],
          vis_model.fwd_model.z_output_grid()[0], vis_model.fwd_model.z_output_grid()[-1])
norm = Normalize(0, 5)
plt.imshow(abs(20*jnp.log10(jnp.abs(f1+1e-16)) - 20*jnp.log10(jnp.abs(f2+1e-16))).T[::-1,:], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('binary'))
plt.grid(True)
plt.xlabel('Range (km)')
plt.show()

r = 5000
r_i = abs(vis_model.fwd_model.x_output_grid() - r).argmin()

plt.figure(figsize=(6, 3.2))
plt.plot(20*jnp.log10(jnp.abs(f1[r_i,:]+1e-16)), vis_model.fwd_model.z_output_grid())
plt.plot(20*jnp.log10(jnp.abs(f2[r_i,:]+1e-16)), vis_model.fwd_model.z_output_grid())
plt.show()

plt.figure(figsize=(6, 3.2))
plt.plot(jnp.unwrap(jnp.angle(f1[r_i,:]+1e-16)), vis_model.fwd_model.z_output_grid())
plt.plot(jnp.unwrap(jnp.angle(f2[r_i,:]+1e-16)), vis_model.fwd_model.z_output_grid())
plt.show()

# inverted_profiles_snr10, inversion_time_snr10, nfev_list_snr10 = realtime(inv_model, profiles, gamma=1e-2, snr=10)
# inverted_profiles_snr10_2, inversion_time_snr10_2, nfev_list_snr10_2 = realtime(inv_model, profiles, gamma=1e-2, snr=10)
# inverted_profiles_snr20, inversion_time_snr20, nfev_list_snr20 = realtime(inv_model, profiles, gamma=1e-3, snr=20)
# inverted_profiles_snr30, inversion_time_snr30, nfev_list_snr30 = realtime(inv_model, profiles, gamma=1e-3, snr=30)
# inverted_profiles_snr40, inversion_time_snr40, nfev_list_snr40 = realtime(inv_model, profiles, gamma=1e-3, snr=40)
# inverted_profiles_snr50, inversion_time_snr50, nfev_list_snr50 = realtime(inv_model, profiles, gamma=1e-3, snr=50)
#
# errors_snr10 = []
# errors_snr10_2 = []
# errors_snr20 = []
# errors_snr30 = []
# errors_snr40 = []
# errors_snr50 = []
# for ind in range(0, len(profiles)):
#     errors_snr10 += [jnp.linalg.norm((profiles[ind](z_grid)) - (inverted_profiles_snr10[ind](z_grid))) / jnp.linalg.norm(
#         (profiles[ind](z_grid)))]
#     errors_snr10_2 += [
#         jnp.linalg.norm((profiles[ind](z_grid)) - (inverted_profiles_snr10_2[ind](z_grid))) / jnp.linalg.norm(
#             (profiles[ind](z_grid)))]
#     errors_snr20 += [
#         jnp.linalg.norm((profiles[ind](z_grid)) - (inverted_profiles_snr20[ind](z_grid))) / jnp.linalg.norm(
#             (profiles[ind](z_grid)))]
#     errors_snr30 += [
#         jnp.linalg.norm((profiles[ind](z_grid)) - (inverted_profiles_snr30[ind](z_grid))) / jnp.linalg.norm(
#             (profiles[ind](z_grid)))]
#     errors_snr40 += [
#         jnp.linalg.norm((profiles[ind](z_grid)) - (inverted_profiles_snr40[ind](z_grid))) / jnp.linalg.norm(
#             (profiles[ind](z_grid)))]
#     errors_snr50 += [
#         jnp.linalg.norm((profiles[ind](z_grid)) - (inverted_profiles_snr50[ind](z_grid))) / jnp.linalg.norm(
#             (profiles[ind](z_grid)))]
#
# plt.figure(figsize=(6, 3.2))
# plt.xlabel("Time step number")
# plt.ylabel("Rel. error")
# plt.plot(errors_snr10_2, label='SNR = 10 dB')
# plt.plot(errors_snr20, label='SNR = 20 dB')
# plt.plot(errors_snr30, label='SNR = 30 dB')
# plt.plot(errors_snr40, label='SNR = 40 dB')
# plt.plot(errors_snr50, label='SNR = 50 dB')
# plt.grid(True)
# plt.xlim([0, len(profiles) - 1])
# plt.ylim([0.0, 1])
# plt.tight_layout()
# plt.legend()
# plt.show()

inv_model_500mhz = RWPModel(params=ComputationalParams(
        max_range_m=5000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*11,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    src=GaussSourceModel(freq_hz=500e6, height_m=10.0, beam_width_deg=3.0)
)
inv_model_1ghz = RWPModel(params=ComputationalParams(
        max_range_m=5000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*11,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    src=GaussSourceModel(freq_hz=1E9, height_m=10.0, beam_width_deg=3.0)
)
inv_model_5ghz = RWPModel(params=ComputationalParams(
        max_range_m=5000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*11,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    src=GaussSourceModel(freq_hz=5E9, height_m=10.0, beam_width_deg=3.0)
)
inverted_profiles_freq500mhz, inversion_time_freq500mhz, nfev_list_freq500mhz = realtime(inv_model_500mhz, profiles, gamma=1e-3, snr=30)
inverted_profiles_freq1ghz, inversion_time_freq1ghz, nfev_list_freq1ghz = realtime(inv_model_1ghz, profiles, gamma=1e-3, snr=30)
inverted_profiles_freq3ghz, inversion_time_freq3ghz, nfev_list_freq3ghz = inverted_profiles, inversion_time, nfev_list #realtime(inv_model, profiles, gamma=1e-3, snr=40)
inverted_profiles_freq5ghz, inversion_time_freq5ghz, nfev_list_freq5ghz = realtime(inv_model_5ghz, profiles, gamma=1e-3, snr=30)

errors_freq500mgh = []
errors_freq1ghz = []
errors_freq3ghz = []
errors_freq5ghz = []
for ind in range(0, len(profiles)):
    errors_freq500mgh += [
        jnp.linalg.norm((profiles[ind](z_grid)) - (inverted_profiles_freq500mhz[ind](z_grid))) / jnp.linalg.norm(
            (profiles[ind](z_grid)))]
    errors_freq1ghz += [jnp.linalg.norm((profiles[ind](z_grid)) - (inverted_profiles_freq1ghz[ind](z_grid))) / jnp.linalg.norm(
        (profiles[ind](z_grid)))]
    errors_freq3ghz += [
        jnp.linalg.norm((profiles[ind](z_grid)) - (inverted_profiles_freq3ghz[ind](z_grid))) / jnp.linalg.norm(
            (profiles[ind](z_grid)))]
    errors_freq5ghz += [
        jnp.linalg.norm((profiles[ind](z_grid)) - (inverted_profiles_freq5ghz[ind](z_grid))) / jnp.linalg.norm(
            (profiles[ind](z_grid)))]

plt.figure(figsize=(6, 3.2))
plt.xlabel("Time step number")
plt.ylabel("Rel. error")
plt.plot(errors_freq500mgh, label='500 MHz')
plt.plot(errors_freq1ghz, label='1 GHz')
plt.plot(errors_freq3ghz, label='3 GHz')
plt.plot(errors_freq5ghz, label='5 GHz')
plt.grid(True)
plt.xlim([0, len(profiles) - 1])
plt.ylim([0.0, 0.5])
plt.tight_layout()
plt.legend()
plt.show()