from common import *
from matplotlib.lines import Line2D


inv_model_ver5 = RWPModel(params=RWPComputationalParams(
        max_range_m=5000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*11,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    src=RWPGaussSourceModel(freq_hz=3E9, height_m=10.0, beam_width_deg=3.0)
)
inv_model_ver5_6 = RWPModel(params=RWPComputationalParams(
        max_range_m=5050,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*6,
    measure_points_z=[2, 10, 20, 30, 40, 50],
    src=RWPGaussSourceModel(freq_hz=3E9, height_m=10.0, beam_width_deg=3.0)
)
inv_model_ver5_hor = RWPModel(params=RWPComputationalParams(
        max_range_m=5100,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11],
    measure_points_z=[10]*11,
    src=RWPGaussSourceModel(freq_hz=3E9, height_m=10.0, beam_width_deg=3.0)
)
inv_model_ver10 = RWPModel(params=RWPComputationalParams(
        max_range_m=10000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*11,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    src=RWPGaussSourceModel(freq_hz=3E9, height_m=10.0, beam_width_deg=3.0)
)
inv_model_ver25 = RWPModel(params=RWPComputationalParams(
        max_range_m=25000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*11,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    src=RWPGaussSourceModel(freq_hz=3E9, height_m=10.0, beam_width_deg=3.0)
)

inverted_profiles_ver5, _, _, loss_ver5 = realtime(inv_model_ver5, profiles, gamma=1e-3, snr=30)
inverted_profiles_ver5_6, _, _, loss_ver5_6 = realtime(inv_model_ver5_6, profiles, gamma=1e-3, snr=30)
inverted_profiles_ver5_hor, _, _, loss_ver5_hor = realtime(inv_model_ver5_hor, profiles, gamma=1e-3, snr=30)
inverted_profiles_ver10, _, _, loss_ver10 = realtime(inv_model_ver10, profiles, gamma=1e-3, snr=30)
inverted_profiles_ver25, _, _, loss_ver25 = realtime(inv_model_ver25, profiles, gamma=1e-3, snr=30)

errors_ver5 = get_rel_errors(profiles, inverted_profiles_ver5, z_grid)
errors_ver5_6 = get_rel_errors(profiles, inverted_profiles_ver5_6, z_grid)
errors_ver5_hor = get_rel_errors(profiles, inverted_profiles_ver5_hor, z_grid)
errors_ver10 = get_rel_errors(profiles, inverted_profiles_ver10, z_grid)
errors_ver25 = get_rel_errors(profiles, inverted_profiles_ver25, z_grid)


plt.figure(figsize=(6, 2.8))
plt.xlabel("Time step number")
plt.ylabel("Rel. error")
plt.plot(errors_ver5, label='Ver 11 elem., R = 5 km', color='blue')
plt.plot(errors_ver10, label='Ver 11 elem., R = 10 km', linestyle='-', color='green')
plt.plot(errors_ver25, label='Ver 11 elem., R = 25 km', linestyle='-', color='red')
plt.plot(errors_ver5_hor, label='Hor 11 elem., R = 5 km', linestyle='--', color='blue')
plt.plot(errors_ver5_6, label='Ver 6 elem., R = 5 km', linestyle='--', color='red')
plt.grid(True)
plt.xlim([0, len(profiles) - 1])
plt.ylim([0.0, 0.5])
plt.tight_layout()
plt.legend()
plt.show()

f, ax = plt.subplots(1, 4, figsize=(6, 3.2), constrained_layout=True)
inds = [10, 20, 30, 40]
ax[0].set_ylabel("Height (m)")
for i in range(4):
    ax[i].set_title(f't = {inds[i]}')
    inv_model.env.N_profile = profiles[inds[i]-1]
    ax[i].plot(inv_model.env.M_profile(z_grid), z_grid, color='black')
    inv_model.env.N_profile = inverted_profiles_ver5[inds[i]-1]
    ax[i].plot(inv_model.env.M_profile(z_grid), z_grid, color='blue')
    inv_model.env.N_profile = inverted_profiles_ver10[inds[i] - 1]
    ax[i].plot(inv_model.env.M_profile(z_grid), z_grid, color='green')
    inv_model.env.N_profile = inverted_profiles_ver25[inds[i] - 1]
    ax[i].plot(inv_model.env.M_profile(z_grid), z_grid, '-', color='red')
    inv_model.env.N_profile = inverted_profiles_ver5_hor[inds[i] - 1]
    ax[i].plot(inv_model.env.M_profile(z_grid), z_grid, '--', color='blue')
    inv_model.env.N_profile = inverted_profiles_ver5_6[inds[i] - 1]
    ax[i].plot(inv_model.env.M_profile(z_grid), z_grid, '--', color='red')
    ax[i].set_xlim([330, 358])
    ax[i].set_ylim([z_grid[0], z_grid[-1]])
    ax[i].set_xlabel("M-profile")
    ax[i].grid(True)
for i in range(1, 4):
    ax[i].set_yticklabels([])
legend_elements = [Line2D([0], [0], color='black', lw=1, label='True profile'),
                   Line2D([0], [0], color='blue', lw=1, label='Ver 11 elem., 5 km'),
                   Line2D([0], [0], color='green', lw=1, label='Ver 11 elem., 10 km'),
                   Line2D([0], [0],  linestyle='-', color='red', lw=1, label='Ver 11 elem., 25 km'),
                   Line2D([0], [0],  linestyle='--', color='blue', lw=1, label='Hor 11 elem., 5 km'),
                   Line2D([0], [0],  linestyle='--', color='red', lw=1, label='Ver 6 elem., 5 km'),
                   ]
f.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.00), ncol=3)
f.tight_layout()
plt.show()


plt.figure(figsize=(6, 2.8))
plt.xlabel("Time step number")
plt.ylabel("Rel. error")
plt.plot(loss_ver5, label='Ver 11 elem., R = 5 km', color='blue')
plt.plot(loss_ver10, label='Ver 11 elem., R = 10 km', linestyle='-', color='green')
plt.plot(loss_ver25, label='Ver 11 elem., R = 25 km', linestyle='-', color='red')
plt.plot(loss_ver5_hor, label='Hor 11 elem., R = 5 km', linestyle='--', color='blue')
plt.plot(loss_ver5_6, label='Ver 6 elem., R = 5 km', linestyle='--', color='red')
plt.grid(True)
plt.xlim([0, len(profiles) - 1])
plt.ylim([0.0, 0.5])
plt.tight_layout()
plt.legend()
plt.show()