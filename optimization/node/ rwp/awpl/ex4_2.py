from common import *
from matplotlib.lines import Line2D


inv_model_ver5_6 = RWPModel(params=RWPComputationalParams(
        max_range_m=25000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*6,
    measure_points_z=[50, 60, 70, 80, 90, 99],
    src=RWPGaussSourceModel(freq_hz=3E9, height_m=10.0, beam_width_deg=3.0)
)
inv_model_ver5_hor = RWPModel(params=RWPComputationalParams(
        max_range_m=26000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11],
    measure_points_z=[70]*11,
    src=RWPGaussSourceModel(freq_hz=3E9, height_m=10.0, beam_width_deg=3.0)
)

inverted_profiles_ver5_6, _, _ = realtime(inv_model_ver5_6, profiles, gamma=1e-3, snr=30)
inverted_profiles_ver5_hor, _, _ = realtime(inv_model_ver5_hor, profiles, gamma=1e-3, snr=30)

errors_ver5_6 = get_rel_errors(profiles, inverted_profiles_ver5_6, z_grid)
errors_ver5_hor = get_rel_errors(profiles, inverted_profiles_ver5_hor, z_grid)


plt.figure(figsize=(6, 3.2))
plt.xlabel("Time step number")
plt.ylabel("Rel. error")
plt.plot(errors_ver5_6, label='Ver (6 elem.), R = 5 km')
plt.plot(errors_ver5_hor, label='Hor, R = 5 km')
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
    inv_model.env.N_profile = inverted_profiles_ver5_hor[inds[i] - 1]
    ax[i].plot(inv_model.env.M_profile(z_grid), z_grid, color='blue')
    inv_model.env.N_profile = inverted_profiles_ver5_6[inds[i] - 1]
    ax[i].plot(inv_model.env.M_profile(z_grid), z_grid, '--', color='red')
    ax[i].set_xlim([330, 358])
    ax[i].set_ylim([z_grid[0], z_grid[-1]])
    ax[i].set_xlabel("M-profile")
    ax[i].grid(True)
for i in range(1, 4):
    ax[i].set_yticklabels([])
legend_elements = [Line2D([0], [0], color='blue', lw=1, label='Ver, R = 5 km'),
                   Line2D([0], [0], color='red', lw=1, label='Inverted profile')]
f.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.00), ncol=6)
f.tight_layout()
plt.show()