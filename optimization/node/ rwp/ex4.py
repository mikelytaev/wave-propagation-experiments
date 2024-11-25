from common import *


inv_model_ver5 = RWPModel(params=ComputationalParams(
        max_range_m=5000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*11,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    src=GaussSourceModel(freq_hz=3E9, height_m=10.0, beam_width_deg=3.0)
)
inv_model_ver10 = RWPModel(params=ComputationalParams(
        max_range_m=10000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*11,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    src=GaussSourceModel(freq_hz=3E9, height_m=10.0, beam_width_deg=3.0)
)
inv_model_ver25 = RWPModel(params=ComputationalParams(
        max_range_m=25000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*11,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    src=GaussSourceModel(freq_hz=3E9, height_m=10.0, beam_width_deg=3.0)
)

inverted_profiles_ver5, _, _ = realtime(inv_model_ver5, profiles, gamma=1e-3, snr=30)
inverted_profiles_ver10, _, _ = realtime(inv_model_ver10, profiles, gamma=1e-3, snr=30)
inverted_profiles_ver25, _, _ = realtime(inv_model_ver25, profiles, gamma=1e-3, snr=30)

errors_ver5 = get_rel_errors(profiles, inverted_profiles_ver5, z_grid)
errors_ver10 = get_rel_errors(profiles, inverted_profiles_ver10, z_grid)
errors_ver25 = get_rel_errors(profiles, inverted_profiles_ver25, z_grid)


plt.figure(figsize=(6, 3.2))
plt.xlabel("Time step number")
plt.ylabel("Rel. error")
plt.plot(errors_ver5, label='Ver, R = 5 km')
plt.plot(errors_ver10, label='Ver, R = 10 km')
plt.plot(errors_ver25, label='Ver, R = 25 km')
plt.grid(True)
plt.xlim([0, len(profiles) - 1])
plt.ylim([0.0, 0.5])
plt.tight_layout()
plt.legend()
plt.show()