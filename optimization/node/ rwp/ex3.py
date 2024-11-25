from common import *


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
inv_model_3ghz = RWPModel(params=ComputationalParams(
        max_range_m=5000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*11,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    src=GaussSourceModel(freq_hz=3E9, height_m=10.0, beam_width_deg=3.0)
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
inverted_profiles_freq3ghz, inversion_time_freq3ghz, nfev_list_freq3ghz = realtime(inv_model_3ghz, profiles, gamma=1e-3, snr=30)
inverted_profiles_freq5ghz, inversion_time_freq5ghz, nfev_list_freq5ghz = realtime(inv_model_5ghz, profiles, gamma=1e-3, snr=30)

errors_freq500mgh = get_rel_errors(profiles, inverted_profiles_freq500mhz, z_grid)
errors_freq1ghz = get_rel_errors(profiles, inverted_profiles_freq1ghz, z_grid)
errors_freq3ghz = get_rel_errors(profiles, inverted_profiles_freq3ghz, z_grid)
errors_freq5ghz = get_rel_errors(profiles, inverted_profiles_freq5ghz, z_grid)

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