from utils import *
from common import *


inverted_profiles_snr10_2, inversion_time_snr10_2, nfev_list_snr10_2 = realtime(inv_model, profiles, gamma=1e-2, snr=10)
inverted_profiles_snr20, inversion_time_snr20, nfev_list_snr20 = realtime(inv_model, profiles, gamma=1e-3, snr=20)
inverted_profiles_snr30, inversion_time_snr30, nfev_list_snr30 = realtime(inv_model, profiles, gamma=1e-3, snr=30)
inverted_profiles_snr40, inversion_time_snr40, nfev_list_snr40 = realtime(inv_model, profiles, gamma=1e-3, snr=40)
inverted_profiles_snr50, inversion_time_snr50, nfev_list_snr50 = realtime(inv_model, profiles, gamma=1e-3, snr=50)

errors_snr10_2 = get_rel_errors(profiles, inverted_profiles_snr10_2, z_grid)
errors_snr20 = get_rel_errors(profiles, inverted_profiles_snr20, z_grid)
errors_snr30 = get_rel_errors(profiles, inverted_profiles_snr30, z_grid)
errors_snr40 = get_rel_errors(profiles, inverted_profiles_snr40, z_grid)
errors_snr50 = get_rel_errors(profiles, inverted_profiles_snr50, z_grid)

plt.figure(figsize=(6, 3.2))
plt.xlabel("Time step number")
plt.ylabel("Rel. error")
plt.plot(errors_snr10_2, label='SNR = 10 dB')
plt.plot(errors_snr20, label='SNR = 20 dB')
plt.plot(errors_snr30, label='SNR = 30 dB')
plt.plot(errors_snr40, label='SNR = 40 dB')
plt.plot(errors_snr50, label='SNR = 50 dB')
plt.grid(True)
plt.xlim([0, len(profiles) - 1])
plt.ylim([0.0, 1])
plt.tight_layout()
plt.legend()
plt.show()