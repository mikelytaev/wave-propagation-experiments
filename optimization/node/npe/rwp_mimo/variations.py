import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def realistic_refractive_index(h, sigma0=5.0, H=1.0, L=0.2):
    white_noise = np.random.randn(len(h))
    sigma_h_profile = sigma0 * np.exp(-h / H)
    scaled_noise = white_noise * sigma_h_profile
    correlated_noise = gaussian_filter1d(scaled_noise, sigma=L)

    # Add noise to profile
    N_realistic = correlated_noise
    return N_realistic


t = np.linspace(0, 100, 101)
plt.plot(t)
t_v = realistic_refractive_index(t)
plt.plot(t_v)
plt.show()