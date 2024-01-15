import matplotlib.pyplot as plt

from rwp.sspade import *
from rwp.vis import *
from scipy.stats import norm


def show(random_field: RandomField, mean_index_field: Field):
    plt.rcParams['font.size'] = '13'
    f, ax = plt.subplots(1, 4, figsize=(9, 2.5), constrained_layout=True)
    # ax = [a[0, 0], a[0, 1], a[1, 0], a[1, 1]]
    norm = Normalize(100, 200)
    extent = [mean_index_field.x_grid[0] * 1E-3, mean_index_field.x_grid[-1] * 1E-3, mean_index_field.z_grid[0], mean_index_field.z_grid[-1]]
    im = ax[0].imshow(mean_index_field.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
    f.colorbar(im, ax=ax[0], fraction=0.046, location='bottom')
    ax[0].set_xlabel("Range (km)", fontsize=13)
    ax[0].set_ylabel("Height (m)", fontsize=13)
    ax[0].set_title("L, dB", fontsize=13)
    ax[0].grid(True)

    mean_field = random_field.mean()
    norm = Normalize(100, 200)
    extent = [mean_field.x_grid[0] * 1E-3, mean_field.x_grid[-1] * 1E-3, mean_field.z_grid[0], mean_field.z_grid[-1]]
    im = ax[1].imshow(mean_field.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
    f.colorbar(im, ax=ax[1], fraction=0.046, location='bottom')
    ax[1].set_xlabel("Range (km)", fontsize=13)
    ax[1].set_title("E[L], dB", fontsize=13)
    ax[1].set_yticklabels([])
    ax[1].grid(True)

    norm = Normalize(-20, 20)
    extent = [mean_field.x_grid[0] * 1E-3, mean_field.x_grid[-1] * 1E-3, mean_field.z_grid[0], mean_field.z_grid[-1]]
    im = ax[2].imshow(mean_index_field.field.T[::-1, :] - mean_field.field.T[::-1, :], extent=extent, norm=norm, aspect='auto',
                      cmap=plt.get_cmap('seismic'))
    f.colorbar(im, ax=ax[2], fraction=0.046, location='bottom')
    ax[2].set_xlabel("Range (km)", fontsize=13)
    ax[2].set_yticklabels([])
    # ax[2].set_ylabel("Height (m)")
    ax[2].set_title("L - E[L]")
    ax[2].grid(True)

    sd_field = random_field.sd()
    norm = Normalize(0, 20)
    extent = [sd_field.x_grid[0] * 1E-3, sd_field.x_grid[-1] * 1E-3, sd_field.z_grid[0], sd_field.z_grid[-1]]
    im = ax[3].imshow(sd_field.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
    f.colorbar(im, ax=ax[3], fraction=0.046, location='bottom')
    ax[3].set_xlabel("Range (km)", fontsize=13)
    ax[3].set_title("SD")
    ax[3].set_yticklabels([])
    ax[3].grid(True)

    for a in ax[:]:
        for label in (a.get_xticklabels() + a.get_yticklabels()):
            label.set_fontsize(13)

def calc(M_profile, range_m, height_m, freq_hz, ant_height, file_name, max_monte_carlo_iterations=100):
    environment = Troposphere()
    environment.terrain = Terrain(ground_material=SaltWater())

    environment.M_profile = M_profile

    antenna = GaussAntenna(
        freq_hz=freq_hz,
        height=ant_height,
        beam_width=2,
        elevation_angle=0,
        polarz='H'
    )

    params = RWPSSpadeComputationalParams(
        max_range_m=range_m,
        max_height_m=height_m,
        dx_m = 100, # output grid steps affects only on the resulting field, NOT the computational grid
        dz_m = 1,
        max_monte_carlo_iterations=max_monte_carlo_iterations
    )


    random_field = rwp_ss_pade_r(antenna=antenna, env=environment, params=params).path_loss()
    mean_index_field = rwp_ss_pade(antenna=antenna, env=environment, params=params).path_loss()

    show(random_field, mean_index_field)
    plt.savefig(file_name)