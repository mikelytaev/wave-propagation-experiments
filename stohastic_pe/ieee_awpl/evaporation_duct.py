import logging

import matplotlib.pyplot as plt
import numpy as np

from rwp.environment import *
from rwp.sspade import *
from rwp.vis import *
from duct_plot import plot_profile_prop_density

environment = Troposphere()
environment.terrain = Terrain(ground_material=SaltWater())

#logging.basicConfig(level=logging.DEBUG)


def do_model(freq_hz, antenna_height_m, duct_expected_height_m, duct_mean_height_m, max_range_m, max_height_m, n=10):
    antenna = GaussAntenna(freq_hz=freq_hz, height=antenna_height_m, beam_width=2, elevation_angle=0, polarz='H')

    params = RWPSSpadeComputationalParams(
        max_range_m=max_range_m,
        max_height_m=max_height_m,
        dx_m = 100, # output grid steps affects only on the resulting field, NOT the computational grid
        dz_m = 1,
        #storage=PickleStorage()
    )

    expected_height = duct_expected_height_m
    heights = np.random.normal(expected_height, duct_mean_height_m, n)
    environment.M_profile = lambda x, z: evaporation_duct(height=expected_height, z_grid_m=z) * (z < 95) + \
                                         evaporation_duct(height=expected_height, z_grid_m=95) * (z >= 95)
    mid_field = rwp_ss_pade(antenna=antenna, env=environment, params=params)
    expected_field = Field(mid_field.x_grid, mid_field.z_grid, freq_hz=antenna.freq_hz)
    expected_field.log10 = True
    error = Field(mid_field.x_grid, mid_field.z_grid, freq_hz=antenna.freq_hz)
    expected_error = Field(mid_field.x_grid, mid_field.z_grid, freq_hz=antenna.freq_hz)

    prev_norm = 0
    for h_i, height in enumerate(heights):
        environment.M_profile = lambda x, z: evaporation_duct(height=height, z_grid_m=z) * (z < 95) + \
                                             evaporation_duct(height=height, z_grid_m=95) * (z >= 95)
        field = rwp_ss_pade(antenna=antenna, env=environment, params=params)
        expected_field.field = (expected_field.field*h_i + 20*np.log10(np.abs(field.field+1e-16))) / (h_i+1)
        norm = np.linalg.norm(expected_field.field)
        print(f'{round(h_i/n, 1)} {round(height, 3)} {20*fm.log10(abs(norm-prev_norm)/norm)}')
        prev_norm = norm

    for h_i, height in enumerate(heights):
        environment.M_profile = lambda x, z: evaporation_duct(height=height, z_grid_m=z) * (z < 95) + \
                                             evaporation_duct(height=height, z_grid_m=95) * (z >= 95)
        field = rwp_ss_pade(antenna=antenna, env=environment, params=params)
        print(f'{round(h_i/n, 1)} {round(height, 3)}')
        error.field += (20*np.log10(np.abs(field.field+1e-16)) - expected_field.field)**2

    expected_error.field = 20*np.log10(np.abs(mid_field.field+1e-16)) - expected_field.field
    error.field = np.sqrt(error.field/n)

    vis = FieldVisualiser(mid_field, env=environment, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), label='Pade + Transparent BC', x_mult=1E-3)
    expected_vis = FieldVisualiser(expected_field, env=environment, trans_func=lambda v: v,
                                   label='Pade + Transparent BC', x_mult=1E-3)

    vis_error = FieldVisualiser(error, env=environment, trans_func=lambda v: v, label='СКО', x_mult=1E-3)
    vis_expected_error = FieldVisualiser(expected_error, env=environment, trans_func=lambda v: v, label='СКО', x_mult=1E-3)

    return vis, expected_vis, vis_expected_error, vis_error

def show(vis, expected_vis, vis_expected_error, vis_error):
    plt.rcParams['font.size'] = '13'
    f, ax = plt.subplots(1, 4, figsize=(9, 2.5), constrained_layout=True)
    # ax = [a[0, 0], a[0, 1], a[1, 0], a[1, 1]]
    norm = Normalize(-80, 0)
    extent = [vis.x_grid[0], vis.x_grid[-1], vis.z_grid[0], vis.z_grid[-1]]
    im = ax[0].imshow(vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
    f.colorbar(im, ax=ax[0], fraction=0.046, location='bottom')
    ax[0].set_xlabel("Range (km)", fontsize=13)
    ax[0].set_ylabel("Height (m)", fontsize=13)
    ax[0].set_title("L, dB", fontsize=13)
    ax[0].grid(True)

    norm = Normalize(-80, 0)
    extent = [expected_vis.x_grid[0], expected_vis.x_grid[-1], expected_vis.z_grid[0], expected_vis.z_grid[-1]]
    im = ax[1].imshow(expected_vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
    f.colorbar(im, ax=ax[1], fraction=0.046, location='bottom')
    ax[1].set_xlabel("Range (km)", fontsize=13)
    ax[1].set_title("E[L], dB", fontsize=13)
    ax[1].set_yticklabels([])
    ax[1].grid(True)

    norm = Normalize(-20, 20)
    extent = [vis_error.x_grid[0], vis_error.x_grid[-1], vis_error.z_grid[0], vis_error.z_grid[-1]]
    im = ax[2].imshow(vis_expected_error.field.T[::-1, :], extent=extent, norm=norm, aspect='auto',
                      cmap=plt.get_cmap('seismic'))
    f.colorbar(im, ax=ax[2], fraction=0.046, location='bottom')
    ax[2].set_xlabel("Range (km)", fontsize=13)
    ax[2].set_yticklabels([])
    # ax[2].set_ylabel("Height (m)")
    ax[2].set_title("L - E[L]")
    ax[2].grid(True)

    norm = Normalize(0, 20)
    extent = [vis_error.x_grid[0], vis_error.x_grid[-1], vis_error.z_grid[0], vis_error.z_grid[-1]]
    im = ax[3].imshow(vis_error.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
    f.colorbar(im, ax=ax[3], fraction=0.046, location='bottom')
    ax[3].set_xlabel("Range (km)", fontsize=13)
    ax[3].set_title("SD")
    ax[3].set_yticklabels([])
    ax[3].grid(True)

    for a in ax[:]:
        for label in (a.get_xticklabels() + a.get_yticklabels()):
            label.set_fontsize(13)


if __name__ == "__main__":
    ##### 1 Варьируем СКО высоты В.И.№№№№№№№№№№№№
    show(*do_model(
        freq_hz=10E9,
        antenna_height_m=10,
        duct_expected_height_m=20,
        duct_mean_height_m=0.5,
        max_range_m=300E3,
        max_height_m=100,
        n=100
    ))
    plt.savefig('1.1.eps')

    show(*do_model(
        freq_hz=10E9,
        antenna_height_m=10,
        duct_expected_height_m=20,
        duct_mean_height_m=1,
        max_range_m=300E3,
        max_height_m=100,
        n=100
    ))
    plt.savefig('1.2.eps')

    show(*do_model(
        freq_hz=10E9,
        antenna_height_m=10,
        duct_expected_height_m=20,
        duct_mean_height_m=2,
        max_range_m=300E3,
        max_height_m=100,
        n=100
    ))
    plt.savefig('1.3.eps')

    show(*do_model(
        freq_hz=10E9,
        antenna_height_m=10,
        duct_expected_height_m=20,
        duct_mean_height_m=3,
        max_range_m=300E3,
        max_height_m=100,
        n=100
    ))
    plt.savefig('1.4.eps')

    show(*do_model(
        freq_hz=10E9,
        antenna_height_m=10,
        duct_expected_height_m=20,
        duct_mean_height_m=4,
        max_range_m=300E3,
        max_height_m=100,
        n=100
    ))
    plt.savefig('1.5.eps')

    #### 2 варьируем частоту ####

    show(*do_model(
        freq_hz=300E6,
        antenna_height_m=10,
        duct_expected_height_m=20,
        duct_mean_height_m=2,
        max_range_m=300E3,
        max_height_m=100,
        n=100
    ))
    plt.savefig('2.1.eps')

    show(*do_model(
        freq_hz=1E9,
        antenna_height_m=10,
        duct_expected_height_m=20,
        duct_mean_height_m=2,
        max_range_m=300E3,
        max_height_m=100,
        n=100
    ))
    plt.savefig('2.2.eps')

    show(*do_model(
        freq_hz=3E9,
        antenna_height_m=10,
        duct_expected_height_m=20,
        duct_mean_height_m=2,
        max_range_m=300E3,
        max_height_m=100,
        n=100
    ))
    plt.savefig('2.3.eps')

    #### 3 варьируем высоту В.И. ####

    show(*do_model(
        freq_hz=10E9,
        antenna_height_m=10,
        duct_expected_height_m=10,
        duct_mean_height_m=2,
        max_range_m=300E3,
        max_height_m=100,
        n=100
    ))
    plt.savefig('3.1.eps')

    show(*do_model(
        freq_hz=10E9,
        antenna_height_m=10,
        duct_expected_height_m=20,
        duct_mean_height_m=2,
        max_range_m=300E3,
        max_height_m=100,
        n=100
    ))
    plt.savefig('3.2.eps')

    show(*do_model(
        freq_hz=10E9,
        antenna_height_m=10,
        duct_expected_height_m=30,
        duct_mean_height_m=2,
        max_range_m=300E3,
        max_height_m=100,
        n=100
    ))
    plt.savefig('3.3.eps')

    #### 4 варьируем высоту В.И. ####

    show(*do_model(
        freq_hz=10E9,
        antenna_height_m=5,
        duct_expected_height_m=20,
        duct_mean_height_m=2,
        max_range_m=300E3,
        max_height_m=100,
        n=100
    ))
    plt.savefig('4.1.eps')

    show(*do_model(
        freq_hz=10E9,
        antenna_height_m=30,
        duct_expected_height_m=20,
        duct_mean_height_m=2,
        max_range_m=300E3,
        max_height_m=100,
        n=100
    ))
    plt.savefig('4.2.eps')

    show(*do_model(
        freq_hz=10E9,
        antenna_height_m=60,
        duct_expected_height_m=20,
        duct_mean_height_m=2,
        max_range_m=300E3,
        max_height_m=100,
        n=100
    ))
    plt.savefig('4.3.eps')

    # do_model(
    #     freq_hz=10E9,
    #     antenna_height_m=20,
    #     duct_expected_height_m=20,
    #     duct_mean_height_m=5,
    #     max_range_m=300E3,
    #     max_height_m=100
    # )