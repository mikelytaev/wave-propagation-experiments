import logging

import numpy as np

from rwp.environment import *
from rwp.sspade import *
from rwp.vis import *
from duct_plot import plot_profile_prop_density

environment = Troposphere()
environment.terrain = Terrain(ground_material=SaltWater())

#logging.basicConfig(level=logging.INFO)


def do_model(freq_hz, antenna_height_m, duct_expected_height_m, duct_mean_height_m, max_range_m, max_height_m, n=10):
    antenna = GaussAntenna(freq_hz=freq_hz, height=antenna_height_m, beam_width=2, elevation_angle=0, polarz='H')

    params = RWPSSpadeComputationalParams(
        max_range_m=max_range_m,
        max_height_m=max_height_m,
        dx_m = 100, # output grid steps affects only on the resulting field, NOT the computational grid
        dz_m = 1
    )

    expected_height = duct_expected_height_m
    heights = np.random.normal(expected_height, duct_mean_height_m, n)
    environment.M_profile = lambda x, z: evaporation_duct(height=expected_height, z_grid_m=z)
    mid_field = rwp_ss_pade(antenna=antenna, env=environment, params=params)
    expected_field = Field(mid_field.x_grid, mid_field.z_grid, freq_hz=antenna.freq_hz)
    expected_field.log10 = True
    error = Field(mid_field.x_grid, mid_field.z_grid, freq_hz=antenna.freq_hz)
    expected_error = Field(mid_field.x_grid, mid_field.z_grid, freq_hz=antenna.freq_hz)

    for h_i, height in enumerate(heights):
        print(f'{h_i/n} {height}')
        environment.M_profile = lambda x, z: evaporation_duct(height=height, z_grid_m=z)
        field = rwp_ss_pade(antenna=antenna, env=environment, params=params)
        expected_field.field += 10*np.log10(np.abs(field.field+1e-16))
        error.field += (10*np.log10(np.abs(field.field+1e-16)) - 10*np.log10(np.abs(mid_field.field+1e-16)))**2

    expected_field.field /= n
    error.field = np.sqrt(error.field/n)
    expected_error.field = 10*np.log10(np.abs(mid_field.field+1e-16)) - expected_field.field

    vis = FieldVisualiser(mid_field, env=environment, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='Pade + Transparent BC', x_mult=1E-3)
    expected_vis = FieldVisualiser(expected_field, env=environment, trans_func=lambda v: v,
                          label='Pade + Transparent BC', x_mult=1E-3)

    # plt = vis.plot2d(min=-50, max=0)
    # plt.xlabel('Расстояние, км')
    # plt.ylabel('Высота, м')
    # plt.colorbar(fraction=0.046, pad=0.04, location="bottom")
    # plt.tight_layout()
    # plt.show()

    # plt = vis.plot_hors(1, 3, 10)
    # plt.xlabel('Расстояние, км')
    # plt.ylabel('10log|u|')
    # plt.tight_layout()
    # plt.grid(True)
    # #plt.ylim([0, 15])
    # plt.show()

    vis_error = FieldVisualiser(error, env=environment, trans_func=lambda v: v, label='СКО', x_mult=1E-3)
    vis_expected_error = FieldVisualiser(expected_error, env=environment, trans_func=lambda v: v, label='СКО', x_mult=1E-3)

    # plt = vis_error.plot2d(min=0, max=10)
    # plt.xlabel('Расстояние, км')
    # plt.ylabel('Высота, м')
    # plt.colorbar(fraction=0.046, pad=0.04, location="bottom")
    # plt.tight_layout()
    # plt.grid(True)
    # plt.show()

    # plt = vis_error.plot_hors(1, 3, 10)
    # plt.xlabel('Расстояние, км')
    # plt.ylabel('10log|u|')
    # plt.tight_layout()
    # plt.grid(True)
    # plt.ylim([0, 15])
    # plt.show()

    return vis, expected_vis, vis_expected_error, vis_error


if __name__ == "__main__":
    vis, expected_vis, vis_expected_error, vis_error = do_model(
        freq_hz=10E9,
        antenna_height_m=10,
        duct_expected_height_m=20,
        duct_mean_height_m=5,
        max_range_m=50E3,
        max_height_m=100,
        n=20
    )

    f, ax = plt.subplots(1, 4, figsize=(9, 4), constrained_layout=True)
    norm = Normalize(-40, 0)
    extent = [vis.x_grid[0], vis.x_grid[-1], vis.z_grid[0], vis.z_grid[-1]]
    im = ax[0].imshow(vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
    f.colorbar(im, ax=ax[0], fraction=0.046, location='bottom')
    ax[0].set_xlabel("Расстояние, км")
    ax[0].set_ylabel("Высота, м")
    ax[0].set_title("10log(|u|)")
    ax[0].grid(True)

    norm = Normalize(-40, 0)
    extent = [expected_vis.x_grid[0], expected_vis.x_grid[-1], expected_vis.z_grid[0], expected_vis.z_grid[-1]]
    im = ax[1].imshow(expected_vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
    f.colorbar(im, ax=ax[1], fraction=0.046, location='bottom')
    ax[1].set_xlabel("Расстояние, км")
    ax[1].set_ylabel("Высота, м")
    ax[1].set_title("10log(|u|)")
    ax[1].grid(True)

    norm = Normalize(-10, 10)
    extent = [vis_error.x_grid[0], vis_error.x_grid[-1], vis_error.z_grid[0], vis_error.z_grid[-1]]
    im = ax[2].imshow(vis_expected_error.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('seismic'))
    f.colorbar(im, ax=ax[2], fraction=0.046, location='bottom')
    ax[2].set_xlabel("Расстояние, км")
    ax[2].set_title("СКО")
    ax[2].grid(True)

    norm = Normalize(0, 10)
    extent = [vis_error.x_grid[0], vis_error.x_grid[-1], vis_error.z_grid[0], vis_error.z_grid[-1]]
    im = ax[3].imshow(vis_error.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
    f.colorbar(im, ax=ax[3], fraction=0.046, location='bottom')
    ax[3].set_xlabel("Расстояние, км")
    ax[3].set_title("СКО")
    ax[3].grid(True)

    # do_model(
    #     freq_hz=10E9,
    #     antenna_height_m=20,
    #     duct_expected_height_m=20,
    #     duct_mean_height_m=5,
    #     max_range_m=300E3,
    #     max_height_m=100
    # )