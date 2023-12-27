import logging
from copy import deepcopy

import numpy as np
from scipy.interpolate import interp1d

from rwp.antennas import GaussAntenna
from rwp.environment import Troposphere, Impediment, Terrain
from rwp.field import Field
from rwp.sspade import RWPSSpadeComputationalParams, rwp_ss_pade, SSPadeZOrder
from rwp.terrain import inv_geodesic_problem
from rwp.vis import FieldVisualiser

import importlib


def solution(
        freq_hz: float,
        polarz: str,
        src_height_m: float,
        src_1m_power_db: float,
        drone_1m_power_db: float,
        drone_max_height_m: float,
        drone_max_range_m: float,
        dst_height_m: float,
        dst_range_m: float,
        dst_1m_power_db: float,
        src_min_power_db: float,
        drone_min_power_db: float,
        dst_min_power_db: float,
        env: Troposphere,
        beam_width=30
) -> (Field, Field):
    logging.basicConfig(level=logging.DEBUG)
    antenna_src = GaussAntenna(freq_hz=freq_hz,
                           height=src_height_m,
                           beam_width=beam_width,
                           elevation_angle=0,
                           polarz=polarz)

    antenna_dst = GaussAntenna(freq_hz=freq_hz,
                           height=dst_height_m,
                           beam_width=beam_width,
                           elevation_angle=0,
                           polarz=polarz)

    params = RWPSSpadeComputationalParams(
        max_range_m=dst_range_m,
        max_height_m=drone_max_height_m,
        dx_m=25,  # output grid steps affects only on the resulting field, NOT the computational grid
        dz_m=0.5,
        #dx_computational_grid_wl=25,
        #dz_computational_grid_wl=0.5,
        max_propagation_angle_deg=beam_width,
        z_order=SSPadeZOrder.joined
    )

    inv_env = Troposphere(flat=env.is_flat)
    inv_env.M_profile = (lambda x, z: env.M_profile(params.max_range_m - x, z)) if env.M_profile else None
    inv_env.terrain = Terrain(
        elevation=lambda x: env.terrain.elevation(params.max_range_m - x),
        ground_material=lambda x: env.terrain.ground_material(params.max_range_m - x)
    )
    inv_env.vegetation = [Impediment(
        left_m=params.max_range_m - imp.x2,
        right_m=params.max_range_m - imp.x1,
        height_m=imp.height,
        material=imp.material)
        for imp in env.vegetation]

    loss_src = rwp_ss_pade(antenna=antenna_src, env=env, params=params).path_loss()
    loss_src.normalize_loss()
    loss_dst = rwp_ss_pade(antenna=antenna_dst, env=inv_env, params=params).path_loss()
    loss_dst.normalize_loss()
    loss_dst.field = loss_dst.field[::-1, :]

    src_bw = deepcopy(loss_src)
    src_bw.field = np.logical_and(
        src_1m_power_db - loss_src.field > drone_min_power_db,
        drone_1m_power_db - loss_src.field > src_min_power_db)
    src_bw.log10 = False

    dst_bw = deepcopy(loss_dst)
    dst_bw.field = np.logical_and(
        dst_1m_power_db - loss_dst.field > drone_min_power_db,
        drone_1m_power_db - loss_dst.field > dst_min_power_db)
    dst_bw.log10 = False

    merge = deepcopy(src_bw)
    merge.field = np.logical_and(src_bw.field, dst_bw.field)
    merge.log10 = False

    opt = deepcopy(loss_src)
    opt.field = np.minimum(np.minimum(np.minimum(
        src_1m_power_db - loss_src.field - drone_min_power_db,
        drone_1m_power_db - loss_src.field - src_min_power_db),
        dst_1m_power_db - loss_dst.field - drone_min_power_db),
        drone_1m_power_db - loss_dst.field - dst_min_power_db)

    src_vis = FieldVisualiser(loss_src, env=env, trans_func=lambda v: v, x_mult=1E-3)
    dst_vis = FieldVisualiser(loss_dst, env=env, trans_func=lambda v: v, x_mult=1E-3)
    src_bw_vis = FieldVisualiser(src_bw, env=env, trans_func=lambda v: v, x_mult=1E-3, bw=True)
    dst_bw_vis = FieldVisualiser(dst_bw, env=env, trans_func=lambda v: v, x_mult=1E-3, bw=True)
    merge_vis = FieldVisualiser(merge, env=env, trans_func=lambda v: v, x_mult=1E-3, bw=True)
    opt_vis = FieldVisualiser(opt, env=env, trans_func=lambda v: v, x_mult=1E-3, bw=True)

    return src_vis, dst_vis, src_bw_vis, dst_bw_vis, merge_vis, opt_vis


def get_elevation_func(lat1: float, long1: float, lat2: float, long2: float, n_points: int):
    from podpac.datalib.terraintiles import TerrainTiles
    from podpac import Coordinates
    from podpac import settings

    settings['DEFAULT_CACHE'] = ['disk']
    node = TerrainTiles(tile_format='geotiff', zoom=11)
    coords, x_grid = inv_geodesic_problem(lat1, long1, lat2, long2, n_points)
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    c = Coordinates([lats, lons], dims=['lat', 'lon'])
    o = node.eval(c)
    eval = np.array([o.data[i, i] for i in range(0, len(x_grid))])
    eval[np.isnan(eval)] = 0
    eval = np.array([max(a, 0) for a in eval])

    #podpac делает какую-то дичь с логгером
    importlib.reload(logging)

    return interp1d(x=x_grid, y=eval, fill_value="extrapolate")
