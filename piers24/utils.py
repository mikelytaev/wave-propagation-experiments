from copy import deepcopy

import numpy as np

from rwp.antennas import GaussAntenna
from rwp.environment import Troposphere, Impediment
from rwp.field import Field
from rwp.sspade import RWPSSpadeComputationalParams, rwp_ss_pade
from rwp.vis import FieldVisualiser


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
        env: Troposphere
) -> (Field, Field):
    antenna_src = GaussAntenna(freq_hz=freq_hz,
                           height=src_height_m,
                           beam_width=5,
                           elevation_angle=0,
                           polarz=polarz)

    antenna_dst = GaussAntenna(freq_hz=freq_hz,
                           height=dst_height_m,
                           beam_width=5,
                           elevation_angle=0,
                           polarz=polarz)

    params = RWPSSpadeComputationalParams(
        max_range_m=dst_range_m,
        max_height_m=drone_max_height_m,
        dx_m=100,  # output grid steps affects only on the resulting field, NOT the computational grid
        dz_m=1
    )

    inv_env = Troposphere(flat=env.is_flat)
    inv_env.M_profile = (lambda x, z: env.M_profile(params.max_range_m - x, z)) if env.M_profile else None
    inv_env.terrain.elevation = lambda x: env.terrain.elevation(params.max_range_m - x)
    inv_env.terrain.ground_material = lambda x: env.terrain.ground_material(params.max_range_m - x)
    inv_env.vegetation = [Impediment(
        left_m=params.max_range_m - imp.right_m,
        right_m=params.max_range_m - imp.left_m,
        height_m=imp.height,
        material=imp.material)
        for imp in env.vegetation]

    field_src = rwp_ss_pade(antenna=antenna_src, env=env, params=params)
    field_src.field = 10 * np.log10(np.abs(field_src.field))
    field_src.log10 = True
    field_src.normalize()
    field_dst = rwp_ss_pade(antenna=antenna_dst, env=inv_env, params=params)
    field_dst.field = 10 * np.log10(np.abs(field_dst.field))
    field_dst.log10 = True
    field_dst.normalize()

    src_loss = deepcopy(field_src)
    src_loss.field += src_1m_power_db
    dst_loss = deepcopy(field_dst)
    dst_loss.field += dst_1m_power_db

    src_vis = FieldVisualiser(field_src, env=env, x_mult=1E-3)
    dst_vis = FieldVisualiser(field_dst, env=inv_env, x_mult=1E-3)

    return src_vis, dst_vis
