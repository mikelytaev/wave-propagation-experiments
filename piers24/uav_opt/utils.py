import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np

from rwp.antennas import GaussAntenna
from rwp.environment import Troposphere, Impediment, Terrain
from rwp.field import Field
from rwp.sspade import RWPSSpadeComputationalParams, rwp_ss_pade, SSPadeZOrder
from rwp.vis import FieldVisualiser


@dataclass
class AntennaParams:
    power_dBm: float
    gain_dBi: float
    sensitivity_dBm: float
    height_m: Optional[float] = None
    beam_width_deg: Optional[float] = None


def solution(
        freq_hz: float,
        polarz: str,
        src_params: AntennaParams,
        drone_params: AntennaParams,
        dst_params: AntennaParams,
        drone_max_height_m: float,
        drone_max_range_m: float,
        dst_range_m: float,
        env: Troposphere,
) -> (Field, Field):
    logging.basicConfig(level=logging.DEBUG)
    antenna_src = GaussAntenna(freq_hz=freq_hz,
                           height=src_params.height_m,
                           beam_width=src_params.beam_width_deg,
                           elevation_angle=0,
                           polarz=polarz)


    antenna_dst = GaussAntenna(freq_hz=freq_hz,
                           height=dst_params.height_m,
                           beam_width=dst_params.beam_width_deg,
                           elevation_angle=0,
                           polarz=polarz)
    params = RWPSSpadeComputationalParams(
        max_range_m=dst_range_m,
        max_height_m=drone_max_height_m,
        max_propagation_angle_deg=max(src_params.beam_width_deg, dst_params.beam_width_deg),
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
    loss_dst = rwp_ss_pade(antenna=antenna_dst, env=inv_env, params=params).path_loss()
    loss_dst.field = loss_dst.field[::-1, :]

    src_drone_received = src_params.power_dBm + src_params.gain_dBi - loss_src.field + drone_params.gain_dBi
    drone_src_received = drone_params.power_dBm + drone_params.gain_dBi - loss_src.field + src_params.gain_dBi
    dst_drone_received = dst_params.power_dBm + dst_params.gain_dBi - loss_dst.field + drone_params.gain_dBi
    drone_dst_received = drone_params.power_dBm + drone_params.gain_dBi - loss_dst.field + dst_params.gain_dBi

    src_bw = deepcopy(loss_src)
    src_bw.field = np.logical_and(
        src_drone_received > drone_params.sensitivity_dBm,
        drone_src_received > src_params.sensitivity_dBm)
    src_bw.log10 = False


    dst_bw = deepcopy(loss_dst)
    dst_bw.field = np.logical_and(
        dst_drone_received > drone_params.sensitivity_dBm,
        drone_dst_received > dst_params.sensitivity_dBm)
    dst_bw.log10 = False

    merge = deepcopy(src_bw)
    merge.field = np.logical_and(src_bw.field, dst_bw.field)
    merge.log10 = False

    opt_link_margin = deepcopy(loss_src)
    opt_link_margin.field = np.minimum(np.minimum(np.minimum(
        src_drone_received - drone_params.sensitivity_dBm,
        drone_src_received - src_params.sensitivity_dBm),
        dst_drone_received - drone_params.sensitivity_dBm),
        drone_dst_received - dst_params.sensitivity_dBm)

    src_vis = FieldVisualiser(loss_src, env=env, trans_func=lambda v: v, x_mult=1E-3)
    dst_vis = FieldVisualiser(loss_dst, env=env, trans_func=lambda v: v, x_mult=1E-3)
    src_bw_vis = FieldVisualiser(src_bw, env=env, trans_func=lambda v: v, x_mult=1E-3, bw=True)
    dst_bw_vis = FieldVisualiser(dst_bw, env=env, trans_func=lambda v: v, x_mult=1E-3, bw=True)
    merge_vis = FieldVisualiser(merge, env=env, trans_func=lambda v: v, x_mult=1E-3, bw=True)
    opt_link_margin_vis = FieldVisualiser(opt_link_margin, env=env, trans_func=lambda v: v, x_mult=1E-3, bw=True)

    return src_vis, dst_vis, src_bw_vis, dst_bw_vis, merge_vis, opt_link_margin_vis


