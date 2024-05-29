import logging

import numpy as np

from uwa.source import GaussSource
from uwa.environment import UnderwaterEnvironment, Bathymetry, munk_profile
from uwa.sspade import UWASSpadeComputationalParams, uwa_ss_pade
from uwa.vis import AcousticPressureFieldVisualiser2d
import torch
import torch.nn as nn
import math as fm
from torchdiffeq import odeint_adjoint as odeint


logging.basicConfig(level=logging.DEBUG)
max_range_m = 25E3
env = UnderwaterEnvironment(
    sound_speed_profile_m_s=lambda x, z: munk_profile(z),
    bottom_profile=Bathymetry(ranges_m=[0], depths_m=[5000]),
    bottom_sound_speed_m_s=1700,
    bottom_density_g_cm=1.5,
    bottom_attenuation_dm_lambda=0.5
)

src = GaussSource(
    freq_hz=50,
    depth_m=100,
    beam_width_deg=3,
    elevation_angle_deg=0,
    multiplier=5
)

params = UWASSpadeComputationalParams(
    max_range_m=max_range_m,
    max_depth_m=5500,
    dx_m=100, # output grid steps affects only on the resulting field, NOT the computational grid
    dz_m=5,
)

field = uwa_ss_pade(
    src=src,
    env=env,
    params=params
)

vis = AcousticPressureFieldVisualiser2d(field=field, env=env)

vis.sound_speed_profile().show()
vis.plot2d(min_val=-80, max_val=0, grid=True, show_terrain=True).show()

measure_points = [(50000, z) for z in np.linspace(10, 1000, 50)]
measures = [field.nearest_value(*point) for point in measure_points]
measure_points = [(measure[0], measure[1]) for measure in measures]
measures = [measure[2] for measure in measures]


dtype = torch.complex
device = "cpu"
torch.set_default_device(device)


class NarrowParabolicEquation(nn.Module):

    def __init__(self, size: int, dz_m: float, k0: float, c0: float):
        super().__init__()
        self.dz_m = dz_m
        self.z_grid_m = np.linspace(0, (size-1)*dz_m, size)
        self.k0 = k0
        self.c0 = c0
        self.depth = nn.Parameter(torch.tensor([5.0]), requires_grad=True)
        self.c_min = nn.Parameter(torch.tensor([5.0]), requires_grad=True)

    def forward(self, x):
        res = np.empty(x.shape, dtype=complex)
        res[0] = 0.0
        res[1:-1] = (1j / (2*self.k0 * self.dz_m**2) * (x[0:-2] - 2*x[:1:-1] + x[2:]) +
                     1j*self.k0*(self.c0 / munk_profile(z_grid_m=self.z_grid_m[1:-1], ref_depth=self.depth, ref_sound_speed=self.c_min))**2-1)
        res[-1] = 0.0
        return res


freq_hz = 50
c0 = 1500
k0 = 2*fm.pi*freq_hz/c0
z_grid_m = np.linspace(0, 1000, 1000)
y0 = src.aperture(k0, z_grid_m)

true_y0 = torch.tensor(y0, dtype=torch.complex128)
x_grid_m = torch.linspace(10, 20000, 1000)


with torch.no_grad():
    true_y = odeint(NarrowParabolicEquation(1000, 1, 2*fm.pi/50, 1500), true_y0, x_grid_m, method='dopri5')
