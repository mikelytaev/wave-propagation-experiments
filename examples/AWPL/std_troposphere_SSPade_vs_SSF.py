from rwp.sspade import *
from rwp.vis import *
from rwp.environment import *
from rwp.petool import PETOOLPropagationTask


logging.basicConfig(level=logging.DEBUG)
environment = Troposphere()
environment.z_max = 300
environment.ground_material = WetGround()
antenna = GaussAntenna(wavelength=0.1, height=30, beam_width=2, eval_angle=0, polarz='H')
max_range = 120000
computational_params = HelmholtzPropagatorComputationalParams(exp_pade_order=(7, 8), storage=PickleStorage())
pade_task = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range,
                                                  comp_params=computational_params)
pade_field = pade_task.calculate()

petool_task = PETOOLPropagationTask(antenna=antenna, env=environment, two_way=False, max_range_m=max_range, dx_wl=400,
                                    n_dx_out=1, dz_wl=3)
petool_field = petool_task.calculate()

pade_vis = FieldVisualiser(pade_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                           label='Pade-[7/8] + NLBC', x_mult=1E-3)
petool_vis = FieldVisualiser(petool_field, trans_func=lambda x: x, label='SSF (PETOOL)', x_mult=1E-3)
plt = petool_vis.plot_hor(30, pade_vis)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.show()

plt = petool_vis.plot2d(min=-120, max=0)
plt.title('10log|u| - Split-step Fourier')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = pade_vis.plot2d(min=-120, max=0)
plt.title('10log|u| - Pade approx. + Transparent BS')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()