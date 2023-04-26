# Incidence of vertically polarized waves at the Brewster angle

from rwp.sspade import *
from rwp.vis import *
from rwp.environment import *
from propagators.wavenumber import *
from rwp.petool import PETOOLPropagationTask

logging.basicConfig(level=logging.DEBUG)
environment = Troposphere(flat=True)
environment.z_max = 100
environment.ground_material = CustomMaterial(eps=3, sigma=0)

freq_hz = 3000e6
b_angle = brewster_angle(1, environment.ground_material.complex_permittivity(freq_hz))

antenna = GaussAntenna(freq_hz=freq_hz, height=50, beam_width=0.3, eval_angle=(90-b_angle).real, polarz='V')
h1 = antenna.height_m
h2 = 0
a = abs((h1 - h2) / cm.tan(abs(antenna.eval_angle) * cm.pi / 180))
max_range = 2 * a + 200

pade_params = HelmholtzPropagatorComputationalParams(two_way=False,
                                                     exp_pade_order=(7, 8),
                                                     max_propagation_angle=abs(antenna.eval_angle)+5,
                                                     z_order=4)
pade_task = TroposphericRadioWaveSSPadePropagator(antenna=antenna, env=environment, max_range_m=max_range, comp_params=pade_params)
pade_field = pade_task.calculate()

petool_task = PETOOLPropagationTask(antenna=antenna, env=environment, two_way=False, max_range_m=max_range, dx_wl=1, n_dx_out=1, dz_wl=0.2, n_dz_out=5)
petool_field = petool_task.calculate()

wnparams = WaveNumberIntegratorParams(fcc_tol=1e-9,
                                      x_grid_m=np.linspace(1, max_range, 500),
                                      z_computational_grid_m=np.linspace(0, 100, 3000),
                                      z_out_grid_m=np.linspace(0, 100, 501),
                                      alpha=1e-4,
                                      max_p_k0=1000,
                                      lower_refl_coef=lambda theta: reflection_coef(1+1e-4j,
                                                                                    environment.ground_material.complex_permittivity(antenna.freq_hz), 90-theta, 'V'))
wnp = WaveNumberIntegrator(k0=2*cm.pi / antenna.wavelength, initial_func=lambda z: antenna.aperture(z), params=wnparams)
res = wnp.calculate()

pade_vis = FieldVisualiser(pade_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='Pade + NLBC', x_mult=1)

plt = pade_vis.plot2d(min=-70, max=0)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.grid(True)
plt.show()

wn_field = Field(x_grid=wnparams.x_grid_m, z_grid=wnparams.z_out_grid_m, freq_hz=300e6)
wn_field.field[:, :] = res
wn_vis = FieldVisualiser(wn_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='Wavenumber integration')
plt = wn_vis.plot2d(min=-70, max=0)
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.grid(True)
plt.tight_layout()
plt.show()

petool_vis = FieldVisualiser(petool_field, trans_func=lambda x: x, label='SSF (PETOOL)', x_mult=1)

plt = petool_vis.plot2d(min=-70, max=0)
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.grid(True)
plt.tight_layout()
plt.show()

plt = petool_vis.plot_hor(50, pade_vis, wn_vis)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.grid(True)
plt.show()