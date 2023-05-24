from propagators.wavenumber import *
from rwp.field import *
from rwp.vis import *
import logging
from rwp.antennas import *
from propagators._utils import *

logging.basicConfig(level=logging.DEBUG)
wnparams = WaveNumberIntegratorParams(fcc_tol=1e-9,
                                      x_grid_m=np.linspace(1, 200, 500),
                                      z_computational_grid_m=np.linspace(0, 100, 3001),
                                      z_out_grid_m=np.array([0, 50.03, 100]),#np.linspace(0, 100, 501),
                                      alpha=1e-4,
                                      max_p_k0=1000,
                                      lower_refl_coef=lambda theta: reflection_coef(1+1e-4j, 3, 90-theta, 'V'))
wavelength = 0.1
antenna = GaussAntenna(freq_hz=LIGHT_SPEED/wavelength, height=50, beam_width=0.3, elevation_angle=30, polarz='V')
wnp = WaveNumberIntegrator(k0=2*cm.pi / wavelength, initial_func=lambda z: antenna.aperture(z), params=wnparams)
#wnp = WaveNumberIntegrator(k0=2*cm.pi / wavelength, initial_func=DeltaFunction(x_c=50), params=wnparams)
res = wnp.calculate()

field = Field(x_grid=wnparams.x_grid_m, z_grid=wnparams.z_out_grid_m, freq_hz=300e6)
field.field[:, :] = res
vis = FieldVisualiser(field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='ke')
plt = vis.plot2d(min=-90, max=-25)
plt.title('The intensity of the field component 10log10|u|')
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.show()

plt = vis.plot_hor(50)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.grid(True)
plt.show()
