from rwp.urban import *
from rwp.antennas import GaussSource3D
from rwp.vis import FieldVisualiser3D
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

env = Manhattan3D(domain_width=50, domain_height=100, x_max=1000)
env.add_block(center=(500, 50), size=(1000, 70, 50))
env.add_block(center=(500, -50), size=(1000, 70, 50))
ant = GaussSource3D(freq_hz=900E6, height=50, ver_beamwidth=30, hor_beamwidth=30, polarz='H')

comp_params = FDUrbanPropagatorComputationParameters(dx_wl=5, dy_wl=0.5, dz_wl=0.5,
                                                         n_dx_out=1, n_dy_out=1, n_dz_out=1,
                                                         pade_order=(3, 4), abs_layer_scale=0.25)
urban_propagator = FDUrbanPropagator(env=env, comp_params=comp_params, freq_hz=ant.freq_hz)
field = urban_propagator.calculate(ant)

vis = FieldVisualiser3D(field=field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)))

vis.plot_xz(y0=0, min_val=-45, max_val=0)
plt.xlabel('Расстояние, м')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()
plt.savefig("xz.eps")

vis.plot_xy(z0=5, min_val=-45, max_val=0)
plt.xlabel('Расстояние, м')
plt.ylabel('y, м)')
plt.tight_layout()
plt.show()
plt.savefig("xy_z=5.eps")

vis.plot_yz(x0=300, min_val=-45, max_val=0)
plt.xlabel('y, м')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()
plt.savefig("yz_x=300.eps")

vis.plot_yz(x0=900, min_val=-45, max_val=0)
plt.xlabel('y, м')
plt.ylabel('Высота, м')
plt.tight_layout()
plt.show()
plt.savefig("yz_x=900.eps")