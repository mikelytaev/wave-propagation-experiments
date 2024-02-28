import os
os.chdir('../../')
from rwp.kediffraction import *
from rwp.antennas import *
from rwp.environment import *
from rwp.vis import *

logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
env.z_max = 150
env.knife_edges = [
    KnifeEdge(range=200, height=50),
    KnifeEdge(range=800, height=50),
    KnifeEdge(range=-500, height=50)
]
#antenna = Source(wavelength=1, height_m=50)
antenna = GaussAntenna(freq_hz=300e6, height=100, beam_width=15, elevation_angle=0, polarz='H')

kdc = KnifeEdgeDiffractionCalculator(src=antenna, env=env, min_range_m=-1000, max_range_m=1000, max_propagation_angle=90)
field = kdc.calculate()

vis = FieldVisualiser(field, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='ke')
plt = vis.plot2d(min=-40, max=10)
plt.title('The intensity of the field component 10log10|u|')
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.show()

plt = vis.plot_hor(51)
plt.title('The intensity of the field component 10log10|u| at the height 51 m')
plt.xlabel('Range (m)')
plt.ylabel('10log|u| (dB)')
plt.show()


@dataclass
class Measure:
    x_m: float
    height_m: float
    value: complex
    freq_hz: float


def mfp(measures: List[Measure], env: Troposphere):
    for measure in measures:
        antenna = Source(freq_hz=measure.freq_hz, height_m=measure.value)
        shifted_env = deepcopy(env)
        for knife_edge in shifted_env.knife_edges:
            pass
            knife_edge.range -= measure.x_m
        kdc = KnifeEdgeDiffractionCalculator(src=antenna, env=env, max_range_m=1000)
        field = kdc.calculate()
