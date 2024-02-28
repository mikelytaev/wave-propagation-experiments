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
antenna = Source(wavelength=1, height_m=50)
#antenna = GaussAntenna(freq_hz=300e6, height=100, beam_width=15, elevation_angle=0, polarz='H')

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


@dataclass
class SearchArea:
    min_x_m: float
    max_x_m: float
    min_z_m: float = None
    max_x_m: float = None


def bartlett_mvp(measures: List[Measure], fields: List[Field]) -> Field:
    res = deepcopy(fields[0])
    res.field *= 0
    for ind in range(0, len(fields)):
        res.field += (measures[ind].value.conjugate() * fields[ind].field) ** 2

    return res


def mfp(measures: List[Measure], env: Troposphere, search_area: SearchArea):
    fields = []
    for measure in measures:
        antenna = PointSource(freq_hz=measure.freq_hz, height_m=measure.height_m, value=measure.value)
        shifted_env = deepcopy(env)
        shifted_search_area = deepcopy(search_area)
        for knife_edge in shifted_env.knife_edges:
            pass
            knife_edge.range -= measure.x_m
        shifted_search_area.min_x_m -= measure.x_m
        shifted_search_area.max_x_m -= measure.x_m
        kdc = KnifeEdgeDiffractionCalculator(src=antenna, env=env,
                                             min_range_m=shifted_search_area.min_x_m,
                                             max_range_m=shifted_search_area.max_x_m,
                                             )
        field = kdc.calculate()
        fields.append(field)


measure = Measure(x_m=900, height_m=50, value=field.value(900, 50), freq_hz=antenna.freq_hz)
mfp(measures=[measure], env=env, search_area=SearchArea(
    min_x_m=0,
    max_x_m=500
))
