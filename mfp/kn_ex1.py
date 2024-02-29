import os

import numpy as np

os.chdir('../../')
from rwp.kediffraction import *
from rwp.antennas import *
from rwp.environment import *
from rwp.vis import *

logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
env.z_max = 150
env.knife_edges = [
    KnifeEdge(range=-200, height=70),
    KnifeEdge(range=1300, height=70),
]
antenna = Source(wavelength=1, height_m=50)
#antenna = GaussAntenna(freq_hz=300e6, height=100, beam_width=15, elevation_angle=0, polarz='H')

kdc = KnifeEdgeDiffractionCalculator(src=antenna, env=env, min_range_m=-500, max_range_m=1500, max_propagation_angle=90)
field = kdc.calculate()

vis = FieldVisualiser(field, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='ke')
plt = vis.plot2d(min=-40, max=10)
plt.title('The intensity of the field component 10log10|u|')
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
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
    normalizer = deepcopy(fields[0])
    normalizer.field *= 0
    for ind in range(0, len(fields)):
        res.field += measures[ind].value.conjugate() * fields[ind].field
        normalizer.field += abs(fields[ind].field) ** 2

    res.field = res.field**2 / normalizer.field
    return res


def mv_mvp(measures: List[Measure], fields: List[Field]) -> Field:
    res = deepcopy(fields[0])
    res.field *= 0
    d = np.array([m.value for m in measures], dtype=complex)
    k = np.matmul(d.reshape(len(d), 1), d.reshape(1, len(d)).conj())
    k_inv = np.linalg.inv(k)

    normalizer = deepcopy(fields[0])
    normalizer.field *= 0
    for ind in range(0, len(fields)):
        normalizer.field += abs(fields[ind].field) ** 2

    normalizer.field = np.sqrt(normalizer.field)

    for ind_i in range(0, res.field.shape[0]):
        for ind_j in range(0, res.field.shape[1]):
            w = np.array([f.field[ind_i, ind_j] for f in fields]) / normalizer.field[ind_i, ind_j]
            w = w.reshape(len(w), 1)
            t = np.matmul(np.matmul(w.reshape(1, len(w)).conj(), k_inv), w)
            res.field[ind_i, ind_j] = (1 / t) / (1 / np.matmul(w.reshape(1, len(w)).conj(), w))

    return res


def mfp(measures: List[Measure], env: Troposphere, search_area: SearchArea) -> List[Field]:
    fields = []
    for measure in measures:
        antenna = PointSource(freq_hz=measure.freq_hz, height_m=measure.height_m)
        shifted_env = deepcopy(env)
        shifted_search_area = deepcopy(search_area)
        for knife_edge in shifted_env.knife_edges:
            knife_edge.range -= measure.x_m
        shifted_search_area.min_x_m -= measure.x_m
        shifted_search_area.max_x_m -= measure.x_m
        kdc = KnifeEdgeDiffractionCalculator(src=antenna, env=shifted_env,
                                             min_range_m=shifted_search_area.min_x_m,
                                             max_range_m=shifted_search_area.max_x_m,
                                             )
        field = kdc.calculate()
        fields.append(field)
    return fields

measures = [Measure(x_m=900, height_m=h, value=field.value(900, h), freq_hz=antenna.freq_hz) for h in range(10, 60, 2)]

fields = mfp(measures=measures, env=env, search_area=SearchArea(
    min_x_m=-250,
    max_x_m=250
))


vis = FieldVisualiser(fields[0], env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='ke')
plt = vis.plot2d(min=-40, max=10)
plt.title('The intensity of the field component 10log10|u|')
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.show()


bartlett_mfp_field = bartlett_mvp(measures=measures, fields=fields)
vis = FieldVisualiser(bartlett_mfp_field, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='ke')
plt = vis.plot2d(min=vis.max-5, max=vis.max)
plt.title('The intensity of the field component 10log10|u|')
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.show()

mv_mfp_field = mv_mvp(measures=measures, fields=fields)
vis = FieldVisualiser(mv_mfp_field, env=env, trans_func=lambda v: 10 * cm.log10(1e-66 + abs(v)), label='ke')
plt = vis.plot2d(min=vis.max-10, max=vis.max)
plt.title('The intensity of the field component 10log10|u|')
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.show()
