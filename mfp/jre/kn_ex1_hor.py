import matplotlib.pyplot as plt
import numpy as np


from utils import *


#logging.basicConfig(level=logging.DEBUG)
true_env = Troposphere(flat=True)
true_env.z_max = 150
antenna = Source(freq_hz=300E6, height_m=50)
measures = [Measure(x_m=x, height_m=10, freq_hz=antenna.freq_hz) for x in range(1000, 1250, 10)]
search_area = SearchArea(min_x_m=-250, max_x_m=250)

true_env.knife_edges = [
    KnifeEdge(range=-200, height=70),
]
expected_env = deepcopy(true_env)
expected_env.knife_edges = [
    KnifeEdge(range=-200, height=70),
]
calc(src=antenna, true_env=true_env, expected_env=expected_env, range_bounds_m=[-500, 2000], measures=measures,
     search_area=search_area)
plt.savefig("kn_ex1_hor.png")


for r in [-195, -199, -201, -205]:
    expected_env = deepcopy(true_env)
    expected_env.knife_edges = [
        KnifeEdge(range=r, height=70),
    ]
    print(f'r={r}')
    calc(src=antenna, true_env=true_env, expected_env=expected_env, range_bounds_m=[-500, 2000], measures=measures,
         search_area=search_area)


for h in [68, 72]:
    expected_env = deepcopy(true_env)
    expected_env.knife_edges = [
        KnifeEdge(range=-200, height=h),
    ]
    print(f'h={h}')
    calc(src=antenna, true_env=true_env, expected_env=expected_env, range_bounds_m=[-500, 2000], measures=measures,
         search_area=search_area)
