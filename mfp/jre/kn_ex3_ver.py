import matplotlib.pyplot as plt
import numpy as np


from utils import *


#logging.basicConfig(level=logging.DEBUG)
true_env = Troposphere(flat=True)
true_env.z_max = 150
antenna = Source(freq_hz=300E6, height_m=50)
measures = [Measure(x_m=1500, height_m=h, freq_hz=antenna.freq_hz) for h in range(10, 60, 2)]
search_area = SearchArea(min_x_m=-250, max_x_m=250)

true_env.knife_edges = [
    KnifeEdge(range=-200, height=70),
    KnifeEdge(range=1800, height=50),
    KnifeEdge(range=1000, height=30),
]
calc(src=antenna, true_env=true_env, range_bounds_m=[-500, 2000], measures=measures,
     search_area=search_area)
plt.savefig("kn_ex3_ver.png")
