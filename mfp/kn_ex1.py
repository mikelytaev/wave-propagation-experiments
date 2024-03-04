import numpy as np


from utils import *


#logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
env.z_max = 150
#env.knife_edges = [
#    KnifeEdge(range=-200, height=70),
#    KnifeEdge(range=1300, height=70),
#]
antenna = Source(freq_hz=300E6, height_m=50)
measures = [Measure(x_m=900, height_m=h, freq_hz=antenna.freq_hz) for h in range(10, 60, 2)]
search_area=SearchArea(min_x_m=-250, max_x_m=250)
calc(src=antenna, env=env, range_bounds_m=[-500, 1500], measures=measures, search_area=search_area)

