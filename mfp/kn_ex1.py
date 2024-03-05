import matplotlib.pyplot as plt
import numpy as np


from utils import *


#logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
env.z_max = 150
antenna = Source(freq_hz=300E6, height_m=50)
measures = [Measure(x_m=900, height_m=h, freq_hz=antenna.freq_hz) for h in range(10, 60, 2)]
search_area = SearchArea(min_x_m=-250, max_x_m=250)

calc(src=antenna, env=env, range_bounds_m=[-500, 1500], measures=measures, search_area=search_area)
plt.savefig("kn_ex1.eps")

#env.knife_edges = [
#   KnifeEdge(range=-200, height=70),
#]
#calc(src=antenna, env=env, range_bounds_m=[-500, 1500], measures=measures, search_area=search_area)
#plt.savefig("kn_ex2.eps")

#env.knife_edges = [
#   KnifeEdge(range=200, height=70),
#]
#calc(src=antenna, env=env, range_bounds_m=[-500, 1500], measures=measures, search_area=search_area)
#plt.savefig("kn_ex3.eps")

#env.knife_edges = [
#   KnifeEdge(range=-200, height=70),
#   KnifeEdge(range=1300, height=70),
#]
#calc(src=antenna, env=env, range_bounds_m=[-500, 1500], measures=measures, search_area=search_area)
#plt.savefig("kn_ex4.eps")

#measures_h = [Measure(x_m=x, height_m=10, freq_hz=antenna.freq_hz) for x in range(500, 1000, 25)]
#calc(src=antenna, env=env, range_bounds_m=[-500, 1500], measures=measures_h, search_area=search_area)
#plt.savefig("kn_ex5.eps")

env.knife_edges = [
   KnifeEdge(range=-300, height=100),
   KnifeEdge(range=-100, height=30),
   KnifeEdge(range=50, height=20),
   KnifeEdge(range=1300, height=30)
]
calc(src=antenna, env=env, range_bounds_m=[-500, 1500], measures=measures_h, search_area=search_area)
plt.savefig("kn_ex6.eps")