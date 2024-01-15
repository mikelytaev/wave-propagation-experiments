from utils import *

logging.basicConfig(level=logging.INFO)

calc(
    M_profile=RandomSurfaceDuct(
        height=norm(loc=200, scale=1),
        m0=norm(loc=350, scale=1),
        m1=norm(loc=300, scale=1),
        slope=norm(loc=0.15, scale=0.005)
    ),
    range_m=250e3,
    height_m=300,
    freq_hz=3E9,
    ant_height=150,
    file_name="surface_duct_1.eps",
    max_monte_carlo_iterations=100
)

calc(
    M_profile=RandomSurfaceDuct(
        height=norm(loc=200, scale=2),
        m0=norm(loc=350, scale=2),
        m1=norm(loc=300, scale=2),
        slope=norm(loc=0.15, scale=0.005)
    ),
    range_m=250e3,
    height_m=300,
    freq_hz=3E9,
    ant_height=150,
    file_name="surface_duct_2.eps",
    max_monte_carlo_iterations=100
)

calc(
    M_profile=RandomSurfaceDuct(
        height=norm(loc=200, scale=5),
        m0=norm(loc=350, scale=5),
        m1=norm(loc=300, scale=5),
        slope=norm(loc=0.15, scale=0.005)
    ),
    range_m=250e3,
    height_m=300,
    freq_hz=3E9,
    ant_height=150,
    file_name="surface_duct_3.eps",
    max_monte_carlo_iterations=100
)
