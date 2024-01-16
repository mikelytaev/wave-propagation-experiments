from utils import *

logging.basicConfig(level=logging.INFO)

calc(
    M_profile=RandomTrilinearDuct(
        z1=norm(loc=50, scale=3),
        z2=norm(loc=150, scale=3),
        m0=norm(loc=300, scale=3),
        m1=norm(loc=330, scale=3),
        m2=norm(loc=310, scale=3),
        slope=norm(loc=0.15, scale=0.005)
    ),
    range_m=250e3,
    height_m=300,
    freq_hz=3E9,
    ant_height=10,
    file_name="elevated_duct_1.eps",
    max_monte_carlo_iterations=100
)

calc(
    M_profile=RandomTrilinearDuct(
        z1=norm(loc=50, scale=3),
        z2=norm(loc=150, scale=3),
        m0=norm(loc=300, scale=3),
        m1=norm(loc=330, scale=3),
        m2=norm(loc=310, scale=3),
        slope=norm(loc=0.15, scale=0.005)
    ),
    range_m=250e3,
    height_m=300,
    freq_hz=3E9,
    ant_height=50,
    file_name="elevated_duct_2.eps",
    max_monte_carlo_iterations=100
)

calc(
    M_profile=RandomTrilinearDuct(
        z1=norm(loc=50, scale=3),
        z2=norm(loc=150, scale=3),
        m0=norm(loc=300, scale=3),
        m1=norm(loc=330, scale=3),
        m2=norm(loc=310, scale=3),
        slope=norm(loc=0.15, scale=0.005)
    ),
    range_m=250e3,
    height_m=300,
    freq_hz=3E9,
    ant_height=150,
    file_name="elevated_duct_3.eps",
    max_monte_carlo_iterations=100
)

calc(
    M_profile=RandomTrilinearDuct(
        z1=norm(loc=50, scale=3),
        z2=norm(loc=150, scale=3),
        m0=norm(loc=300, scale=3),
        m1=norm(loc=330, scale=3),
        m2=norm(loc=310, scale=3),
        slope=norm(loc=0.15, scale=0.005)
    ),
    range_m=250e3,
    height_m=300,
    freq_hz=3E9,
    ant_height=250,
    file_name="elevated_duct_4.eps",
    max_monte_carlo_iterations=100
)
