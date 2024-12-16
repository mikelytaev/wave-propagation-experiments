from utils import *


inv_model_1ghz = RWPModel(params=RWPComputationalParams(
        max_range_m=5000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*11,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    src=RWPGaussSourceModel(freq_hz=1E9, height_m=10.0, beam_width_deg=3.0)
)

inv_model_10ghz = RWPModel(params=RWPComputationalParams(
        max_range_m=5000,
        max_height_m=250,
        dx_m=100,
        dz_m=1
    ),
    measure_points_x=[-1]*11,
    measure_points_z=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    src=RWPGaussSourceModel(freq_hz=0.5E9, height_m=10.0, beam_width_deg=3.0)
)