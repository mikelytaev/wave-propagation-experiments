from rwp.environment import evaporation_duct


def get_modified_evaporation_duct(duct_height_m, z_shift):
    ed_shift = evaporation_duct(height=0, z_grid_m=z_shift) - evaporation_duct(height=duct_height_m, z_grid_m=z_shift)
    a = evaporation_duct(height=0, z_grid_m=0)
    b = evaporation_duct(height=0, z_grid_m=1) - evaporation_duct(height=0, z_grid_m=0)
    return lambda x, z: (
            (evaporation_duct(height=duct_height_m, z_grid_m=z) + ed_shift) * (z <= z_shift) + (a + b * z) * (
                z > z_shift))
