from utils import *
#logging.basicConfig(level=logging.DEBUG)


def evaporation_operator(height_m: float, model: RWPModel):
    return model.apply_N_profile(EvaporationDuctModel(height_m=height_m))


def evaporation_loss0(height_m: float, model: RWPModel, measure):
    val = evaporation_operator(height_m, model)
    return Bartlett_loss(val, measure)


inv_model = RWPModel()
evaporation_operator(10.0, inv_model)
d = jax.jacfwd(evaporation_operator)(15.0, inv_model)
print(jnp.linalg.norm(d))

g = jax.grad(evaporation_loss0)(10.0, inv_model, jnp.array([1.0 +1j, 2, 3, 4, 5]))
print(g)
