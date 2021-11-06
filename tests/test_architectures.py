import unittest
import jax.numpy as jnp
import numpy as np


class TestRealNVP(unittest.TestCase):

    def test_is_layer_bijection(self):
        from models.normalizing_flow.architectures import RealNVP
        dim = 2
        model = RealNVP(dim=dim)

        output_shape, layer_params = model.init_layer_params()
        u0 = jnp.array(np.random.normal(size=(4, dim)))

        v = model.forward_layer(u0, layer_params)
        u1 = model.backward_layer(v, layer_params)[0]

        assert jnp.allclose(jnp.round(u0, 4), jnp.round(u1, 4))
