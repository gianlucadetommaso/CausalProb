import unittest
import jax.numpy as jnp
import numpy as np


class TestRealNVP(unittest.TestCase):

    def test_is_forward_layer_bijection(self):
        from models.normalizing_flow.architectures import RealNVP
        dim = 2
        model = RealNVP(dim=dim)

        output_shape, layer_params = model.init_layer_params()
        u0 = jnp.array(np.random.normal(size=(4, dim)))

        v = model.forward_layer(u0, layer_params)
        u1 = model.backward_layer(v, layer_params)[0]

        assert jnp.allclose(jnp.round(u0, 4), jnp.round(u1, 4))

    def test_is_forward_bijection(self):
        from models.normalizing_flow.architectures import RealNVP
        dim = 2
        model = RealNVP(dim=dim)

        all_params = model.init_all_params()
        u0 = jnp.array(np.random.normal(size=(4, dim)))

        v = model.forward(u0, all_params)
        u1 = model.backward(v, all_params)[0]

        assert jnp.allclose(jnp.round(u0, 4), jnp.round(u1, 4))

    def test_shift_and_log_scale(self):
        from models.normalizing_flow.architectures import RealNVP
        dim = 2
        model = RealNVP(dim=dim)

        output_shape, layer_params = model.init_layer_params()
        u0 = jnp.array(np.random.normal(size=(4, dim // 2)))

        s = model.shift_and_log_scale_fn(u0, layer_params)
        assert s[0].shape == u0.shape
        assert s[1].shape == u0.shape
