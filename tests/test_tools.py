import unittest
import jax.numpy as jnp
import numpy as np


class TestTools(unittest.TestCase):

    def test_pack(self):
        from tools.structures import pack

        d = {'a': jnp.array(np.arange(24).reshape(2, 3, 4)), 'b': jnp.array([]), 'c': jnp.array([5., 6.])}
        assert jnp.allclose(pack(d), jnp.concatenate((jnp.arange(24), jnp.array([5., 6.]))))

    def test_unpack(self):
        from tools.structures import unpack

        d = {'a': jnp.array(np.arange(24).reshape(2, 3, 4)), 'b': jnp.array([]), 'c': jnp.array([5., 6.])}
        a = jnp.array(10 + np.arange(26))

        true_unpacked = {'a': a[:24].reshape(d['a'].shape), 'b': jnp.array([]), 'c': a[-2:]}
        unpacked = unpack(a, d)
        for k in d:
            assert jnp.allclose(unpacked[k], true_unpacked[k])


