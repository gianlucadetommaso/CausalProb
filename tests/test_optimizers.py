from inference.optimization.adam import adam

import unittest
import jax.numpy as jnp
import numpy as np


def quadratic(m=3):
    return lambda x: (x - m) ** 2, lambda x: 2 * (x - m)


class TestAdam(unittest.TestCase):

    def test_quadratic(self, n_iter=100):
        loss, grad_loss = quadratic()

        x0 = jnp.array(np.random.normal())
        x, losses = adam(loss, grad_loss, x0, n_iter)
        assert jnp.allclose(x, jnp.array([3.]), atol=1e-2, rtol=1e-2)
