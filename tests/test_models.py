from causalprob import CausalProb

import unittest
import jax.numpy as jnp
import numpy as np


class TestNFConfounderModel(unittest.TestCase):

    def test_is_inverse_function(self):
        from models.nf_confounder_model import define_model
        dim = 2
        model = define_model(dim=dim)
        cp = CausalProb(model=model)
        theta = {k: cp.init_params[k](i) for i, k in enumerate(cp.init_params)}
        u, v = cp.fill({k: u(1, theta) for k, u in cp.draw_u.items()}, {}, theta, cp.draw_u.keys())

        for rv in cp.f:
            assert jnp.allclose(cp.finv[rv](cp.f[rv](u[rv], theta, v), theta, v), u[rv])



