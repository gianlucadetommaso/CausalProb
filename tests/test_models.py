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
        u, v = cp.fill({k: cp.draw_u[k](1, theta, seed) for seed, k in enumerate(cp.draw_u)}, {}, theta, cp.draw_u.keys())

        for rv in cp.f:
            assert jnp.allclose(cp.finv[rv](cp.f[rv](u[rv], theta, v), theta, v), u[rv])

    def test_determinant(self):
        from models.nf_confounder_model import define_model
        dim = 2
        model = define_model(dim=dim)
        cp = CausalProb(model=model)
        theta = {k: cp.init_params[k](i) for i, k in enumerate(cp.init_params)}
        u, v = cp.fill({k: cp.draw_u[k](1, theta, seed) for seed, k in enumerate(cp.draw_u)}, {}, theta, cp.draw_u.keys())

        for rv in cp.ldij:
            assert jnp.allclose(jnp.round(cp.ldij[rv](v[rv], theta, v).squeeze(), 4),
                                jnp.round(
                                    jnp.log(
                                        jnp.abs(
                                            jnp.linalg.det(
                                                cp.dfinvv_dv(rv, {k: _v.squeeze(0) for k, _v in v.items()}, theta)))), 4))

