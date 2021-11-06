from causalprob import CausalProb

import unittest
import jax.numpy as jnp
import numpy as np


def test_model_causal_estimates(cp, x, o, theta, true_causal_effect, true_causal_bias):
    n_samples = 1000000
    est_causal_effect = cp.causal_effect(x=x, o=o, theta=theta, n_samples=n_samples)
    est_causal_bias = cp.causal_bias(x=x, o=o, theta=theta, n_samples=n_samples)

    assert jnp.allclose(true_causal_effect, est_causal_effect, atol=1e-2, rtol=1e-2)
    assert jnp.allclose(true_causal_bias, est_causal_bias, atol=1e-2, rtol=1e-2)


class TestCausalEstimates(unittest.TestCase):

    def test_linear_confounder_model_causal_estimates(self):
        from models.linear_confounder_model import define_model

        dims = [1, 2, 10]
        for dim in dims:
            with self.subTest(dim=dim):
                theta = {'V1->X': jnp.array(np.random.normal(size=dim)),
                         'X->Y': jnp.array(np.random.normal(size=dim)),
                         'V1->Y': jnp.array(np.random.normal(size=dim))}

                cp = CausalProb(model=define_model(dim=dim))
                u, v = cp.fill({k: u(1, theta) for k, u in cp.draw_u.items()}, {}, theta, cp.draw_u.keys())
                x = v['X'].squeeze(0)
                o = {}

                alpha, beta, gamma = list(theta.values())
                true_causal_effect = jnp.diag(beta)
                true_causal_bias = jnp.diag(gamma * alpha / (1 + alpha ** 2))

                test_model_causal_estimates(cp, x, o, theta, true_causal_effect, true_causal_bias)

    def test_linear_overcontrol_model_causal_estimates(self):
        from models.linear_overcontrol_model import define_model

        dims = [1, 2, 10]
        for dim in dims:
            with self.subTest(dim=dim):
                theta = {'X->V1': jnp.array(np.random.normal(size=dim)),
                         'X->Y': jnp.array(np.random.normal(size=dim)),
                         'V1->Y': jnp.array(np.random.normal(size=dim))}

                cp = CausalProb(model=define_model(dim=dim))
                u, v = cp.fill({k: u(1, theta) for k, u in cp.draw_u.items()}, {}, theta, cp.draw_u.keys())
                x = v['X'].squeeze(0)
                o = {'V1': v['V1'].squeeze(0)}

                alpha, beta, gamma = list(theta.values())
                true_causal_effect = jnp.diag(beta + gamma * alpha)
                true_causal_bias = jnp.diag(-gamma * alpha)

                test_model_causal_estimates(cp, x, o, theta, true_causal_effect, true_causal_bias)

    def test_linear_selection_model_causal_estimates(self):
        from models.linear_selection_model import define_model

        dims = [1, 2, 10]
        for dim in dims:
            with self.subTest(dim=dim):
                theta = {'X->Y': jnp.array(np.random.normal(size=dim)),
                         'X->V1': jnp.array(np.random.normal(size=dim)),
                         'Y->V1': jnp.array(np.random.normal(size=dim))}

                cp = CausalProb(model=define_model(dim=dim))
                u, v = cp.fill({k: u(1, theta) for k, u in cp.draw_u.items()}, {}, theta, cp.draw_u.keys())
                x = v['X'].squeeze(0)
                o = {'V1': v['V1'].squeeze(0)}

                alpha, beta, gamma = list(theta.values())
                true_causal_effect = jnp.diag(alpha)
                true_causal_bias = jnp.diag(-gamma * (beta + gamma * alpha) / (1 + gamma ** 2))

                test_model_causal_estimates(cp, x, o, theta, true_causal_effect, true_causal_bias)



