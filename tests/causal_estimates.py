from causalprob import CausalProb

import unittest
import models
import jax.numpy as jnp
import numpy as np


def test_model_causal_estimates(model, theta, true_causal_effect, true_causal_bias):
    cp = CausalProb(model=model)

    u, v = cp.fill({k: u(1) for k, u in cp.draw_u.items()}, {}, theta, cp.draw_u.keys())
    x = v['X'].squeeze(0)
    o = {'V1': v['V1'].squeeze(0)}

    n_samples = 1000000
    est_causal_effect = cp.causal_effect(x=x, o=o, theta=theta, n_samples=n_samples)
    est_causal_bias = cp.causal_effect(x=x, o=o, theta=theta, n_samples=n_samples)

    jnp.allclose(true_causal_effect, est_causal_effect)
    jnp.allclose(true_causal_bias, est_causal_bias)


class TestCausalEstimates(unittest.TestCase):

    def test_linear_confounder_model_causal_estimates(self):
        from models.linear_confounder_model import define_model

        dims = [1, 2, 10]
        for dim in dims:
            with self.subTest(dim=dim):
                theta = {'V1->X': jnp.array(np.random.normal(size=dim)),
                         'X->Y': jnp.array(np.random.normal(size=dim)),
                         'V1->Y': jnp.array(np.random.normal(size=dim))}
                alpha, beta, gamma = list(theta.values())
                true_causal_effect = jnp.diag(beta)
                true_causal_bias = jnp.diag(gamma * alpha / (1 + alpha ** 2))

                test_model_causal_estimates(define_model(dim=dim), theta, true_causal_effect, true_causal_bias)

    def test_linear_overcontrol_model_causal_estimates(self):
        from models.linear_overcontrol_model import define_model

        dims = [1, 2, 10]
        for dim in dims:
            with self.subTest(dim=dim):
                theta = {'X->V1': jnp.array(np.random.normal(size=dim)),
                         'X->Y': jnp.array(np.random.normal(size=dim)),
                         'V1->Y': jnp.array(np.random.normal(size=dim))}
                alpha, beta, gamma = list(theta.values())
                true_causal_effect = jnp.diag(beta + gamma * alpha)
                true_causal_bias = jnp.diag(-gamma * alpha)

                test_model_causal_estimates(define_model(dim=dim), theta, true_causal_effect, true_causal_bias)

    def test_linear_selection_model_causal_estimates(self):
        from models.linear_selection_model import define_model

        dims = [1, 2, 10]
        for dim in dims:
            with self.subTest(dim=dim):
                theta = {'X->Y': jnp.array(np.random.normal(size=dim)),
                         'X->V1': jnp.array(np.random.normal(size=dim)),
                         'Y->V1': jnp.array(np.random.normal(size=dim))}
                alpha, beta, gamma = list(theta.values())
                true_causal_effect = jnp.diag(alpha)
                true_causal_bias = jnp.diag(-gamma * (beta + gamma * alpha) / (1 + gamma ** 2))

                test_model_causal_estimates(define_model(dim=dim), theta, true_causal_effect, true_causal_bias)



