from causalprob import CausalProb
from inference.optimization.adam import adam

import jax.numpy as jnp


def train(model, x: jnp.array, o: jnp.array, theta0: jnp.array, n_samples: int = 1000000, options: dict = None):
    cp = CausalProb(model=model)

    def loss(_theta):
        u, v = cp.fill({k: u(n_samples) for k, u in cp.draw_u.items()}, {**o, 'X': x}, _theta, cp.draw_u.keys())

        def _loss(i: int):
            ui = {k: _u[i] if _u.ndim > 1 else _u for k, _u in u.items()}
            vi = {k: _v[i] if _v.ndim > 1 else _v for k, _v in v.items()}
            return cp.llkd(ui, x, o, _theta, vi)

        return -jnp.mean(jnp.vectorize(_loss)(range(n_samples)))

    def dloss_dtheta(_theta):
        u, v = cp.fill({k: u(n_samples) for k, u in cp.draw_u.items()}, {**o, 'X': x}, _theta, cp.draw_u.keys())

        def _dloss_dtheta(i: int):
            ui = {k: _u[i] if _u.ndim > 1 else _u for k, _u in u.items()}
            vi = {k: _v[i] if _v.ndim > 1 else _v for k, _v in v.items()}

            lkd = jnp.exp(cp.llkd(ui, x, o, _theta, vi))
            dllkd_dtheta = {key: cp.dllkd_dtheta(key, ui, x, o, theta, vi) for key in theta}
            dlpu_dtheta = cp.dlpu_dtheta()
            return (dlpu_dtheta + dllkd_dtheta) * lkd

        return -jnp.mean(jnp.vectorize(_dloss_dtheta)(range(n_samples)))

    theta = adam(loss, dloss_dtheta, theta0, )