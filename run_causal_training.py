#!/usr/bin/env python

from causalprob import CausalProb
from inference.training import train
from tools.structures import unpack

import jax.numpy as jnp
from jax import vmap
import numpy as np
import matplotlib.pyplot as plt
import time


if __name__ == '__main__':
    dim = 2
    n_obs = 1000
    n_samples = 1000000

    from models.nf_confounder_model import define_model
    model = define_model(dim=dim)
    true_theta = {k: model['init_params'][k](i) for i, k in enumerate(model['init_params'])}

    cp = CausalProb(model=model)
    u, v = cp.fill({k: u(n_obs, true_theta) for k, u in cp.draw_u.items()}, {}, true_theta, cp.draw_u.keys())
    x = v['X']
    o = {'V1': v['V1']}
    y = v['Y']

    theta0 = {k: model['init_params'][k](10 + i) for i, k in enumerate(model['init_params'])}

    oy = {**o, 'Y': y}

    def loss(_theta: dict):
        u_prior = {k: u(n_samples, _theta) for k, u in cp.draw_u.items()}

        def _loss(xj, oyj):
            u, v = cp.fill(u_prior, {**oyj, 'X': xj}, _theta, cp.draw_u.keys())

            def __loss(i: int):
                ui = {k: _u[i] if _u.ndim > 1 else _u for k, _u in u.items()}
                vi = {k: _v[i] if _v.ndim > 1 else _v for k, _v in v.items()}
                llkd = cp.llkd(ui, xj, oyj, _theta, vi)
                return llkd
            return -jnp.mean(jnp.vectorize(__loss)(range(n_samples)))

        return vmap(_loss, (0, {k: 0 for k in oy}))(x, oy)

    start_time = time.time()
    print('Train model parameters...')
    theta, losses = train(model, x, y, o, theta0, loss_type='neg-avg-log-evidence', step_size=1e-4, n_iter=10000)
    print('Training completed in {} seconds.'.format(np.round(time.time() - start_time, 2)))
    plt.plot(losses)
    plt.show()
