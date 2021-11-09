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
    n_obs = 10000
    n_samples = 100

    from models.linear_confounder_model import define_model
    linear_model = define_model(dim=dim)
    true_theta = {k: linear_model['init_params'][k]() for k in linear_model['init_params']}

    linear_cp = CausalProb(model=linear_model)
    u, v = linear_cp.fill({k: u(n_obs, true_theta) for k, u in linear_cp.draw_u.items()}, {}, true_theta, linear_cp.draw_u.keys())
    x = v['X']
    o = {'V1': v['V1']}
    y = v['Y']

    from models.nf_confounder_model import define_model
    nf_model = define_model(dim=dim)
    theta0 = {k: nf_model['init_params'][k](i) for i, k in enumerate(nf_model['init_params'])}

    start_time = time.time()
    print('Train model parameters...')
    theta, losses = train(nf_model, x, y, o, theta0, loss_type='neg-avg-log-evidence', step_size=1e-2, n_iter=1000)
    print('Training completed in {} seconds.'.format(np.round(time.time() - start_time, 2)))
    plt.plot(losses)
    plt.show()

    nf_cp = CausalProb(model=nf_model)
    _, est_v = nf_cp.fill({k: u(n_samples, theta) for k, u in nf_cp.draw_u.items()}, {}, theta, nf_cp.draw_u.keys())

    plt.figure(figsize=(10, 6))
    for i, rv, in enumerate(u):
        plt.subplot(3, 2, 2 * i + 1)
        plt.title(rv, fontsize=15)
        plt.scatter(v[rv][:, 0], v[rv][:, 1], s=1, color='blue')
        plt.legend(['true samples'], fontsize=12)
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.subplot(3, 2, 2 * i + 2)
        plt.title(rv, fontsize=15)
        plt.scatter(est_v[rv][:, 0], est_v[rv][:, 1], s=1, color='red')
        plt.legend(['estimated samples'], fontsize=12)
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
    plt.tight_layout()
    plt.show()

