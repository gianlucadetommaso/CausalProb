#!/usr/bin/env python

from causalprob import CausalProb
from inference.training import train
from tools.structures import unpack

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time


if __name__ == '__main__':
    dim = 2
    n_obs = 100
    n_samples = 5

    from models.nf_confounder_model import define_model
    model = define_model(dim=dim)
    true_theta = {k: model['init_params'][k](i) for i, k in enumerate(model['init_params'])}

    cp = CausalProb(model=model)
    u, v = cp.fill({k: u(n_obs, true_theta) for k, u in cp.draw_u.items()}, {}, true_theta, cp.draw_u.keys())
    x = v['X']
    o = {'V1': v['V1']}
    y = v['Y']

    theta0 = {k: model['init_params'][k](10 + i) for i, k in enumerate(model['init_params'])}

    start_time = time.time()
    print('Train model parameters...')
    theta, losses = train(model, x, y, o, theta0, loss_type='neg-avg-log-evidence')
    print('Training completed in {} seconds.'.format(np.round(time.time() - start_time, 2)))
    print('optimal theta2 parameters:', theta)
    plt.plot(losses)
    plt.show()

    # _, est_v = cp.fill({k: u(n_samples, theta) for k, u in cp.draw_u.items()}, {}, theta, cp.draw_u.keys())
    #
    # plt.figure(figsize=(20, 4))
    # for i, rv, in enumerate(u):
    #     plt.subplot(1, 3, i + 1)
    #     plt.title(rv, fontsize=15)
    #     plt.hist(v[rv].squeeze(1), bins=int(np.sqrt(2 * n_obs)), alpha=0.5, density=True)[-1]
    #     plt.hist(est_v[rv].squeeze(1), bins=int(np.sqrt(2 * n_samples)), alpha=0.5, density=True)[-1]
    #     plt.legend(['true distribution', 'estimated distribution'], fontsize=12)
    # plt.show()
