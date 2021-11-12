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
    n_samples = 100

    from models.linear_confounder_model import define_model
    linear_model = define_model(dim=dim)
    true_theta = {k: linear_model['init_params'][k]() for k in linear_model['init_params']}

    linear_cp = CausalProb(model=linear_model)
    u, v = linear_cp.fill({k: u(n_obs, true_theta) for k, u in linear_cp.draw_u.items()}, {}, true_theta, linear_cp.draw_u.keys())
    x = v['X']
    o = {}  # {'V1': v['V1']}
    y = v['Y']

    print("\nCompute true causal effect and true causal bias...")
    start_time = time.time()
    true_causal_effect = jnp.mean(linear_cp.causal_effect(x, o, true_theta), 0)
    true_causal_bias = jnp.mean(linear_cp.causal_bias(x, o, true_theta), 0)
    print("Completed in {:0.2f} seconds.".format(time.time() - start_time))

    from models.nf_confounder_model import define_model
    nf_model = define_model(dim=dim)
    theta0 = {k: nf_model['init_params'][k](i) for i, k in enumerate(nf_model['init_params'])}

    print('\nTrain normalizing flow model parameters...')
    start_time = time.time()
    theta, losses = train(nf_model, x, y, o, theta0, loss_type='neg-avg-log-evidence', step_size=1e-4, n_epoch=100, batch_size=100)
    print("Completed in {:0.2f} seconds.".format(time.time() - start_time))

    nf_cp = CausalProb(model=nf_model)
    _, est_v = nf_cp.fill({k: u(n_obs, theta) for k, u in nf_cp.draw_u.items()}, {}, theta, nf_cp.draw_u.keys())

    print("\nEstimate causal effect and causal bias from pre-trained normalizing flow model.")
    start_time = time.time()
    est_causal_effect = jnp.mean(nf_cp.causal_effect(x, o, theta), 0)
    est_causal_bias = jnp.mean(nf_cp.causal_bias(x, o, theta), 0)
    print("Completed in {:0.2f} seconds.".format(time.time() - start_time))

    plt.plot(losses)
    plt.show()

    plt.figure(figsize=(10, 6))
    for i, rv, in enumerate(u):
        plt.suptitle("recover linear model observations via NF model")
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

    print('true causal effect: ', true_causal_effect)
    print('estimated causal effect: ', est_causal_effect)
    print()
    print('true causal bias: ', true_causal_bias)
    print('estimated causal bias: ', est_causal_bias)

