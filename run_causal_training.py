#!/usr/bin/env python

from causalprob import CausalProb
from inference.training import train
from tools.structures import sum_trees

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time


if __name__ == '__main__':
    dim = 2
    n_train_obs = 1000
    n_samples = 1000
    n_pred_obs = 10
    lam = 10

    from models.linear_confounder_model import define_model
    linear_model = define_model(dim=dim)
    # true_theta = {k: linear_model['init_params'][k]() for k in linear_model['init_params']}
    true_theta = {'V1->X': jnp.array([3., 4.]), 'X->Y': jnp.array([1., 2.]), 'V1->Y': jnp.array([-2., -3.])}

    linear_cp = CausalProb(model=linear_model)
    u_train, v_train = linear_cp.fill({k: u(n_train_obs, true_theta) for k, u in linear_cp.draw_u.items()}, {}, true_theta, linear_cp.draw_u.keys())
    x_train = v_train['X']
    o_train = {}  # {'V1': v_train['V1']}
    y_train = v_train['Y']

    u_pred, v_pred = linear_cp.fill({k: u(n_pred_obs, true_theta) for k, u in linear_cp.draw_u.items()}, {}, true_theta, linear_cp.draw_u.keys())
    x_pred = v_pred['X']
    o_pred = {}  # {'V1': v_pred['V1']}

    from models.linear_confounder_model import define_model
    est_model = define_model(dim=dim)
    theta0 = {k: est_model['init_params'][k](i) for i, k in enumerate(est_model['init_params'])}
    # theta0 = {k: th + jnp.array(np.random.normal(scale=1, size=2)) for k, th in true_theta.items()}

    print('\nTrain normalizing flow model parameters...')
    start_time = time.time()
    theta, losses, training_losses, biases = train(est_model, x_train, y_train, o_train, x_pred, o_pred, theta0,
                                                   lam=lam, step_size=1, n_epoch=100, batch_size=1000)
    print("Completed in {:0.2f} seconds.".format(time.time() - start_time))
    print(theta)
    est_cp = CausalProb(model=est_model)
    _, est_v = est_cp.fill({k: u(n_train_obs, theta) for k, u in est_cp.draw_u.items()}, {}, theta, est_cp.draw_u.keys())

    plt.figure(figsize=(20, 3))
    plt.subplot(1, 3, 1)
    plt.plot(losses, color='C0')
    plt.legend(['overall loss'], fontsize=12, loc='upper right')
    plt.subplot(1, 3, 2)
    plt.plot(training_losses, color='C1')
    plt.legend(['training loss'], fontsize=12, loc='upper right')
    plt.subplot(1, 3, 3)
    plt.semilogy(biases, color='C2')
    plt.legend(['bias'], fontsize=12, loc='upper right')
    plt.show()

    plt.figure(figsize=(10, 6))
    for i, rv, in enumerate(u_train):
        plt.suptitle("recover linear model observations via NF model")
        plt.subplot(3, 2, 2 * i + 1)
        plt.title(rv, fontsize=15)
        plt.scatter(v_train[rv][:, 0], v_train[rv][:, 1], s=1, color='blue')
        plt.legend(['true samples'], fontsize=12)
        plt.subplot(3, 2, 2 * i + 2)
        plt.title(rv, fontsize=15)
        plt.scatter(est_v[rv][:, 0], est_v[rv][:, 1], s=1, color='red')
        plt.legend(['estimated samples'], fontsize=12)
    plt.tight_layout()
    plt.show()

    print("\nCompute true causal effect and true causal bias...")
    start_time = time.time()
    true_causal_effect = jnp.mean(linear_cp.causal_effect(x_pred, o_pred, true_theta, n_samples=1000), 0)
    true_causal_bias = jnp.mean(linear_cp.causal_bias(x_pred, o_pred, true_theta, n_samples=1000), 0)
    print("Completed in {:0.2f} seconds.".format(time.time() - start_time))

    print("\nEstimate causal effect and causal bias from pre-trained normalizing flow model.")
    start_time = time.time()
    est_causal_effect = jnp.mean(est_cp.causal_effect(x_pred, o_pred, theta, n_samples=1000), 0)
    est_causal_bias = jnp.mean(est_cp.causal_bias(x_pred, o_pred, theta, n_samples=1000), 0)
    print("Completed in {:0.2f} seconds.".format(time.time() - start_time))

    print('true causal effect: ', true_causal_effect)
    print('estimated causal effect: ', est_causal_effect)
    print()
    print('true causal bias: ', true_causal_bias)
    print('estimated causal bias: ', est_causal_bias)

