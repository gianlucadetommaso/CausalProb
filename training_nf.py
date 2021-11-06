#!/usr/bin/env python

from models.normalizing_flow.architectures import RealNVP

from jax.experimental import optimizers
from jax import jit, grad
import jax as jnp
import numpy as np

from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def train_nf(loss, x: np.array, y: np.array, params0: jnp.array, step_size: float = 1e-4, n_iters: int = 10000):
    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-4)

    @jit
    def step(i, batch, opt_state):
        params = get_params(opt_state)
        g = grad(loss)(batch, params)
        return opt_update(i, g, opt_state)

    data_generator = (x[np.random.choice(x.shape[0], 100)] for _ in range(n_iters))

    opt_state = opt_init(params0)
    for i in range(n_iters):
        opt_state = step(i, next(data_generator), opt_state)

    return get_params(opt_state)


if __name__ == '__main__':
    n_samples = 2000
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    x, y = noisy_moons
    x = StandardScaler().fit_transform(x)

    model = RealNVP(dim=2)
    loss = lambda batch, params: -jnp.mean(model.evaluate_forward_layer_logpdf(batch, params))

    params0 = model.init_all_params()

    print(params0)

