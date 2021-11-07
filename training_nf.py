#!/usr/bin/env python

from models.normalizing_flow.architectures import RealNVP

from jax.experimental import optimizers
from jax import jit, grad
import jax.numpy as jnp
import numpy as np
import time
import os

from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import animation


def train(loss, X: np.array, params0: jnp.array, step_size: float = 1e-4, n_iter: int = 10000, batch_size: int = 100) -> list:
    """
    Train model to data `X` using Adam optimizer on input `loss`, starting from initial parameters `params0`. It returns
    the optimal parameters.

    Parameters
    ----------
    loss: func
        Loss function to minimize. It takes as input:
        - params: list
            Parameters of the model.
        - batch: jnp.array
            Batch of data.
    X: np.array
        Data. Shape: (number of samples, number of features)
    params0: list
        Initial solution of model parameters.
    step_size: float
        Initial step size of the optimization algorithm.
    n_iter: int
        Number of interations of the optimization.
    batch_size: int
        Batch size at every iteration.

    Returns
    -------
    params: list
        Learned model parameters.
    """
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)

    @jit
    def step(i, batch, opt_state):
        params = get_params(opt_state)
        g = grad(loss)(params, batch)
        return opt_update(i, g, opt_state)

    data_generator = (X[np.random.choice(X.shape[0], batch_size)] for _ in range(n_iter))

    opt_state = opt_init(params0)
    for i in range(n_iter):
        opt_state = step(i, next(data_generator), opt_state)

    return get_params(opt_state)


if __name__ == '__main__':
    n_samples = 2000
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    X, y = noisy_moons
    X = StandardScaler().fit_transform(X)

    model = RealNVP(dim=2, seed=42)
    loss = lambda params, batch: -jnp.mean(model.evaluate_forward_logpdf(batch, params))

    params0 = model.init_all_params()
    start_time = time.time()
    print('Train model parameters...')
    params = train(loss, X, params0)
    print('Training completed in {} seconds.'.format(np.round(time.time() - start_time, 2)))

    v = model.sample_base(1000)
    values = [v]
    flip = False
    for l in range(model.n_layers):
        v = model.forward_layer(v, params[l], flip=flip)
        values.append(v)
        flip = not flip

    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots()

    y = values[0]
    plt.scatter(X[:, 0], X[:, 1], s=10, color='blue')
    paths = ax.scatter(y[:, 0], y[:, 1], s=10, color='red')

    def animate(i):
        l = i // 48
        t = float(i % 48) / 48
        y = (1 - t) * values[l] + t * values[l + 1]
        paths.set_offsets(y)
        return paths

    # `brew install imagemagick` if not available
    print('Produce animation...')
    anim = animation.FuncAnimation(fig, animate, frames=48 * model.n_layers, interval=1, blit=False)

    animation_name = 'animation.gif'
    anim.save(animation_name, writer='imagemagick', fps=60)
    print('Animation saved in {}/{}.'.format(os.getcwd(), animation_name))

