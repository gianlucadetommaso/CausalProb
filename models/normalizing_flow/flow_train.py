#!/usr/bin/env python

from models.normalizing_flow.distributions import StandardGaussian
from flax.linen import Dense, relu
import optax
import jax
from jax.nn import tanh
import functools

from jax.experimental import optimizers
from jax import jit, grad, value_and_grad
import jax.numpy as jnp
from jax.random import PRNGKey
import numpy as np
import time


from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm


def train(loss, X: np.array, params: jnp.array, step_size: float = 3e-4, n_iter: int = 100 , batch_size: int = 100) -> list:
 
    optimizer = optax.chain(optax.adamw(step_size), optax.clip_by_global_norm(5.0))

    @jit
    def step(i, batch, params, opt_state):
        l, g = value_and_grad(loss)(params, batch)
        updates, opt_state = optimizer.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        return l, params

    data_generator = (X[np.random.choice(X.shape[0], batch_size)] for _ in range(n_iter))

    opt_state = optimizer.init(params)
    tqdm_iter = tqdm(range(n_iter))
    for i in tqdm_iter:
        l, new_params = step(i, next(data_generator), params, opt_state)
        if jnp.any(jnp.isnan(l)) or jnp.any(jnp.isinf(l)):
            return params
        params = new_params
        tqdm_iter.set_postfix({"loss": l})
    return params


def nll_loss(params, batch, flow_dist):
   return -jnp.mean(flow_dist.apply(params, batch, method=flow_dist.log_prob))


def get_moons_dataset(n_samples=5000):
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    X, y = noisy_moons
    X = StandardScaler().fit_transform(X)
    return X


def get_test_loop(flow_dist):
    X = get_moons_dataset(2000)
    return functools.partial(train, X=X, loss=functools.partial(nll_loss, flow_dist=flow_dist))


def plot_samples(X):
     
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        
    #print(y)
    ax.set_title("Training Samples")
    ax.hist2d(*X.T, bins=200)
    
    fig.show()
    plt.show()


if __name__ == '__main__':
    n_samples = 5000
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    X, y = noisy_moons
    X = StandardScaler().fit_transform(X)

   
    mlp = lambda: Sequential([
        Dense(256),
        relu,
        Dense(256),
        relu,
        Dense(2)
     ])
    
    x = jnp.ones((32, 2), dtype=jnp.float32)
    
    prng_key = PRNGKey(0)
    
    
    flow = NormalizingFlow((RealNVP(mlp(), False), RealNVP(mlp(), True)))
    flow_dist = NormalizingFlowDist(StandardGaussian(2), flow)
    
    
    params = flow_dist.init(prng_key, x, method=flow_dist.log_prob)
    loss = lambda params, batch: -jnp.mean(flow_dist.apply(params, batch, method=flow_dist.log_prob))

    start_time = time.time()
    print('Train model parameters...')
    params = train(loss, X, params, n_iter=5000, step_size=2e-5)
    print('Training completed in {} seconds.'.format(np.round(time.time() - start_time, 2)))

    prng_key,_ = jax.random.split(prng_key)
    y = flow_dist.apply(params, prng_key, 2000, method=flow_dist.sample)
    
    
    fig, ax = plt.subplots(2, 1, figsize=(5, 10))
        
    #print(y)
    ax[0].set_title("Training Samples")
    ax[0].hist2d(*X.T, bins=200)
    ax[1].set_title("Flow Samples")
    ax[1].hist2d(*y.T, bins=200)
    # plt.scatter(X[:, 0], X[:, 1], s=10, color='blue')
    # plt.scatter(y[:, 0], y[:, 1], s=10, color='red')
    
    
    fig.show()
    plt.show()