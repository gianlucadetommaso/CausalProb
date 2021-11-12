#!/usr/bin/env python

from causalprob import CausalProb

import numpy as np
import jax.numpy as jnp
from jax import vmap, jit, value_and_grad
import time


def train(model, x: jnp.array, y: jnp.array, o: jnp.array, theta0: jnp.array, n_samples: int = 1000, n_epoch: int = 100,
          batch_size: int = 100, loss_type: str = 'neg-evidence', step_size: float = 1e-4) -> tuple:
    """
    It trains the parameters of the model, given data `x`, `y`, `o`. Starting from an initial solution `theta0`, the
    algorithm uses the Adam optimizer to minimize with respect to the negative evidence (default) or the negative
    averaged log-likelihood.
    Currently, the number of observations for each observed variable must be the same.

    Parameters
    ----------
    x: jnp.array
        Observation of treatment X.
    y: jnp.array
        Observation of outcome Y.
    o: dict
        Observations of observed variables O.
    theta0: dict
        Initial solution of model parameters.
    n_samples: int
        Number of samples to approximate the loss function and its gradient.
    n_epoch: int
        Number of epochs to run the optimization for.
    batch_size: int
        Data batch size at every epoch.
    loss_type: str
        Type of loss function. It can either be:
            - `neg-evidence`: the evidence density function, namely -p(x,y,o);
            - `neg-avg-log-evidence`: the average logarithmic-evidence, namely -E_{u_L}[p(x,y,o|u_L)]
    step_size: float
        Initial step size in Adam optimizer.

    Returns
    -------
    theta, losses: tuple
        theta: jnp.array
            Estimated parameters minimizing the loss function.
        losses: list
            History of loss function evaluations at each iteration of the optimization.
    """
    allowed_losses = ['neg-evidence', 'neg-avg-log-evidence']
    if loss_type not in allowed_losses:
        raise Exception('loss_type={} not recognized. Please select within {}.'.format(loss_type, allowed_losses))

    oy = {**o, 'Y': y}
    all_n_obs = set()
    for v in oy.values():
        if v.ndim > 1:
            all_n_obs.add(v.shape[0])
    if len(all_n_obs) > 1:
        raise Exception('Each observed variable must have the same number of observations.')

    cp = CausalProb(model=model)

    # loss function
    # @jit
    # def loss(_theta: dict):
    #     u_prior = {k: u(n_samples, _theta) for k, u in cp.draw_u.items()}
    #
    #     def _loss(xj, oyj):
    #         u, v = cp.fill(u_prior, {**oyj, 'X': xj}, _theta, cp.draw_u.keys())
    #
    #         def __loss(i: int):
    #             ui = {k: _u[i] if _u.ndim > 1 else _u for k, _u in u.items()}
    #             vi = {k: _v[i] if _v.ndim > 1 else _v for k, _v in v.items()}
    #             llkd = cp.llkd(ui, xj, oyj, _theta, vi)
    #             return jnp.exp(llkd) if loss_type == 'neg-evidence' else llkd
    #         return -jnp.mean(jnp.vectorize(__loss)(range(n_samples)))
    #
    #     if x.ndim > 1:
    #         return jnp.sum(vmap(_loss, (0, {k: 0 for k in oy}))(x, oy))
    #     return _loss(x, oy)

    @jit
    def loss_fn(_theta: dict, x_batch, oy_batch):
        u_prior = {k: u(n_samples, _theta) for k, u in cp.draw_u.items()}

        def _loss_fn(xj, oyj):
            u, v = cp.fill(u_prior, {**oyj, 'X': xj}, _theta, cp.draw_u.keys())
            llkd = cp.llkd(u, xj, oyj, _theta, v)
            return -jnp.mean(llkd)

        if x.ndim > 1:
            return jnp.sum(vmap(_loss_fn, (0, {k: 0 for k in oy}))(x_batch, oy_batch))
        return _loss_fn(x, oy)

    # gradient of loss function
    def dloss_dtheta(_theta: jnp.array):
        u_prior = {k: u(n_samples, _theta) for k, u in cp.draw_u.items()}

        def _dloss_dtheta(xj, oyj):
            u, v = cp.fill(u_prior, {**oyj, 'X': xj}, _theta, cp.draw_u.keys())

            def __dloss_dtheta(i: int):
                ui = {k: _u[i] if _u.ndim > 1 else _u for k, _u in u.items()}

                # prior gradient
                dlpu_dtheta = {key: 0. for key in _theta}
                for rv in ui:
                    if rv != 'X' and rv not in oyj:
                        dlpu_dtheta = {key: dlpu_dtheta[key] + cp.dlpu_dtheta(rv, key, ui, _theta) for key in _theta}

                # likelihood gradient
                dllkd_dtheta = {key: cp.dllkd_dtheta(key, ui, xj, oyj, _theta) for key in _theta}

                # REINFORCE estimator (if prior does not depend on parameters, this is standard gradient direction)
                if loss_type == 'neg-evidence':
                    lkd = jnp.exp(cp.llkd(ui, xj, oyj, _theta))
                    return {key: (dlpu_dtheta[key] + dllkd_dtheta[key]) * lkd for key in _theta}
                else:
                    llkd = cp.llkd(ui, xj, oyj, _theta)
                    return {key: dlpu_dtheta[key] * llkd + dllkd_dtheta[key] for key in _theta}

            return {key: -jnp.mean(g, 0) for key, g in jnp.vectorize(__dloss_dtheta)(range(n_samples)).items()}

        if x.ndim > 1:
            return {key: jnp.sum(g, 0) for key, g in vmap(_dloss_dtheta, (0, {k: 0 for k in oy}))(x, oy).items()}
        return _dloss_dtheta(x, oy)

    # batching
    n_batches = int(np.ceil(x.shape[0] / batch_size))
    rand_perm = np.random.permutation(x.shape[0])
    x_batches = np.array_split(x[rand_perm], n_batches, axis=0)
    oy_batches = {k: np.array_split(v[rand_perm], n_batches, axis=0) for k, v in oy.items()}

    # run optimization
    from jax.experimental import optimizers
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)

    losses = []

    @jit
    def update(i, _opt_state, _x_batch, _oy_batch):
        params = get_params(_opt_state)

        # g = dloss_dtheta(params)
        loss, g = value_and_grad(lambda _theta: loss_fn(_theta, _x_batch, _oy_batch))(params)
        return opt_update(i, g, _opt_state), loss

    opt_state = opt_init(theta0)
    for epoch in range(n_epoch):
        epoch_losses = []

        start_time = time.time()
        for i in range(n_batches):
            x_batch = x_batches[i]
            oy_batch = {k: _oy_batches[i] for k, _oy_batches in oy_batches.items()}

            opt_state, epoch_loss = update(i, opt_state, x_batch, oy_batch)
            epoch_losses.append(epoch_loss)
        epoch_time = time.time() - start_time

        losses.append(np.mean(epoch_losses))
        print("Epoch: {} | time: {:0.2f}s | loss: {:0.8f}".format(epoch + 1, epoch_time, losses[-1]))

    return get_params(opt_state), losses

