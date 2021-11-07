#!/usr/bin/env python

from causalprob import CausalProb
from inference.optimization.adam import adam
from tools.structures import pack, unpack

import jax.numpy as jnp
from jax import vmap, jit
from jax import random


def train(model, x: jnp.array, y: jnp.array, o: jnp.array, theta0: jnp.array, n_samples: int = 1000, n_iter: int = 100,
          loss_type: str = 'neg-evidence', step_size: float = 1., print_every: int = 1) -> tuple:
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
    n_iter: int
        Number of iterations of the optimizer.
    loss_type: str
        Type of loss function. It can either be:
            - `neg-evidence`: the evidence density function, namely -p(x,y,o);
            - `neg-avg-log-evidence`: the average logarithmic-evidence, namely -E_{u_L}[p(x,y,o|u_L)]
    step_size: float
        Initial step size in Adam optimizer.
    print_every: int
        After how many iterations to print current status of the optimization.

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
        raise Exception('`loss_type={} not recognized. Please select within {}.'.format(loss_type, allowed_losses))

    oy = {**o, 'Y': y}
    all_n_obs = set()
    for v in oy.values():
        if v.ndim > 1:
            all_n_obs.add(v.shape[0])
    if len(all_n_obs) > 1:
        raise Exception('Each observed variable must have the same number of observations.')

    cp = CausalProb(model=model)

    # loss function
    def loss(_theta: dict):
        u_prior = {k: u(n_samples, _theta) for k, u in cp.draw_u.items()}

        def _loss(xj, oyj):
            u, v = cp.fill(u_prior, {**oyj, 'X': xj}, _theta, cp.draw_u.keys())

            def __loss(i: int):
                ui = {k: _u[i] if _u.ndim > 1 else _u for k, _u in u.items()}
                vi = {k: _v[i] if _v.ndim > 1 else _v for k, _v in v.items()}
                llkd = cp.llkd(ui, xj, oyj, _theta, vi)
                return jnp.exp(llkd) if loss_type == 'neg-evidence' else llkd

            return -jnp.mean(jnp.vectorize(__loss)(range(n_samples)))

        if x.ndim > 1:
            return jnp.sum(vmap(_loss, (0, {k: 0 for k in oy}))(x, oy))
        return _loss(x, oy)

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

    # run optimization
    from jax.experimental import optimizers
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)

    losses = []

    strf = "{:<10} {:<25}"
    print(strf.format("iter", "loss"))
    print(40 * '-')

    @jit
    def step(i, opt_state):
        params = get_params(opt_state)

        _loss = loss(params)
        g = dloss_dtheta(params)
        return opt_update(i, g, opt_state), _loss

    opt_state = opt_init(theta0)
    for i in range(n_iter):
        opt_state, _loss = step(i, opt_state)
        losses.append(_loss)

        if i % print_every == 0:
            print(strf.format(i, _loss))

    return get_params(opt_state), losses

