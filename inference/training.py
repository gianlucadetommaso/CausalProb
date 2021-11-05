#!/usr/bin/env python

from causalprob import CausalProb
from inference.optimization.adam import adam
from tools.structures import pack, unpack

import jax.numpy as jnp
from jax import grad, jvp, vmap, jacfwd


def train(model, x: jnp.array, y: jnp.array, o: jnp.array, theta0: jnp.array, n_samples: int = 1000, n_iter: int = 100,
          loss_type: str = 'neg-evidence') -> tuple:
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

    def loss(_theta: jnp.array):
        unpacked_theta = unpack(_theta, theta0)
        u_prior = {k: u(n_samples, unpacked_theta) for k, u in cp.draw_u.items()}

        def _loss(xj, oyj):
            u, v = cp.fill(u_prior, {**oyj, 'X': xj}, unpacked_theta, cp.draw_u.keys())

            def __loss(i: int):
                ui = {k: _u[i] if _u.ndim > 1 else _u for k, _u in u.items()}
                vi = {k: _v[i] if _v.ndim > 1 else _v for k, _v in v.items()}
                return cp.llkd(ui, xj, oyj, unpacked_theta, vi)

            return -jnp.mean(jnp.vectorize(__loss)(range(n_samples)))

        if x.ndim > 1:
            return jnp.sum(vmap(_loss, (0, {k: 0 for k in oy}))(x, oy))
        return _loss(x, oy)

    def dloss_dtheta(_theta: jnp.array):
        unpacked_theta = unpack(_theta, theta0)
        u_prior = {k: u(n_samples, unpacked_theta) for k, u in cp.draw_u.items()}

        def _dloss_dtheta(xj, oyj):
            u, v = cp.fill(u_prior, {**oyj, 'X': xj}, unpacked_theta, cp.draw_u.keys())

            def __dloss_dtheta(i: int):
                ui = {k: _u[i] if _u.ndim > 1 else _u for k, _u in u.items()}

                # prior gradient
                dlpu_dtheta = 0
                for rv in ui:
                    if rv != 'X' and rv not in oyj:
                        dlpu_dtheta += pack({key: cp.dlpu_dtheta(rv, key, ui, _theta) for key in _theta})

                # likelihood gradient
                dllkd_dtheta = pack({key: cp.dllkd_dtheta(key, ui, xj, oyj, unpacked_theta) for key in unpacked_theta})

                # REINFORCE estimator (if prior does not depend on parameters, this is standard gradient direction)
                if loss_type == 'neg-evidence':
                    lkd = jnp.exp(cp.llkd(ui, xj, oyj, unpacked_theta))
                    return (dlpu_dtheta + dllkd_dtheta) * lkd
                else:
                    llkd = cp.llkd(ui, xj, oyj, unpacked_theta)
                    return dlpu_dtheta * llkd + dllkd_dtheta

            return -jnp.mean(jnp.vectorize(__dloss_dtheta, signature=('()->(ntheta)'))(range(n_samples)), 0)

        if x.ndim > 1:
            return jnp.sum(vmap(_dloss_dtheta, (0, {k: 0 for k in oy}))(x, oy), 0)
        return _dloss_dtheta(x, oy)

    theta, losses = adam(loss, dloss_dtheta, pack(theta0), n_iter=n_iter)
    return unpack(theta, theta0), losses


