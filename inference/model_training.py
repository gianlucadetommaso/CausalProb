#!/usr/bin/env python

from causalprob import CausalProb
from inference.optimization.adam import adam
from tools.structures import pack, unpack

import jax.numpy as jnp


def train(model, x: jnp.array, y: jnp.array, o: jnp.array, theta0: jnp.array, n_samples: int = 1000, n_iter: int = 100):
    oy = {**o, 'Y': y}
    cp = CausalProb(model=model)

    def loss(_theta: jnp.array):
        unpacked_theta = unpack(_theta, theta0)
        u_prior = {k: u(n_samples) for k, u in cp.draw_u.items()}

        def _loss(xj, oyj):
            u, v = cp.fill(u_prior, {**oyj, 'X': xj}, unpacked_theta, cp.draw_u.keys())

            def __loss(i: int):
                ui = {k: _u[i] if _u.ndim > 1 else _u for k, _u in u.items()}
                vi = {k: _v[i] if _v.ndim > 1 else _v for k, _v in v.items()}
                return cp.llkd(ui, x, oy, unpacked_theta, vi)

        return -jnp.mean(jnp.vectorize(_loss)(range(n_samples)))

    def dloss_dtheta(_theta: jnp.array):
        unpacked_theta = unpack(_theta, theta0)
        u, v = cp.fill({k: u(n_samples) for k, u in cp.draw_u.items()}, {**oy, 'X': x}, unpacked_theta, cp.draw_u.keys())

        def _dloss_dtheta(i: int):
            ui = {k: _u[i] if _u.ndim > 1 else _u for k, _u in u.items()}

            # likelihood gradient part
            llkd = cp.llkd(ui, x, oy, unpacked_theta)
            dllkd_dtheta = pack({key: cp.dllkd_dtheta(key, ui, x, oy, unpacked_theta) for key in unpacked_theta})

            # prior gradient part
            dlpu_dtheta = 0
            for rv in ui:
                if rv != 'X' and rv not in oy:
                    dlpu_dtheta += pack({key: cp.dlpu_dtheta(rv, key, ui, _theta) for key in _theta})

            # REINFORCE estimator (if prior does not depend on parameters, this is standard gradient direction
            # return (dlpu_dtheta + dllkd_dtheta) * lkd
            return dlpu_dtheta * llkd + dllkd_dtheta

        return -jnp.mean(jnp.vectorize(_dloss_dtheta, signature='(s)->(ntheta)')(range(n_samples)), 0)

    theta, losses = adam(loss, dloss_dtheta, pack(theta0), n_iter=n_iter)
    return unpack(theta, theta0), losses


