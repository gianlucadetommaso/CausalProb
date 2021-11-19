#!/usr/bin/env python

import numpy as np
import jax.numpy as jnp
from jax import random
from jax.config import config
config.update("jax_enable_x64", True)


def lp_standard_normal(x: jnp.array, theta: dict):
    z = jnp.sum(x ** 2, -1) if x.ndim > 1 else jnp.sum(x ** 2)
    return -0.5 * z #- x.shape[0] * jnp.log(2 * jnp.pi))


def define_model(dim=2):
    f, finv, lpu, draw_u, init_params, ldij = dict(), dict(), dict(), dict(), dict(), dict()

    # X
    def _f(u: jnp.array, theta: dict, parents: dict):
        return u
    f['X'] = _f

    def _finv(v: jnp.array, theta: dict, parents: dict):
        return v
    finv['X'] = _finv

    lpu['X'] = lambda u, theta: lp_standard_normal(u, theta)
    draw_u['X'] = lambda size, theta: jnp.array(np.random.normal(size=(size, dim)))
    ldij['X'] = lambda v, theta, parents: 0.

    # Y
    def _f(u: jnp.array, theta: dict, parents: dict):
        x = parents['X']
        return theta['X->Y'] * x + u

    f['Y'] = _f

    def _finv(v: jnp.array, theta: dict, parents: dict):
        x = parents['X']
        return v - theta['X->Y'] * x

    finv['Y'] = _finv

    lpu['Y'] = lambda u, theta: lp_standard_normal(u, theta)
    draw_u['Y'] = lambda size, theta: jnp.array(np.random.normal(size=(size, dim)))
    ldij['Y'] = lambda v, theta, parents: 0.
    init_params['X->Y'] = lambda seed=0, scale=1: scale * random.normal(random.PRNGKey(seed), (dim,))

    return dict(f=f, finv=finv, lpu=lpu, draw_u=draw_u, init_params=init_params)
