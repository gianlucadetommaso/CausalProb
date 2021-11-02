#!/usr/bin/env python

import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jacfwd, jvp, vmap, grad
from jax.config import config
config.update("jax_enable_x64", True)


def lpdf_normal(x: jnp.array, mu=0, log_sigma=0):
    z = ((x - mu) / jnp.exp(log_sigma)) ** 2
    z = jnp.sum(z, -1) if z.ndim > 1 else jnp.sum(z)
    return -0.5 * (z - jnp.sum(log_sigma)) #- x.shape[0] * jnp.log(2 * jnp.pi))


def define_mechanisms(n=2):
    f, finv, lpdf_u, draw_u = dict(), dict(), dict(), dict()

    # V
    def _f(u: jnp.array, theta: dict, parents: dict):
        return u
    f['V1'] = _f

    def _finv(v: jnp.array, theta: dict, parents: dict):
        return v
    finv['V1'] = _finv

    lpdf_u['V1'] = lambda u: lpdf_normal(u)
    draw_u['V1'] = lambda size: jnp.array(np.random.normal(size=(size, n)))

    # X
    def _f(u: jnp.array, theta: dict, parents: dict):
        v1 = parents['V1']
        return theta['V1->X'] * v1 + u
    f['X'] = _f

    def _finv(v: jnp.array, theta: dict, parents: dict):
        v1 = parents['V1']
        return v - theta['V1->X'] * v1
    finv['X'] = _finv

    lpdf_u['X'] = lambda u: lpdf_normal(u)
    draw_u['X'] = lambda size: jnp.array(np.random.normal(size=(size, n)))

    # Y
    def _f(u: jnp.array, theta: dict, parents: dict):
        v1, x = parents['V1'], parents['X']
        return theta['X->Y'] * x + theta['V1->Y'] * v1 + u

    f['Y'] = _f

    def _finv(v: jnp.array, theta: dict, parents: dict):
        v1, x = parents['V1'], parents['X']
        return v - theta['X->Y'] * x - theta['V1->Y'] * v1

    finv['Y'] = _finv

    lpdf_u['Y'] = lambda u: lpdf_normal(u)
    draw_u['Y'] = lambda size: jnp.array(np.random.normal(size=(size, n)))

    return f, finv, lpdf_u, draw_u