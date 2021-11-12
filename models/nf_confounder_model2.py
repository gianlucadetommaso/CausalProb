#!/usr/bin/env python

from models.normalizing_flow.architectures import RealNVP

import jax.numpy as jnp
from jax.config import config
from jax.experimental import stax  # neural network library
from jax.experimental.stax import Dense, Relu, normal  # neural network layers
from jax import random
config.update("jax_enable_x64", True)


class NeuralNet:
    def __init__(self, dim: int, seed: int = 0):
        self.dim = dim
        self.net_init, self.net_apply = stax.serial(Dense(8, W_init=normal()), Relu, Dense(8, W_init=normal()), Relu, Dense(self.dim, W_init=normal()))
        self.seed = seed

    def shift_and_log_scale_fn(self, u: jnp.array, params: jnp.array) -> list:
        s = self.net_apply(params, u)
        return jnp.split(s, 2, axis=-1)

    def init_params(self, seed: int = 0) -> tuple:
        in_shape = (-1, self.dim)
        out_shape, layer_params = self.net_init(random.PRNGKey(self.seed + seed), in_shape)
        return out_shape, layer_params


def define_model(dim=2):
    f, finv, lpu, draw_u, init_params, ldij = dict(), dict(), dict(), dict(), dict(), dict()

    nf = RealNVP(dim=dim, seed=42)
    nn2 = NeuralNet(dim=2, seed=43)
    nn4 = NeuralNet(dim=4, seed=44)

    # V
    def _f_V1(u: jnp.array, theta: dict, parents: dict):
        return nf.forward(u, theta['V1'])
    f['V1'] = _f_V1

    def _finv_V1(v: jnp.array, theta: dict, parents: dict):
        return nf.backward(v, theta['V1'])
    finv['V1'] = lambda v, theta, parents: _finv_V1(v, theta, parents)[0]
    ldij['V1'] = lambda v, theta, parents: jnp.sum(_finv_V1(v, theta, parents)[1], -1)

    lpu['V1'] = lambda u, theta: nf.evaluate_base_logpdf(u)
    draw_u['V1'] = lambda size, theta: nf.sample_base(size)
    init_params['V1'] = lambda seed: nf.init_all_params(seed)

    # X
    def _f_X(u: jnp.array, theta: dict, parents: dict):
        v1 = parents['V1']
        return nf.forward(v1, theta['V1->X']) + nf.forward(u, theta['U_X->X'])
    f['X'] = _f_X

    def _finv_X(v: jnp.array, theta: dict, parents: dict):
        v1 = parents['V1']
        return nf.backward(v - nf.forward(v1, theta['V1->X']), theta['U_X->X'])

    finv['X'] = lambda v, theta, parents: _finv_X(v, theta, parents)[0]
    ldij['X'] = lambda v, theta, parents: jnp.sum(_finv_X(v, theta, parents)[1], -1)

    lpu['X'] = lambda u, theta: nf.evaluate_base_logpdf(u)
    draw_u['X'] = lambda size, theta: nf.sample_base(size)
    init_params['V1->X'] = lambda seed: nn2.init_params(seed)[1]
    init_params['U_X->X'] = lambda seed: nf.init_all_params(seed)

    # Y
    def _f_Y(u: jnp.array, theta: dict, parents: dict):
        v1, x = parents['V1'], parents['X']
        return nf.forward(v1, theta['V1->Y']) + nf.forward(x, theta['X->Y']) + nf.forward(u, theta['U_Y->Y'])
    f['Y'] = _f_Y

    def _finv_Y(v: jnp.array, theta: dict, parents: dict):
        v1, x = parents['V1'], parents['X']
        return nf.backward(v - nf.forward(v1, theta['V1->Y']) - nf.forward(x, theta['X->Y']), theta['U_Y->Y'])
    finv['Y'] = lambda v, theta, parents: _finv_Y(v, theta, parents)[0]
    ldij['Y'] = lambda v, theta, parents: jnp.sum(_finv_Y(v, theta, parents)[1], -1)

    lpu['Y'] = lambda u, theta: nf.evaluate_base_logpdf(u)
    draw_u['Y'] = lambda size, theta: nf.sample_base(size)
    init_params['V1--X->Y'] = lambda seed: nn4.init_params(seed)[1]
    init_params['U_Y->Y'] = lambda seed: nf.init_all_params(seed)

    return dict(f=f, finv=finv, lpu=lpu, draw_u=draw_u, init_params=init_params, ldij=ldij)
