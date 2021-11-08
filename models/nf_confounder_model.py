#!/usr/bin/env python

from models.normalizing_flow.architectures import RealNVP

import jax.numpy as jnp
from jax.config import config
from jax.experimental import stax  # neural network library
from jax.experimental.stax import Dense, Relu  # neural network layers
from jax import random
config.update("jax_enable_x64", True)


class NeuralNet:
    def __init__(self, dim: int, seed: int = 0):
        self.dim = dim
        self.net_init, self.net_apply = stax.serial(Dense(8), Relu, Dense(8), Relu, Dense(self.dim))
        self.seed = seed

    def shift_and_log_scale_fn(self, u: jnp.array, params: jnp.array) -> list:
        s = self.net_apply(params, u)
        return jnp.split(s, 2, axis=-1)

    def init_params(self, seed: int = 0) -> tuple:
        in_shape = (-1, self.dim)
        out_shape, layer_params = self.net_init(random.PRNGKey(self.seed + seed), in_shape)
        return out_shape, layer_params


def define_model(dim=2):
    f, finv, lpu, draw_u, init_params = dict(), dict(), dict(), dict(), dict()

    nf = RealNVP(dim=dim, seed=42)
    nn2 = NeuralNet(dim=2, seed=43)
    nn4 = NeuralNet(dim=4, seed=44)

    # V
    def _f(u: jnp.array, theta: dict, parents: dict):
        return nf.forward(u, theta['V1'])
    f['V1'] = _f

    def _finv(v: jnp.array, theta: dict, parents: dict):
        return nf.backward(v, theta['V1'])[0]
    finv['V1'] = _finv

    lpu['V1'] = lambda u, theta: nf.evaluate_base_logpdf(u)
    draw_u['V1'] = lambda size, theta: nf.sample_base(size)
    init_params['V1'] = lambda seed: nf.init_all_params(seed)

    # X
    def _f(u: jnp.array, theta: dict, parents: dict):
        v1 = parents['V1']
        shift, log_scale = nn2.shift_and_log_scale_fn(v1, theta['V1->X'])
        return jnp.exp(log_scale) * nf.forward(u, theta['U_X->X']) + shift
    f['X'] = _f

    def _finv(v: jnp.array, theta: dict, parents: dict):
        v1 = parents['V1']
        shift, log_scale = nn2.shift_and_log_scale_fn(v1, theta['V1->X'])
        return nf.backward(jnp.exp(-log_scale) * (v - shift), theta['U_X->X'])[0]
    finv['X'] = _finv

    lpu['X'] = lambda u, theta: nf.evaluate_base_logpdf(u)
    draw_u['X'] = lambda size, theta: nf.sample_base(size)
    init_params['V1->X'] = lambda seed: nn2.init_params(seed)[1]
    init_params['U_X->X'] = lambda seed: nf.init_all_params(seed)

    # Y
    def _f(u: jnp.array, theta: dict, parents: dict):
        v1, x = parents['V1'], parents['X']
        shift, log_scale = nn4.shift_and_log_scale_fn(jnp.concatenate((v1, x), axis=-1), theta['V1--X->Y'])
        return jnp.exp(log_scale) * nf.forward(u, theta['U_Y->Y']) + shift
    f['Y'] = _f

    def _finv(v: jnp.array, theta: dict, parents: dict):
        v1, x = parents['V1'], parents['X']
        shift, log_scale = nn4.shift_and_log_scale_fn(jnp.concatenate((v1, x), axis=-1), theta['V1--X->Y'])
        return nf.backward(jnp.exp(-log_scale) * (v - shift), theta['U_Y->Y'])[0]
    finv['Y'] = _finv

    lpu['Y'] = lambda u, theta: nf.evaluate_base_logpdf(u)
    draw_u['Y'] = lambda size, theta: nf.sample_base(size)
    init_params['V1--X->Y'] = lambda seed: nn4.init_params(seed)[1]
    init_params['U_Y->Y'] = lambda seed: nf.init_all_params(seed)

    return dict(f=f, finv=finv, lpu=lpu, draw_u=draw_u, init_params=init_params)
