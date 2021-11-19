#!/usr/bin/env python

from models.normalizing_flow.architectures import RealNVP

import jax.numpy as jnp
from jax.config import config
from jax.experimental import stax  # neural network library
from jax.experimental.stax import Dense, Relu, normal  # neural network layers
from jax import random
config.update("jax_enable_x64", True)


class NeuralNet:
    def __init__(self, input_dim, output_dim: int, seed: int = 0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_init, self.net_apply = stax.serial(Dense(8, W_init=normal()),
                                                    Relu,
                                                    Dense(8, W_init=normal()),
                                                    Relu,
                                                    Dense(self.output_dim, W_init=normal()))
        self.seed = seed

    def shift_and_log_scale_fn(self, u: jnp.array, params: jnp.array) -> list:
        s = self.net_apply(params, u)
        return jnp.split(s, 2, axis=-1)

    def init_params(self, seed: int = 0) -> tuple:
        in_shape = (-1, self.input_dim)
        out_shape, layer_params = self.net_init(random.PRNGKey(self.seed + seed), in_shape)
        return out_shape, layer_params


def define_model(dim=2):
    f, finv, lpu, draw_u, init_params, ldij, dlpu_du, dfy_du = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()

    nf = RealNVP(dim=dim, seed=42)
    nn = NeuralNet(input_dim=dim, output_dim=2 * dim, seed=43)

    # V
    def _f_X(u: jnp.array, theta: dict, parents: dict):
        return nf.forward(u, theta['U_X->X'])
    f['X'] = _f_X

    def _finv_X(v: jnp.array, theta: dict, parents: dict):
        return nf.backward(v, theta['U_X->X'])[0]
    finv['X'] = _finv_X

    def _ldij_X(v: jnp.array, theta: dict, parents: dict):
        return nf.backward(v, theta['U_X->X'])[1]
    ldij['X'] = _ldij_X

    lpu['X'] = lambda u, theta: nf.evaluate_base_logpdf(u)
    draw_u['X'] = lambda size, theta, seed=0: nf.sample_base(size, seed=seed)
    init_params['U_X->X'] = lambda seed: nf.init_all_params(seed)
    dlpu_du['X'] = lambda u, theta: -u

    # X
    def _f_Y(u: jnp.array, theta: dict, parents: dict):
        x = parents['X']
        shift, log_scale = nn.shift_and_log_scale_fn(x, theta['X->Y'])
        return jnp.exp(log_scale) * nf.forward(u, theta['U_Y->Y']) + shift
    f['Y'] = _f_Y

    def _finv_Y(v: jnp.array, theta: dict, parents: dict):
        x = parents['X']
        shift, log_scale = nn.shift_and_log_scale_fn(x, theta['X->Y'])
        return nf.backward(jnp.exp(-log_scale) * (v - shift), theta['U_Y->Y'])[0]
    finv['Y'] = _finv_Y

    def _ldij_Y(v: jnp.array, theta: dict, parents: dict):
        x = parents['X']
        shift, log_scale = nn.shift_and_log_scale_fn(x, theta['X->Y'])
        return -jnp.sum(log_scale, -1) + nf.backward(jnp.exp(-log_scale) * (v - shift), theta['U_Y->Y'])[1]
    ldij['Y'] = _ldij_Y

    lpu['Y'] = lambda u, theta: nf.evaluate_base_logpdf(u)
    draw_u['Y'] = lambda size, theta, seed=0: nf.sample_base(size, seed=seed)
    init_params['X->Y'] = lambda seed: nn.init_params(seed)[1]
    init_params['U_Y->Y'] = lambda seed: nf.init_all_params(seed)
    dlpu_du['Y'] = lambda u, theta: -u
    dfy_du['Y'] = lambda u, x, theta: 0.

    return dict(f=f, finv=finv, lpu=lpu, draw_u=draw_u, init_params=init_params, ldij=ldij, dlpu_du=dlpu_du,
                dfy_du=dfy_du)
