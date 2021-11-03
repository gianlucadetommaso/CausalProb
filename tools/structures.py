#!/usr/bin/env python

import jax.numpy as jnp
import numpy as np


def pack(d: dict) -> jnp.array:
    return jnp.concatenate([jnp.ravel(v) for v in d.values()])


def unpack(a: jnp.array, d: dict) -> dict:
    keys, values = [], []
    for k, v in d.items():
        keys.append(k), values.append(v)

    shapes = [v.shape for v in values]
    cum_sizes = np.cumsum([0] + [v.size for v in values]).tolist()
    return {keys[i]: a[cum_sizes[i]:cum_sizes[i+1]].reshape(shapes[i]) for i in range(len(d))}


if __name__ == '__main__':
    d = {'a': jnp.array(np.arange(24).reshape(2,3,4)), 'b': jnp.array([]), 'c': jnp.array([5., 6.])}
    a = jnp.array(np.random.normal(size=sum([v.size for v in d.values()])))

    print('pack', pack(d))
    print('unpack', a, unpack(a, d))
