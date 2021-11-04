#!/usr/bin/env python

from causalprob import CausalProb
from inference.model_training import train
from tools.structures import unpack

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dim = 1
    from models.linear_confounder_model import define_model
    model = define_model(dim=dim)

    true_theta = {'V1->X': jnp.array([1.]), 'X->Y': jnp.array([2.]), 'V1->Y': jnp.array([3.])}
    # true_theta = {'V1->X': jnp.array([1., 2.]), 'X->Y': jnp.array([3., 4.]), 'V1->Y': jnp.array([5., 6.])}
    cp = CausalProb(model=model)
    u, v = cp.fill({k: u(50) for k, u in cp.draw_u.items()}, {}, true_theta, cp.draw_u.keys())
    x = v['X']
    o = {'V1': v['V1']}
    y = v['Y']

    theta0 = unpack(jnp.array(np.random.normal(size=3 * dim)), true_theta)
    theta, losses = train(model, x, y, o, theta0)
    print('optimal theta parameters:', theta)
    plt.plot(losses)
    plt.show()

    print('x', x)
    print('y', y)
    print('v1', o['V1'])

    _, est_v = cp.fill({k: u(100000) for k, u in cp.draw_u.items()}, {}, true_theta, cp.draw_u.keys())
    for rv in ['X', 'Y', 'V1']:
        print('est_{}'.format(rv), est_v[rv].mean(0))
