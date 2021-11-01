#!/usr/bin/env python

from models.linear_confounder_model import define_mechanisms

import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, jvp, vmap, jacfwd
from jax.config import config
config.update("jax_enable_x64", True)


class CausalInference:
    def __init__(self):
        self.f, self.finv, self.lpdf_u, self.draw_u = define_mechanisms()

    def fill(self, u: dict, v: dict, theta: dict, rvs: list):
        u0 = u.copy()
        v0 = v.copy()
        for rv in rvs:
            if rv in v0:
                u0[rv] = self.finv[rv](v0[rv], theta, v0)
            else:
                v0[rv] = self.f[rv](u0[rv], theta, v0)
        return u0, v0

    def test_f(self, theta: dict, rv):
        u0, v0 = self.fill({k: u(1) for k, u in self.draw_u.items()}, {}, theta, self.draw_u.keys())
        u_rv = u0[rv]
        del u0[rv]
        u0, _ = self.fill(u0, v0, theta, self.draw_u.keys())
        assert jnp.linalg.norm(u_rv - u0[rv]) < 1e-12

    def fy(self, x: float, u: dict, theta: dict):
        return self.fill(u, {'X': x}, theta, list(u.keys()))[1]['Y']

    def dlpdf_du(self, rv, u: dict):
        return vmap(lambda _u: jacfwd(self.lpdf_u[rv])(_u))(u[rv])

    def dfv_du(self, rv, x: float, u: dict, o: dict, theta: dict):
        return jnp.vectorize(jacfwd(lambda a: self.f[rv](u=a, theta=theta, parents={**o, 'X': x})), signature='(i)->(s,nv,nu)')(u[rv])

    def dfy_du(self, rv, x: float, u: dict, theta: dict):
        uo = {k: _u for k, _u in u.items() if _u.ndim == 1}
        ul = {k: _u for k, _u in u.items() if _u.ndim > 1}

        def _fy(_ul, _a):
            u_new = {k: _ul[k] if k in _ul else uo[k] for k in u}
            u_new[rv] = _a
            return self.fy(x=x, u=u_new, theta=theta)

        if rv in ul:
            return vmap(lambda _ul, a: jacfwd(lambda _a: _fy(_ul, _a))(a))(ul, u[rv])
        return vmap(lambda _ul: jacfwd(lambda _a: _fy(_ul, _a))(u[rv]))(ul)

    def dfv_dx(self, rv, x: float, u: dict, o: dict, theta: dict):
        return jacfwd(lambda a: self.fill(u, {'X': a, **{k: v for k, v in o.items() if k != rv}}, theta, list(u.keys()))[1][rv])(x)

    def dfinv_dv(self, rv, v: dict, theta: dict):
        o = {k: _v for k, _v in v.items() if _v.ndim == 1}
        l = {k: _v for k, _v in v.items() if _v.ndim > 1}
        if len(l) > 0:
            return vmap(lambda _l: jacfwd(lambda _a: self.finv[rv](v=_a, theta=theta, parents={**o, **{k: _v for k, _v in _l.items()}}))(v[rv]))(l)
        return jacfwd(lambda a: self.finv[rv](v=a, theta=theta, parents=v))(v[rv])

    def dfv_dtheta(self, rv, key, x: float, u: dict, o: dict, theta: dict):
        def _replace_theta(a):
            new_theta = {k: v for k, v in theta.items() if k != key}
            new_theta[key] = a
            return new_theta
        return jnp.vectorize(jacfwd(lambda a: self.f[rv](u=u[rv], theta=_replace_theta(a), parents={**o, 'X': x})))(theta[key])

    def sample_u(self, x: float, o: dict, theta: dict, size: int):
        u, v = self.fill({k: u(size) for k, u in self.draw_u.items()}, {**o, 'X': x}, theta, self.draw_u.keys())

        def log_weight(i: int):
            ui = {k: _u[i] if _u.ndim > 1 else _u for k, _u in u.items()}
            vi = {k: _v[i] if _v.ndim > 1 else _v for k, _v in v.items()}
            lp = lambda rv: self.lpdf_u[rv](ui[rv]) + jnp.sum(jnp.log(jnp.abs(jnp.diag(self.dfinv_dv(rv, vi, theta)))))

            lw = lp('X')
            for rv in o:
                lw += lp(rv)

            return lw

        log_weights = jnp.vectorize(log_weight)(range(size))
        return u, v, jnp.exp(log_weights - jsp.special.logsumexp(log_weights))

    def causal_effect(self, x: float, o: dict, theta: dict, size: int = 100000):
        u, _, w = self.sample_u(x, o, theta, size)

        def _causal_effect(i: int):
            ui = {k: _u[i] if _u.shape[0] == size else _u for k, _u in u.items()}
            return jacfwd(lambda a: self.fill(ui, {'X': a}, theta, list(u.keys()))[1]['Y'])(x) * w[i, None, None]
        return jnp.sum(jnp.vectorize(_causal_effect, signature='(s)->(s,ny,nx)')(range(size)), 0)

    def causal_bias(self, x: float, o: dict, theta: dict, size: int = 100000):
        u, v, w = self.sample_u(x, o, theta, size)
        y = self.fy(x, u, theta)
        ru = y - jnp.sum(y * w[:, None], 0, keepdims=True)

        def _causal_bias(i: int):
            ui = {k: _u[i] if _u.ndim > 1 else _u for k, _u in u.items()}
            vi = {k: _v[i] if _v.ndim > 1 else _v for k, _v in v.items()}

            def cb(rv):
                dfy_du = self.dfy_du(rv, x, ui, theta)
                dlpdf_du = self.dlpdf_du(rv, ui)[:, None, :] if ui[rv].ndim > 1 else self.dlpdf_du(rv, ui)[None, None, :]
                dfinv_dv = self.dfinv_dv(rv, vi, theta)
                return jnp.matmul(dfy_du + ru[i, :, None] * dlpdf_du, dfinv_dv)

            b = cb('X')
            for rv in o:
                b -= jnp.matmul(cb(rv), self.dfv_dx(rv, x, ui, o, theta))

            return b * w[i, None, None]
        return jnp.sum(jnp.vectorize(_causal_bias, signature='(s)->(s,ny,nx)')(range(size)), 0)


if __name__ == '__main__':
    # print(func())

    theta0 = {'V1->X': 1., 'X->Y': 2., 'V1->Y': 3.}
    # theta0 = {'V1->X': jnp.array([1., 2.]), 'X->Y': jnp.array([3., 4.]), 'V1->Y': jnp.array([5., 6.])}
    # theta0 = {'X->V1': jnp.array([1., 2.]), 'X->Y': jnp.array([3., 4.]), 'V1->Y': jnp.array([5., 6.])}
    # theta0 = {'X->Y': jnp.array([1., 2.]), 'X->V1': jnp.array([3., 4.]), 'Y->V1': jnp.array([5., 6.])}
    alpha, beta, gamma = np.array(list(theta0.values()))
    print('true bias', gamma * alpha / (1 + alpha ** 2))
    # print('true bias', -gamma * alpha)
    # print('true bias', -gamma * (beta + gamma * alpha) / (1 + gamma ** 2))

    sem = StructuralEquationModel()
    u, v = sem.fill({k: u(1) for k, u in sem.draw_u.items()}, {}, theta0, sem.draw_u.keys())
    x = v['X'].squeeze()  # jnp.array([1., 2.])
    o = {}#{'V1': v['V1'].squeeze()}  # {"V1": jnp.array([2., 3.])}

    size = 1000000
    print("Causal effect: {}".format(sem.causal_effect(x, o, theta0, size=size)))
    print("Causal bias: {}".format(sem.causal_bias(x, o, theta0, size=size)))
    #
    # print(sem.dfv_dtheta('X', 'V1->X', x, u, v, theta0))
