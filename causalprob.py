#!/usr/bin/env python

import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, jvp, vmap, jacfwd
from jax.config import config
config.update("jax_enable_x64", True)


class CausalProb:
    """
    Given a user-specified probabilistic model, this class contains the necessary tools to infer marginal causal
    effect and marginal causal bias.
    """
    def __init__(self, model):
        self.f, self.finv, self.lpu, self.draw_u = model

    def fill(self, u: dict, v: dict, theta: dict, rvs: list) -> tuple:
        """
        For each observed random variable of V in `v`, this logic replaces the respective samples of U in `u` with 
        values that are coherent with the structural equation model. Furthermore, starting from `u`, it fills all
        values of V for all the random variables that were not already in `v`.
        Importantly, this logic allows for automatic differentiation over the structural equation model, taking into
        account that observed variables in `v` block the differentiation flow on the respective path. As an example, 
        given the graphical structure X -> V -> Y, if V is in `v` the gradient of Y with respect to X should be zero.
        
        Parameters
        ----------
        u: dict
            Samples of random variables U.
        v: dict
            Observations of some random variables V.
        theta: dict
            Model parameters.
        rvs: list
            Random variables for which to operate the logic. Please mind the these need to be ordered from parents to
            children.
            
        Returns
        -------
        u0, v0: tuple
            It returns samples `u0` of U that are coherent with the observations `v` of V, and values `v0` for all 
            variables V starting from the samples `u0`. 
        """
        u0 = u.copy()
        v0 = v.copy()
        for rv in rvs:
            if rv in v0:
                u0[rv] = self.finv[rv](v0[rv], theta, v0)
            else:
                v0[rv] = self.f[rv](u0[rv], theta, v0)
        return u0, v0

    def test_f(self, rv: str, theta: dict) -> None:
        """
        This method tests that the filling logic is correct for the random variable `rv`.
        
        Parameters
        ----------
        rv: str
            Random variable for which to operate the test.
        theta: dict
            Model parameters.
        """
        u0, v0 = self.fill({k: u(1) for k, u in self.draw_u.items()}, {}, theta, self.draw_u.keys())
        u_rv = u0[rv]
        del u0[rv]
        u0, _ = self.fill(u0, v0, theta, self.draw_u.keys())
        assert jnp.linalg.norm(u_rv - u0[rv]) < 1e-12

    def fy(self, u: dict, x: jnp.array, theta: dict) -> jnp.array:
        """
        This method computes samples of Y starting from samples in `u`, given the observation `x` of the treatment X.
        Importantly, no other variable is observed in this method.
        
        Parameters
        ----------
        u: dict
            Samples of random variables U.
        x: jnp.array
            Observation of treatment X.
        theta: dict
            Model parameters.
            
        Returns
        -------
        y: jnp.array
            It returns values for Y.
        """
        return self.fill(u, {'X': x}, theta, list(u.keys()))[1]['Y']

    def dlpu_du(self, rv, u: dict) -> jnp.array:
        """
        This method differentiates the log-probability density of U_{`rv`} with respect to U_{`rv`}.

        Parameters
        ----------
        rv: str
            It specifies random variable to differentiate upon.
        u: dict
            Values of U where the gradient should be evaluated.

        Returns
        -------
        dlpu_du: jnp.array
            It returns the gradient of the log-probability density of U_{`rv`} with respect to U_{`rv`} evaluated at the
            values in `u`.
        """
        return vmap(lambda _u: grad(self.lpu[rv])(_u))(u[rv])

    def dfy_du(self, rv: str, u: dict, x: jnp.array, theta: dict) -> jnp.array:
        """
        This method differentiates the structural equation f_Y with respect to U_{`rv`} when only the value `x` of the
        treatment X is observed but no other variable.

        Parameters
        ----------
        rv: str
            It specifies random variable to differentiate upon.
        u: dict
            Values of U where the gradient should be evaluated.
        x: jnp.array
            Observation of treatment X.
        theta: dict
            Model parameters.

        Returns
        -------
        dfy_du: jnp.array
            It returns the Jacobian of the structural equation f_Y with respect to U_{`rv`} evaluated at the values in
            `u`, when only the value `x` of the treatment X is observed but no other variable.
        """
        uo = {k: _u for k, _u in u.items() if _u.ndim == 1}
        ul = {k: _u for k, _u in u.items() if _u.ndim > 1}

        def _fy(_ul, _a):
            u_new = {k: _ul[k] if k in _ul else uo[k] for k in u}
            u_new[rv] = _a
            return self.fy(u=u_new, x=x, theta=theta)

        if rv in ul:
            return vmap(lambda _ul, a: jacfwd(lambda _a: _fy(_ul, _a))(a))(ul, u[rv])
        return vmap(lambda _ul: jacfwd(lambda _a: _fy(_ul, _a))(u[rv]))(ul)

    def dfv_dx(self, rv: str, u: dict, x: jnp.array, o: dict, theta: dict) -> jnp.array:
        """
        This method differentiates the structural equation f_{V_{`rv`}} with respect to X when both values `x` for the
        treatment and `o` for the observed variables O are observed.

        Parameters
        ----------
        rv: str
            It specifies the structural equation to differentiate.
        u: dict
            Values of U where the gradient should be evaluated.
        x: jnp.array
            Observation of treatment X.
        o: dict
            Values of observed random variables O.
        theta: dict
            Model parameters.

        Returns
        -------
        dfv_dx: jnp.array
            It returns the Jacobian of the structural equation f_{V_{`rv`}} with respect to X evaluated at the values in
             `u`when both values `x` for the treatment and `o` for the observed variables O are observed.
        """
        return jacfwd(lambda a: self.fill(u, {'X': a, **{k: v for k, v in o.items() if k != rv}}, theta, list(u.keys()))[1][rv])(x)

    def dfinvv_dv(self, rv: str, v: dict, theta: dict) -> jnp.array:
        """
        This method differentiates the structural equation f^{-1}_{V_{`rv`}} with respect to V_{`rv`} when variables V
        in `v` are observed.

        Parameters
        ----------
        rv: str
            It specifies random variable to differentiate upon.
        v: dict
            Values of observed random variables V. This can include both X and O.
        theta: dict
            Model parameters.

        Returns
        -------
        dfinvv_dv: jnp.array
            It returns the Jacobian of the structural equation f^{-1}_{V_{`rv`}} with respect to V_{`rv`} evaluated at
            the values in `v`.
        """
        o = {k: _v for k, _v in v.items() if _v.ndim == 1}
        l = {k: _v for k, _v in v.items() if _v.ndim > 1}
        if len(l) > 0:
            return vmap(lambda _l: jacfwd(lambda _a: self.finv[rv](v=_a, theta=theta, parents={**o, **{k: _v for k, _v in _l.items()}}))(v[rv]))(l)
        return jacfwd(lambda a: self.finv[rv](v=a, theta=theta, parents=v))(v[rv])

    def dfv_dtheta(self, rv: str, key: str, u: dict, x: float, o: dict, theta: dict) -> jnp.array:
        """
        This method differentiates the structural equation f_{V_{`rv`}} with respect to \theta_{`key`} when values `x`
        of X and `o` of O are observed.

        Parameters
        ----------
        rv: str
            It specifies the structural equation to differentiate.
        key: str
            It specifies the parameter to differentiate upon.
        u: dict
            Values of U variables where the gradient should be evaluated.
        x: jnp.array
            Observation of treatment X.
        o: dict
            Values of observed random variables O.
        theta: dict
            Model parameters.

        Returns
        -------
        dfv_dtheta: jnp.array
            It returns the Jacobian of the structural equation f_{V_{`rv`}} with respect to \theta_{`key`} evaluated at
            `u` and `theta`, when values `x` of X and `o` of O are observed.
        """
        return jacfwd(lambda a: self.fill(u, {'X': x, **{k: _o for k, _o in o.items() if k != rv}}, {**theta, key: a}, list(u.keys()))[1][rv])(theta[key])

    def llkd(self, u: dict, x: jnp.array, o: dict, theta: dict, v: dict = None) -> jnp.array:
        """
        It evaluate the log-likelihood at `u` given values `x` of X and `o` of O.

        u: dict
            Values of U variables where the likelihood should be evaluated.
        x: jnp.array
            Observation of treatment X.
        o: dict
            Values of observed random variables O.
        theta: dict
            Model parameters.
        v: dict
            Values of V variables. If these are not passed, they are computed from the values `u` of U.

        Returns
        -------
        llkd: jnp.array
            log-likelihood evaluation.
        """
        if v is None:
            u, v = self.fill(u, {'X': x, **o}, theta, list(u.keys()))

        def _lp(rv):
            print(rv, self.dfinvv_dv(rv, v, theta))
            return self.lpu[rv](u[rv]) + jnp.sum(jnp.log(jnp.abs(jnp.diag(self.dfinvv_dv(rv, v, theta)))))

        llkd = _lp('X')
        for rv in o:
            llkd += _lp(rv)
        return llkd

    def dllkd_dtheta(self, key: str, u: dict, x: jnp.array, o: dict, theta: dict, v: dict = None) -> jnp.array:
        """
        It evaluates the gradient of the log-likelihood with respect to \theta_{`key`}.

        Parameters
        ----------
        key: str
            It specifies the parameter to differentiate upon.
        u: dict
            Values of U variables where the likelihood should be evaluated.
        x: jnp.array
            Observation of treatment X.
        o: dict
            Values of observed random variables O.
        theta: dict
            Model parameters.
        v: dict
            Values of V variables. If these are not passed, they are computed from the values `u` of U.

        Returns
        -------
        dllkd_dtheta: jnp.array
            Gradient evaluation of the log-likelihood with respect to \theta_{`key`}.
        """
        return grad(lambda a: self.llkd(u, x, o, {**theta, key: a}, v))(theta[key])

    def sample_u(self, x: jnp.array, o: dict, theta: dict, n_samples: int) -> tuple:
        """
        It generate coherent samples from the prior for U and V given the observations `x` of X and `o` of O, and
        computes the importance weights to get samples from the posterior.

        Parameters
        ----------
        x: jnp.array
            Observation of treatment X.
        o: dict
            Values of observed random variables O.
        theta: dict
            Model parameters.
        n_samples: int
            Number of samples to generate.

        Returns
        -------
        u, v, w: tuple
            It returns coherent samples from the prior `u` for U and `v` for V given the observations `x` of X and `o`
            of O, as well as the the importance weights `w` to get samples from the posterior.
        """
        u, v = self.fill({k: u(n_samples) for k, u in self.draw_u.items()}, {**o, 'X': x}, theta, self.draw_u.keys())

        def log_weight(i: int):
            ui = {k: _u[i] if _u.ndim > 1 else _u for k, _u in u.items()}
            vi = {k: _v[i] if _v.ndim > 1 else _v for k, _v in v.items()}
            return self.llkd(ui, x, o, theta, vi)

        log_weights = jnp.vectorize(log_weight)(range(n_samples))
        return u, v, jnp.exp(log_weights - jsp.special.logsumexp(log_weights))

    def causal_effect(self, x: jnp.array, o: dict, theta: dict, n_samples: int = 100000) -> jnp.array:
        """
        It estimates the marginal causal effect of the treatment X on the outcome Y given values `x` for X and `o` for
        the observed variables O.

        Parameters
        ----------
        x: jnp.array
            Observation of treatment X.
        o: dict
            Values of observed random variables O.
        theta: dict
            Model parameters.
        n_samples: int
            Number of samples to generate.

        Returns
        -------
        c: jnp.array
            It returns an estimate of the marginal causal effect.
        """
        u, v, w = self.sample_u(x, o, theta, n_samples)

        def _causal_effect(i: int):
            ui = {k: _u[i] if _u.ndim > 1 else _u for k, _u in u.items()}
            return jacfwd(lambda a: self.fill(ui, {'X': a}, theta, list(u.keys()))[1]['Y'])(x) * w[:, None, None]

        return jnp.sum(jnp.vectorize(_causal_effect, signature='(s)->(s,ny,nx)')(range(n_samples)), 0)

    def causal_bias(self, x: jnp.array, o: dict, theta: dict, n_samples: int = 100000) -> jnp.array:
        """
        It estimates the marginal causal bias of the treatment X on the outcome Y given values `x` for X and `o` for
        the observed variables O.

        Parameters
        ----------
        x: jnp.array
            Observation of treatment X.
        o: dict
            Values of observed random variables O.
        theta: dict
            Model parameters.
        n_samples: int
            Number of samples to generate.

        Returns
        -------
        b: jnp.array
            It returns an estimate of the marginal causal bias.
        """
        u, v, w = self.sample_u(x, o, theta, n_samples)
        y = self.fy(u, x, theta)
        ru = y - jnp.sum(y * w[:, None], 0, keepdims=True)

        def _causal_bias(i: int):
            ui = {k: _u[i] if _u.ndim > 1 else _u for k, _u in u.items()}
            vi = {k: _v[i] if _v.ndim > 1 else _v for k, _v in v.items()}

            def cb(rv):
                dfy_du = self.dfy_du(rv, ui, x, theta)
                dlpu_du = self.dlpu_du(rv, ui)[:, None, :] if ui[rv].ndim > 1 else self.dlpu_du(rv, ui)[None, None, :]
                dfinvv_dv = self.dfinvv_dv(rv, vi, theta)
                return jnp.matmul(dfy_du + ru[i, :, None] * dlpu_du, dfinvv_dv)

            b = cb('X')
            for rv in o:
                b -= jnp.matmul(cb(rv), self.dfv_dx(rv, ui, x, o, theta))
            return b * w[i, None, None]
        return jnp.sum(jnp.vectorize(_causal_bias, signature='(s)->(s,ny,nx)')(range(n_samples)), 0)


if __name__ == '__main__':
    # theta0 = {'V1->X': jnp.array([1.]), 'X->Y': jnp.array([2.]), 'V1->Y': jnp.array([3.])}
    # theta0 = {'X->V1': jnp.array([1.]), 'X->Y': jnp.array([2.]), 'V1->Y': jnp.array([3.])}
    # theta0 = {'X->Y': jnp.array([1.]), 'X->V1': jnp.array([2.]), 'Y->V1': jnp.array([3.])}

    theta0 = {'V1->X': jnp.array([1., 2.]), 'X->Y': jnp.array([3., 4.]), 'V1->Y': jnp.array([5., 6.])}
    # theta0 = {'X->V1': jnp.array([1., 2.]), 'X->Y': jnp.array([3., 4.]), 'V1->Y': jnp.array([5., 6.])}
    # theta0 = {'X->Y': jnp.array([1., 2.]), 'X->V1': jnp.array([3., 4.]), 'Y->V1': jnp.array([5., 6.])}
    alpha, beta, gamma = np.array(list(theta0.values()))
    print('true bias', gamma * alpha / (1 + alpha ** 2))
    # print('true bias', -gamma * alpha)
    # print('true bias', -gamma * (beta + gamma * alpha) / (1 + gamma ** 2))

    from models.linear_confounder_model import define_model
    cp = CausalProb(model=define_model())
    u, v = cp.fill({k: u(1) for k, u in cp.draw_u.items()}, {}, theta0, cp.draw_u.keys())
    x = v['X'].squeeze(0)  # jnp.array([1., 2.])
    o = {'V1': v['V1'].squeeze(0)}  # {"V1": jnp.array([2., 3.])}

    n_samples = 1000000
    print("Causal effect: {}".format(cp.causal_effect(x, o, theta0, n_samples=n_samples)))
    print("Causal bias: {}".format(cp.causal_bias(x, o, theta0, n_samples=n_samples)))
    #

    # check derivatives wrt theta
    # print('x', x)
    # print('o', o)
    #
    # u, v = sem.fill({k: u(3) for k, u in sem.draw_u.items()}, {}, theta0, sem.draw_u.keys())
    # for rv in v:
    #     for key in theta0:
    #         print('df[{}]_dtheta[{}]'.format(rv, key), sem.dfv_dtheta(rv, key, u, x, o, theta0))
