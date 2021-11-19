#!/usr/bin/env python

from causalprob import CausalProb

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, jit, value_and_grad
from jax.tree_util import tree_flatten

import time


def train(model, x: jnp.array, y: jnp.array, o: jnp.array, theta0: jnp.array, n_samples: int = 1000, n_epoch: int = 100,
          batch_size: int = 100, loss_type: str = 'neg-evidence', step_size: float = 1e-4) -> tuple:
    """
    It trains the parameters of the model, given data `x`, `y`, `o`. Starting from an initial solution `theta0`, the
    algorithm uses the Adam optimizer to minimize with respect to the negative evidence (default) or the negative
    averaged log-likelihood.
    Currently, the number of observations for each observed variable must be the same.

    Parameters
    ----------
    x: jnp.array
        Observation of treatment X.
    y: jnp.array
        Observation of outcome Y.
    o: dict
        Observations of observed variables O.
    theta0: dict
        Initial solution of model parameters.
    n_samples: int
        Number of samples to approximate the loss function and its gradient.
    n_epoch: int
        Number of epochs to run the optimization for.
    batch_size: int
        Data batch size at every epoch.
    loss_type: str
        Type of loss function. It can either be:
            - `neg-evidence`: the evidence density function, namely -p(x,y,o);
            - `neg-avg-log-evidence`: the average logarithmic-evidence, namely -E_{u_L}[p(x,y,o|u_L)]
    step_size: float
        Initial step size in Adam optimizer.

    Returns
    -------
    theta, losses: tuple
        theta: jnp.array
            Estimated parameters minimizing the loss function.
        losses: list
            History of loss function evaluations at each iteration of the optimization.
    """
    allowed_losses = ['neg-evidence', 'neg-avg-log-evidence']
    if loss_type not in allowed_losses:
        raise Exception('loss_type={} not recognized. Please select within {}.'.format(loss_type, allowed_losses))

    oy = {**o, 'Y': y}
    all_n_obs = set()
    for v in oy.values():
        if v.ndim > 1:
            all_n_obs.add(v.shape[0])
    if len(all_n_obs) > 1:
        raise Exception('Each observed variable must have the same number of observations.')

    cp = CausalProb(model=model)

    # loss function
    # @jit
    # def loss(_theta: dict):
    #     u_prior = {k: u(n_samples, _theta) for k, u in cp.draw_u.items()}
    #
    #     def _loss(xj, oyj):
    #         u, v = cp.fill(u_prior, {**oyj, 'X': xj}, _theta, cp.draw_u.keys())
    #
    #         def __loss(i: int):
    #             ui = {k: _u[i] if _u.ndim > 1 else _u for k, _u in u.items()}
    #             vi = {k: _v[i] if _v.ndim > 1 else _v for k, _v in v.items()}
    #             llkd = cp.llkd(ui, xj, oyj, _theta, vi)
    #             return jnp.exp(llkd) if loss_type == 'neg-evidence' else llkd
    #         return -jnp.mean(jnp.vectorize(__loss)(range(n_samples)))
    #
    #     if x.ndim > 1:
    #         return jnp.sum(vmap(_loss, (0, {k: 0 for k in oy}))(x, oy))
    #     return _loss(x, oy)

    @jit
    def loss_fn(_theta: dict, x_batch, oy_batch):
        u_prior = {k: u(n_samples, _theta) for k, u in cp.draw_u.items()}

        def _loss_fn(xj, oyj):
            u, v = cp.fill(u_prior, {**oyj, 'X': xj}, _theta, cp.draw_u.keys())
            llkd = cp.llkd(u, xj, oyj, _theta, v)
            return -jnp.mean(llkd)

        if x.ndim > 1:
            return jnp.sum(vmap(_loss_fn, (0, {k: 0 for k in oy}))(x_batch, oy_batch))
        return _loss_fn(x, oy)

    # gradient of loss function
    def dloss_dtheta(_theta: jnp.array):
        u_prior = {k: u(n_samples, _theta) for k, u in cp.draw_u.items()}

        def _dloss_dtheta(xj, oyj):
            u, v = cp.fill(u_prior, {**oyj, 'X': xj}, _theta, cp.draw_u.keys())

            def __dloss_dtheta(i: int):
                ui = {k: _u[i] if _u.ndim > 1 else _u for k, _u in u.items()}

                # prior gradient
                dlpu_dtheta = {key: 0. for key in _theta}
                for rv in ui:
                    if rv != 'X' and rv not in oyj:
                        dlpu_dtheta = {key: dlpu_dtheta[key] + cp.dlpu_dtheta(rv, key, ui, _theta) for key in _theta}

                # likelihood gradient
                dllkd_dtheta = {key: cp.dllkd_dtheta(key, ui, xj, oyj, _theta) for key in _theta}

                # REINFORCE estimator (if prior does not depend on parameters, this is standard gradient direction)
                if loss_type == 'neg-evidence':
                    lkd = jnp.exp(cp.llkd(ui, xj, oyj, _theta))
                    return {key: (dlpu_dtheta[key] + dllkd_dtheta[key]) * lkd for key in _theta}
                else:
                    llkd = cp.llkd(ui, xj, oyj, _theta)
                    return {key: dlpu_dtheta[key] * llkd + dllkd_dtheta[key] for key in _theta}

            return {key: -jnp.mean(g, 0) for key, g in jnp.vectorize(__dloss_dtheta)(range(n_samples)).items()}

        if x.ndim > 1:
            return {key: jnp.sum(g, 0) for key, g in vmap(_dloss_dtheta, (0, {k: 0 for k in oy}))(x, oy).items()}
        return _dloss_dtheta(x, oy)

    # batching
    n_batches = int(np.ceil(x.shape[0] / batch_size))
    rand_perm = np.random.permutation(x.shape[0])
    x_batches = np.array_split(x[rand_perm], n_batches, axis=0)
    oy_batches = {k: np.array_split(v[rand_perm], n_batches, axis=0) for k, v in oy.items()}

    # run optimization
    from jax.experimental import optimizers
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)

    losses = []

    @jit
    def update(i, _opt_state, _x_batch, _oy_batch):
        params = get_params(_opt_state)

        # g = dloss_dtheta(params)
        loss, g = value_and_grad(lambda _theta: loss_fn(_theta, _x_batch, _oy_batch))(params)
        return opt_update(i, g, _opt_state), loss

    opt_state = opt_init(theta0)
    for epoch in range(n_epoch):
        epoch_losses = []

        start_time = time.time()
        for i in range(n_batches):
            x_batch = x_batches[i]
            oy_batch = {k: _oy_batches[i] for k, _oy_batches in oy_batches.items()}

            opt_state, epoch_loss = update(i, opt_state, x_batch, oy_batch)
            epoch_losses.append(epoch_loss)
        epoch_time = time.time() - start_time

        losses.append(np.mean(epoch_losses))
        print("Epoch: {} | time: {:0.2f}s | loss: {:0.8f}".format(epoch + 1, epoch_time, losses[-1]))

    return get_params(opt_state), losses


def train(model, x_train: jnp.array, y_train: jnp.array, o_train: dict, x_pred: jnp.array, o_pred: dict,
          theta0: jnp.array, lam: float = 0.1, reg_loss: str = 'l1-bias', n_samples: int = 1000,
          n_epoch: int = 100, batch_size: int = 100, step_size: float = 1e-4) -> tuple:
    """
    It trains the parameters of the model, given data `x`, `y`, `o`. Starting from an initial solution `theta0`, the
    algorithm uses the Adam optimizer to minimize with respect to the negative evidence (default) or the negative
    averaged log-likelihood.
    Currently, the number of observations for each observed variable must be the same.

    Parameters
    ----------
    x_train: jnp.array
        Observation of treatment X for training.
    y_train: jnp.array
        Observation of outcome Y for training.
    o_train: dict
        Observations of observed variables O for training.
    x_pred: jnp.array
        Observations of observed variables X for prediction of Y and causal effect and bias estimation.
    o_pred: dict
        Observations of observed variables O for prediction of Y and causal effect and bias estimation.
    theta0: dict
        Initial solution of model parameters.
    lam: float
        It controls the regularization amount.
    reg_loss: str
        Type of regularization function. Allowed: 'l1-bias' (default), 'l2-bias', 'l1' and 'l2'.
    n_samples: int
        Number of samples to approximate the loss function and its gradient.
    n_epoch: int
        Number of epochs to run the optimization for.
    batch_size: int
        Data batch size at every epoch.
    step_size: float
        Initial step size in Adam optimizer.

    Returns
    -------
    theta, losses: tuple
        theta: jnp.array
            Estimated parameters minimizing the loss function.
        losses: list
            History of loss function evaluations.
        training_losses: list
            History of training loss function evaluations.
        reg_losses: list
            History of reg_loss function evaluations.
    """
    allowed_reg_losses = ['l1-bias', 'l2-bias', 'l1', 'l2']
    if reg_loss not in allowed_reg_losses:
        raise Exception('reg_loss={} not recognized. Please select within {}.'.format(reg_loss, allowed_reg_losses))

    all_n_obs = {1 if y_train.ndim <= 1 else y_train.shape[0]}
    for v in o_train.values():
        if v.ndim > 1:
            all_n_obs.add(v.shape[0])
    if len(all_n_obs) > 1:
        raise Exception('Each observed variable must have the same number of observations.')

    cp = CausalProb(model=model)

    @jit
    def training_loss_fn(theta: dict, x_train_batch: jnp.array, y_train_batch: jnp.array, o_train_batch: dict):
        u_prior = {k: u(n_samples, theta) for k, u in cp.draw_u.items()}

        def _training_loss_fn(xj, yj, oj):
            oyj = {**oj, 'Y': yj}
            u, v = cp.fill(u_prior, {**oyj, 'X': xj}, theta, cp.draw_u.keys())
            return -jnp.log(jnp.mean(jnp.exp(cp.llkd(u, xj, oyj, theta, v))))

        if x_train.ndim > 1:
            return jnp.mean(vmap(_training_loss_fn, (0, 0, {k: 0 for k in o_train_batch}))(x_train_batch, y_train_batch, o_train_batch))
        return _training_loss_fn(x_train, y_train, o_train)

    @jit
    def reg_loss_fn(theta: dict, x_pred: jnp.array, o_pred: dict):
        if reg_loss == 'l1-bias':
            return jnp.mean(jnp.abs(cp.causal_bias(x_pred, o_pred, theta, n_samples=n_samples)))
        if reg_loss == 'l2-bias':
            return jnp.mean(cp.causal_bias(x_pred, o_pred, theta, n_samples=n_samples) ** 2)
        if reg_loss in ['l1', 'l2']:
            flat_theta = jnp.concatenate([_theta.flatten() for _theta in tree_flatten(theta)[0]])
            if reg_loss == 'l1':
                return jnp.mean(jnp.abs(flat_theta))
            if reg_loss == 'l2':
                return jnp.mean(flat_theta ** 2)

    # batching
    n_batches = int(np.ceil(x_train.shape[0] / batch_size))
    rand_perm = np.random.permutation(x_train.shape[0])
    x_batches = np.array_split(x_train[rand_perm], n_batches, axis=0)
    y_batches = np.array_split(y_train[rand_perm], n_batches, axis=0)
    o_batches = {k: np.array_split(v[rand_perm], n_batches, axis=0) for k, v in o_train.items()}

    # run optimization
    from jax.experimental import optimizers
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)

    losses = []
    training_losses = []
    reg_losses = []
    strf = "Epoch: {} | time: {:0.2f}s | loss: {:e} | training loss: {:e} | {:s} loss: {:e}"

    @jit
    def update(i, opt_state, x_batch, y_batch, o_batch, x_pred, o_pred):
        params = get_params(opt_state)

        training_loss, training_g = value_and_grad(lambda theta: training_loss_fn(theta, x_batch, y_batch, o_batch))(params)
        if lam > 0:
            reg_loss, reg_loss_g = value_and_grad(lambda theta: reg_loss_fn(theta, x_pred, o_pred))(params)
            g = jax.tree_multimap(lambda x,y: x+lam*y, training_g, reg_loss_g)
        else:
            reg_loss = reg_loss_fn(params, x_pred, o_pred)
            g = training_g
        return opt_update(i, g, opt_state), training_loss, reg_loss

    opt_state = opt_init(theta0)
    for epoch in range(n_epoch):
        epoch_training_losses = []
        epoch_regs = []

        start_time = time.time()
        for i in range(n_batches):
            x_batch = x_batches[i]
            y_batch = y_batches[i]
            o_batch = {k: _o_batches[i] for k, _o_batches in o_batches.items()}

            opt_state, epoch_training_loss, epoch_reg = update(i, opt_state, x_batch, y_batch, o_batch, x_pred, o_pred)
            epoch_training_losses.append(epoch_training_loss)
            epoch_regs.append(epoch_reg)
        epoch_time = time.time() - start_time

        training_losses.append(np.mean(epoch_training_losses))
        reg_losses.append(np.mean(epoch_regs))
        losses.append(training_losses[-1] + lam * reg_losses[-1])
        print(strf.format(epoch + 1, epoch_time, losses[-1], training_losses[-1], reg_loss, reg_losses[-1]))

    return get_params(opt_state), losses, training_losses, reg_losses

