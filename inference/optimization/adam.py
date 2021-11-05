import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)


def adam(loss, grad_loss, theta0: jnp.array, n_iter: int = 10000, alpha: float = 1, beta1: float = 0.9, beta2: float = 0.999):
    """
    Adam minimizer. It minimizes a `loss` function starting from the initial solution `theta0`. The
    gradient of the loss function with respect to the variable to optimize is given as a function `grad_loss`.
    Currently, only available stopping criterion is the number of iterations `n_iter`.

    Parameters
    ----------
    loss: func
        Loss function to minimize.
    grad_loss: func
        Gradient of the loss function.
    theta0: jnp.array
        Initial solution.
    n_iter: int
        Number of iterations.
    alpha: float
        Initial learning rate.
    beta1: float
        First-order momentum parameter.
    beta2: float
        Second-order momentum parameter.

    Returns
    -------
    theta, losses: tuple
        It returns the final state and the history of loss functions.
    """
    # initialize location
    theta = theta0.copy()

    # initialize ADAM parameters
    m1, m2 = 0.0, 0.0
    delta = 1e-8

    # store losses
    _loss = loss(theta)
    d = -grad_loss(theta)
    losses = [_loss]

    # printing output
    strf = "{:<10} {:<25} {:<25}"
    print(strf.format("iter", "loss", "||grad(loss)||"))
    print(60 * '-')
    print(strf.format(0, _loss, jnp.linalg.norm(d)))

    for t in range(1, n_iter):
        # update ADAM parameters
        m1 = beta1 * m1 + (1 - beta1) * d
        m2 = beta2 * m2 + (1 - beta2) * d ** 2
        alpha1 = alpha * jnp.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

        # update parameters
        theta += alpha1 * m1 / (delta + jnp.sqrt(m2)) / jnp.sqrt(t)

        # compute loss
        _loss = loss(theta)
        losses.append(_loss)

        # compute update direction
        d = -grad_loss(theta)

        # print loss
        print(strf.format(t, _loss, jnp.linalg.norm(d)))

    return theta, losses
