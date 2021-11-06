import jax.numpy as jnp
from jax import random
from jax.experimental import stax  # neural network library
from jax.experimental.stax import Dense, Relu  # neural network layers


class RealNVP:
    """
    This class implements a Real NVP normalizing flow architecture. For every layer, an affine coupling block acts as
    a non-linear transformation of the input. The non-linearity is encoded using a dense feed-forward neural network.
    Also, every layer the coupling block is flipped, changing from a lower-triangular to an upper triangular
    transformation.
    """
    def __init__(self, dim: int, n_layers: int = 4, seed: int = 0):
        """
        Parameters
        ----------
        dim: int
            Input and output dimension.
        n_layers: int
            Number of stacked coupling layers.
        seed: int
            It controls the seed of the pseudo-random generator.
        """
        self.dim = dim
        self.n_layers = n_layers
        self.net_init, self.net_apply = stax.serial(Dense(512), Relu, Dense(512), Relu, Dense(self.dim))
        self.seed = seed

    # neural network function
    def shift_and_log_scale_fn(self, u1: jnp.array, layer_params: jnp.array) -> list:
        """
        A neural network returning in output shift and log-scale of the affine coupling block.

        Parameters
        ----------
        u1: jnp.array
            Input of the neural network. It corresponds to half of the total input.
        layer_params: jnp.array
            Parameters of the neural network.

        Returns
        -------
        shift, log_scale: tuple
            Shift and log-scale of the affine couple block.
        """
        s = self.net_apply(layer_params, u1)
        return jnp.split(s, 2, axis=1)

    # layer forward and backward mechanisms
    def forward_layer(self, u: jnp.array, layer_params: jnp.array, flip: bool = False) -> jnp.array:
        """
        It implements the forward logic of an affine coupling block.

        Parameters
        ----------
        u: jnp.array
            Input of the forward transformation.
        layer_params: jnp.array
            Parameters of the neural network for this layer.
        flip: bool
            Flag that flips the transformation if `True`.

        Returns
        -------
        v: jnp.array
            Output of the forward transformation layer.
        """
        mid = u.shape[-1] // 2
        u1, u2 = u[:, :mid], u[:, mid:]
        if flip:
            u2, u1 = u1, u2
        shift, log_scale = self.shift_and_log_scale_fn(u1, layer_params)
        v2 = u2 * jnp.exp(log_scale) + shift
        if flip:
            u1, v2 = v2, u1
        v = jnp.concatenate([u1, v2], axis=-1)
        return v

    def backward_layer(self, v, layer_params, flip=False) -> tuple:
        """
        It implements the backward logic of an affine coupling block. It returns both the output of the backward
        transformation, and the log-determinant of its Jacobian, corresponding to the log-scale.

        Parameters
        ----------
        v: jnp.array
            Input of the backward transformation.
        layer_params: jnp.array
            Parameters of the neural network for this layer.
        flip: bool
            Flag that flips the transformation if `True`.

        Returns
        -------
        u, log_scale: tuple
            u: jnp.array.
                Output of the backward transformation.
            log_scale: log-determinant of the Jacobian of the backward transformation layer.
        """
        mid = v.shape[-1] // 2
        v1, v2 = v[:, :mid], v[:, mid:]
        if flip:
            v1, v2 = v2, v1
        shift, log_scale = self.shift_and_log_scale_fn(v1, layer_params)
        u2 = (v2 - shift) * jnp.exp(-log_scale)
        if flip:
            v1, u2 = u2, v1
        u = jnp.concatenate([v1, u2], axis=-1)
        return u, log_scale

    # full forward and backward mechanisms
    def forward(self, u: jnp.array, all_params: list) -> jnp.array:
        """
        Forward concatenation of coupling blocks. It flips them at every layer.

        Parameters
        ----------
        u: jnp.array
            Input of the forward transformation.
        all_params: list
            Collection of neural network parameters at every layer.

        Returns
        -------
        v: jnp.array
            Output of the forward transformation.
        """
        flip = False
        v = u
        for l in range(self.n_layers):
            v = self.forward_layer(v, all_params[l], flip)
            flip = not flip
        return v

    def backward(self, v: jnp.array, all_params: list) -> tuple:
        """
        Backward concatenation of coupling blocks. It flips them at every layer. It returns both the output of the
        backward transformation, and the log-determinant of its Jacobian, corresponding to the sum of log-scales at
        every layer.

        Parameters
        ----------
        v: jnp.array
            Input of the backward transformation.
        all_params: list
            Collection of neural network parameters at every layer.

        Returns
        -------
        v, log_det_inv_jac: tuple
            v: jnp.array
                Output of the backward transformation.
            log_det_inv_jac: jnp.array
                log-determinant of the Jacobian of the backward transformation.
        """
        flip = bool((self.n_layers + 1) % 2)
        u = v
        tot_log_scale = 0
        for l in reversed(range(self.n_layers)):
            u, log_scale = self.backward_layer(u, all_params[l], flip)
            tot_log_scale += log_scale
            flip = not flip
        return u, tot_log_scale

    # sampling
    def sample_base(self, n_samples: int, seed: int = 0) -> jnp.array:
        """
        It samples from the base distribution. Currently this is a multivariate standard Gaussian.

        Parameters
        ----------
        n_samples: int
            Number of samples to sample.
        seed: int
            It controls the seed of the pseudo-random generator.

        Returns
        -------
        u: jnp.array
            `n_samples` samples from the base distribution.
        """
        return random.normal(random.PRNGKey(self.seed + seed), (n_samples, self.dim))

    def sample_forward(self, all_params: list, n_samples: int, seed: int = 0) -> jnp.array:
        """
        It samples from the push-forward of the base distribution via the forward process.

        Parameters
        ----------
        all_params: list
            Collection of neural network parameters at every layer.
        n_samples: int
            Number of samples to sample.
        seed: int
            It controls the seed of the pseudo-random generator.

        Returns
        -------
        v: jnp.array
            `n_samples` samples from the push-forward of the base distribution.
        """
        u = self.sample_base(n_samples, seed=seed)
        return self.forward(u, all_params)

    # density evaluation
    def evaluate_base_logpdf(self, u: jnp.array) -> jnp.array:
        """
        It evaluates the log-probability density function of the base distribution.

        Parameters
        ----------
        u: jnp.array
            Location where to evaluate at.

        Returns
        -------
        lpu: jnp.array
            Evaluation of the log-probability density function of the base distribution for every input location.
        """
        return jnp.sum(-0.5 * u ** 2 - jnp.log(jnp.sqrt(2 * jnp.pi)), axis=-1)

    def evaluate_forward_logpdf(self, v: jnp.array, all_params: list) -> jnp.array:
        """
        It evaluates the log-probability density function of the push-forward distribution.

        Parameters
        ----------
        v: jnp.array
            Location where to evaluate at.
        all_params: list
            Collection of neural network parameters at every layer.

        Returns
        -------
        lpv: jnp.array
            Evaluation of the log-probability density function of the push-forward distribution for every input location.
        """
        u, tot_log_scale = self.backward(v, all_params)
        ildj = -jnp.sum(tot_log_scale, axis=-1)
        return self.evaluate_base_logpdf(u) + ildj

    # parameters initialization
    def init_layer_params(self, seed: int = 0) -> tuple:
        """
        It initializes the parameters of one neural network.

        Parameters
        ----------
        seed: int
            It controls the seed of the pseudo-random generator.

        Returns
        -------
        out_shape, layer_params: tuple
            out_shape: tuple
                Output shape of the neural network.
            layer_params: jnp.array
                Parameters of the neural network.
        """
        in_shape = (-1, self.dim // 2)
        out_shape, layer_params = self.net_init(random.PRNGKey(self.seed + seed), in_shape)
        return out_shape, layer_params

    def init_all_params(self, seed: int = 0) -> list:
        """
        It initializes parameters for all neural networks.

        Parameters
        ----------
        seed: int
            It controls the seed of the pseudo-random generator.

        Returns
        -------
        all_params: list
            Collection of neural network parameters at every layer.
        """
        all_params = []
        for l in range(self.n_layers):
            out_shape, _layer_params = self.init_layer_params(seed + l)
            all_params.append(_layer_params)
        return all_params



