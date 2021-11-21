import flax
import jax
from jax import numpy as jnp
from flax.linen import Dense, Module
from typing import Any, Callable, Sequence
from jax._src.lax.lax import DType
from jax.random import PRNGKey
from flax.linen.linear import default_kernel_init, zeros
from flax.linen.initializers import lecun_normal, zeros
from flax.linen import compact
from  jax import lax
from models.normalizing_flow.nn import Sequential

default_kernel_init = lecun_normal()


class MaskedDense(Module):
  features: int
  mask: jnp.array = 1.0
  use_bias: bool = True
  dtype: DType = jnp.float32
  precision: Any = None
  kernel_init: Callable = default_kernel_init
  bias_init: Callable = zeros

  @compact
  def __call__(self, inputs):
    inputs = jnp.asarray(inputs, self.dtype)
    kernel = self.param('kernel',
                        self.kernel_init,
                        (inputs.shape[-1], self.features))
    kernel = jnp.asarray(kernel, self.dtype)
    y = lax.dot_general(inputs, kernel*self.mask,
                        (((inputs.ndim - 1,), (0,)), ((), ())),
                        precision=self.precision)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      bias = jnp.asarray(bias, self.dtype)
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y




class MADE(Module):
    """An implementation of `MADE: Masked Autoencoder for Distribution Estimation`
    (https://arxiv.org/abs/1502.03509).
    Args:
        transform: maps inputs of dimension ``num_inputs`` to ``2 * num_inputs``
    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """
    key: PRNGKey
    input_dim: int
    output_dim: int
    hidden_sizes: Sequence[int]
    num_masks: int
    natural_ordering: bool = False

    def setup(self):
        assert self.output_dim % self.input_dim == 0, "output_dim must be integer multiple of input_dim"
        # define a simple MLP neural net
        masks = self._generate_masks(self.key)

        net = []
        hs = [self.input_dim] + list(self.hidden_sizes) + [self.output_dim]
        for m, h1 in zip(masks, hs[1:]):
            net.extend([
                    MaskedDense(h1, mask=m),
                    jax.nn.relu,
                ])
        net.pop() # pop the last ReLU for the output layer
        self.net = Sequential(net)
        
       
    def _generate_masks(self, key):
        m = {}
        if m and self.num_masks == 1: return [1.0]*len(self.hidden_sizes)
        L = len(self.hidden_sizes)
        
        m[-1] = jnp.arange(self.input_dim) if self.natural_ordering else jax.random.permutation(key, self.input_dim)
        key, _ = jax.random.split(key)
        for l, key in zip(range(L), jax.random.split(key, L)):
            m[l] = jax.random.randint(key, (self.hidden_sizes[l],), m[l-1].min(), self.input_dim-1)
        
        # construct the mask matrices
        masks = [m[l-1][:,None] <= m[l][None,:] for l in range(L)]
        masks.append(m[L-1][:,None] < m[-1][None,:])
        
        # handle the case where output_dim = input_dim * k, for integer k > 1
        if self.output_dim > self.input_dim:
            k = int(self.output_dim / self.input_dim)
            # replicate the mask across the other outputs
            masks[-1] = jnp.concatenate([masks[-1]]*k, axis=1)
        
        return masks

    def __call__(self, x):
        return self.net(x)

class ARMLP(Module):
    """ a 4-layer auto-regressive MLP, wrapper around MADE net """
    key: PRNGKey
    output_dim: int
    hidden_dim:int =  24

    @compact
    def __call__(self, x):
        return  MADE(self.key, x.shape[-1], self.output_dim, [self.hidden_dim ,self.hidden_dim, self.hidden_dim] , num_masks=1, natural_ordering=True)(x)




