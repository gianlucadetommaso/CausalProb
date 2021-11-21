import flax
import jax
from jax import numpy as jnp
from flax.linen import Dense, Module
from typing import Callable, Sequence
from jax.random import PRNGKey
from flax.linen.linear import default_kernel_init, zeros
from flax.linen.initializers import lecun_normal, zeros
from flax.linen import compact
from  jax import lax

default_kernel_init = lecun_normal


class MaskedDense(Module):
  """A linear transformation applied over the last dimension of the input.

  Attributes:
    features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """
  features: int
  mask: jnp.array = 1.0
  use_bias: bool = True
  dtype = jnp.float32
  kernel_init =  default_kernel_init
  bias_init = zeros
  precision = None

  @compact
  def __call__(self, x):
    kernel = self.param('kernel',
                        self.kernel_init,
                        (x.shape[-1], self.features))
    kernel = jnp.asarray(kernel, self.dtype)
    y = lax.dot_general(x, kernel*self.mask,
                        (((x.ndim - 1,), (0,)), ((), ())),
                        precision=self.precision)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      bias = jnp.asarray(bias, self.dtype)
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y


class Sequential(flax.linen.Module):
  layers: Sequence[flax.linen.Module]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


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
        for m, h0,h1 in zip(masks, hs, hs[1:]):
            net.extend([
                    MaskedDense(h1, m),
                    flax.linen.relu,
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

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    
    net = MADE(key, 2, 2, [64,64], 1) 
    net = MaskedDense(64)
    x = jnp.ones((32,2))
    params = net.init(key, x)
    
    print(net.apply(params, x))