import jax.numpy as jnp
from jax import random
from jax.experimental import stax  # neural network library
from jax.experimental.stax import Dense, Relu, normal  # neural network layers
import flax
from typing import List, Sequence

from models.normalizing_flow.distributions import Distribution

class Transform(flax.linen.Module):
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, u):
        """
            Returns inverse-transformed x and ldj
        """
    
    def backward(self, x):
        """
            Returns transformed x and ldj
        """

class NormalizingFlow(flax.linen.Module):
    transforms: Sequence[Transform]
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        m, _ = x.shape
        log_det = jnp.zeros(m)
        zs = [x]
        for t in self.transforms:
            x, ld = t.forward(x)
            log_det += ld
            zs.append(x)
        return zs[-1], log_det

    def backward(self, z):
        m, _ = z.shape
        log_det = jnp.zeros(m)
        xs = [z]
        for t in self.transforms[::-1]:
            z, ld = t.backward(z)
            log_det += ld
            xs.append(z)
        return xs[-1], log_det


class NormalizingFlowDist(flax.linen.Module):
    prior: Distribution
    flow: NormalizingFlow
    
    def sample(self, key, size):
        z = self.prior.sample(key, size)
        x, ldj = self.flow.forward(z)
        return x
    
    def log_prob(self, x):
        z, logdet = self.flow.backward(x)
        return self.prior.log_prob(z) + logdet
    

    
class RealNVP(Transform):
    net: flax.linen.Module
    flip: bool
        
    def shift_and_log_scale_fn(self, u1: jnp.array) -> list:
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
        s = self.net(u1)
        return jnp.split(s, 2, axis=-1)
    
    def forward(self, u):
        mid = u.shape[-1] // 2
        u1, u2 = (u[:, :mid], u[:, mid:]) if u.ndim == 2 else (u[:mid], u[mid:])
        if self.flip:
            u2, u1 = u1, u2
        shift, log_scale = self.shift_and_log_scale_fn(u1)
        v2 = u2 * jnp.exp(log_scale) + shift
        if self.flip:
            u1, v2 = v2, u1
        v = jnp.concatenate([u1, v2], axis=-1)
        return v, 0 # 0 is incorrect, but it is not used for log-density estimation anyway
    
    def backward(self, v) -> tuple:
        mid = v.shape[-1] // 2
        v1, v2 = (v[:, :mid], v[:, mid:]) if v.ndim == 2 else (v[:mid], v[mid:])

        if self.flip:
            v1, v2 = v2, v1
        shift, log_scale = self.shift_and_log_scale_fn(v1)
        u2 = (v2 - shift) * jnp.exp(-log_scale)
        if self.flip:
            v1, u2 = u2, v1
        u = jnp.concatenate([v1, u2], axis=-1)
        return u, -log_scale.sum(-1)


class Sequential(flax.linen.Module):
  layers: Sequence[flax.linen.Module]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


from flax.linen import Dense, relu
from jax import numpy as jnp
from jax.random import PRNGKey

from models.normalizing_flow.distributions import StandardGaussian


if __name__ == '__main__':
    
    mlp = lambda: Sequential([
        Dense(64),
        relu,
        Dense(2),
     ])
    
    x = jnp.ones((32, 2), dtype=jnp.float32)
    
    prng_key = PRNGKey(0)
    
    
    flow = NormalizingFlow((RealNVP(mlp(), False), RealNVP(mlp(), False), RealNVP(mlp(), False)))
    flow_dist = NormalizingFlowDist(StandardGaussian(2), flow)
    
    
    params = flow_dist.init(prng_key, x, method=flow_dist.log_prob)
    
    print(params['params'])
    
    samples = flow_dist.apply(params, prng_key, 10, method=flow_dist.sample)
    assert samples.shape[-1] == 2
    assert samples.shape[0] == 10