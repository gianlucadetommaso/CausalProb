import jax.numpy as jnp
from jax import random
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
    


if __name__ == '__main__':

    from flax.linen import Dense, relu
    from jax import numpy as jnp
    from jax.random import PRNGKey

    from models.normalizing_flow.distributions import StandardGaussian
    from models.normalizing_flow.nn import Sequential
    from models.normalizing_flow import RealNVP
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
    
    
    samples = flow_dist.apply(params, prng_key, 10, method=flow_dist.sample)
    assert samples.shape[-1] == 2
    assert samples.shape[0] == 10