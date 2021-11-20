from jax import numpy as jnp
from jax.random import normal

class Distribution:
            
    def rsample(self, key, shape):
        raise NotImplementedError
    
    def samples(self, key, num_samples):
        raise NotImplementedError
    
    def logprob(self, key, x):
        raise NotImplementedError

    
class StandardGaussian(Distribution):
    def __init__(self, dim):
        self.dim = dim
        self.scale = 1.0
        self.loc = 0.0
        
    def sample(self, key, n_samples):
        return normal(key, (n_samples, self.dim))
    
    def log_prob(self, x):
        normalize_term = jnp.log(jnp.sqrt(2 * jnp.pi))
        return jnp.sum(-0.5 * x ** 2 - normalize_term, axis=-1) # returns joint log density