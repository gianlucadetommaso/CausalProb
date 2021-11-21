import flax
from typing import Callable, Sequence
from flax.linen import Dense
from flax.linen.module import compact
import jax
from jax import numpy as jnp

class Sequential(flax.linen.Module):
    layers: Sequence[flax.linen.Module]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(flax.linen.Module):
    hidden: list
    use_bias: bool = True
    activation: Callable = jax.nn.relu

    @compact
    def __call__(self, x):
        for h in self.hidden:
            x = Dense(h, use_bias=self.use_bias)(x)
            x = self.activation(x)

if __name__ == '__main__':
    mlp = MLP([64,64])
    key = jax.random.PRNGKey(0)
    params = mlp.init(key, (-1,2))
    x = jnp.ones((10,2))
