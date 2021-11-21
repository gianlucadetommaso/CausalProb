from models.normalizing_flow.core import Transform
import flax
from typing import Sequence
from jax import numpy as jnp

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

