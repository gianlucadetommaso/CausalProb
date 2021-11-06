import jax.numpy as jnp


class RealNVP:
    def __init__(self):
        pass

    def forward(self, shift_and_log_scale_fn, u: jnp.array, theta: jnp.array, flip: bool = False):
        mid = u.shape[-1] // 2
        u1, u2 = u[:, :mid], u[:, mid:]
        if flip:
            u2, u1 = u1, u2
        shift, log_scale = shift_and_log_scale_fn(theta, u1)
        v2 = u2 * jnp.exp(log_scale) + shift
        if flip:
            u1, v2 = v2, u1
        v = jnp.concatenate([u1, v2], axis=-1)
        return v

    def backward(self, theta, shift_and_log_scale_fn, y, flip=False):
        mid = u.shape[-1] // 2
        v1, v2 = v[:, :mid], v[:, mid:]
        if flip:
            v1, v2 = v2, v1
        shift, log_scale = shift_and_log_scale_fn(theta, v1)
        u2 = (v2 - shift) * jnp.exp(-log_scale)
        if flip:
            v1, u2 = u2, v1
        u = jnp.concatenate([v1, u2], axis=-1)
        return u, log_scale



