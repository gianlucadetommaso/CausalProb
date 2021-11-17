import jax.numpy as jnp
import numpy as np


def pack(d: dict) -> jnp.array:
    """
    It packs dictionary into a 1-dimensional JAX array.

    Parameters
    ----------
    d: dict
        Dictionary to pack.

    Returns
    -------
    a: jnp.array
        1-dimensional JAX array corresponding to `d`.
    """
    return jnp.concatenate([jnp.ravel(v) for v in d.values()])


def unpack(a: jnp.array, d: dict) -> dict:
    """
    It unpacks a 1-dimensional JAX array into a dictionary with keys and value shapes like in `d`.

    Parameters
    ----------
    a: jnp.array
        Array to unpack.
    d: dict
        The unpacked version of `a` should have the same keys and value shapes as this dictionary.

    Returns
    -------
    unpacked: dict
        Dictionary corresponding to `d`-like unpacked version of `a`.
    """
    keys, values = [], []
    for k, v in d.items():
        keys.append(k), values.append(v)

    shapes = [v.shape for v in values]
    cum_sizes = np.cumsum([0] + [v.size for v in values]).tolist()
    return {keys[i]: a[cum_sizes[i]:cum_sizes[i+1]].reshape(shapes[i]) for i in range(len(d))}


# def sum_trees(tree_a, tree_b, lam):
#     leaves_a, treedef_a = tree_flatten(tree_a)
#     leaves_b, treedef_b = tree_flatten(tree_b)
#     assert treedef_a == treedef_b
#     return treedef_a.unflatten(map(lambda a, b: a + lam * b, leaves_a, leaves_b))


