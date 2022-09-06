import jax
import jax.numpy as jnp
from jax.random import KeyArray


def take_each_n(array, count=1, step=1, shift=0):
    _, index_set = jnp.divmod(shift + jnp.arange(0, count, dtype=int) * step, array.shape[0])

    return index_set, jnp.take(array, index_set, axis=0)


def grid_in_cube(spacing=(2, 2, 2), scale=2., center_shift=(0., 0., 0.)):
    center_shift_ = jnp.array(center_shift)
    cube = jnp.mgrid[0:1:spacing[0] * 1j,
           0:1:spacing[1] * 1j,
           0:1:spacing[2] * 1j].transpose((1, 2, 3, 0))

    return scale * (cube - 0.5) + center_shift_


def grid_in_cube2(spacing=(4, 4, 4), lower=(-2, -2, -2), upper=(2, 2, 2)):
    cube = jnp.mgrid[lower[0]:upper[0]:spacing[0] * 1j,
           lower[1]:upper[1]:spacing[1] * 1j,
           lower[2]:upper[2]:spacing[2] * 1j].transpose((1, 2, 3, 0))

    return cube


def uniform_in_cube(rng_key: KeyArray, count=100, lower=(-2, -2, -2), upper=(2, 2, 2)):
    x = jax.random.uniform(rng_key, shape=(count, 1), minval=lower[0], maxval=upper[0])
    y = jax.random.uniform(rng_key, shape=(count, 1), minval=lower[1], maxval=upper[1])
    z = jax.random.uniform(rng_key, shape=(count, 1), minval=lower[2], maxval=upper[2])
    return jnp.hstack([x, y, z])
