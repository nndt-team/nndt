from typing import *

import jax
import jax.numpy as jnp
from jax.random import KeyArray


def take_each_n(array, count=1, step=1, shift=0):
    """Takes elements from an array along an axis
    
    An advanced range iterator that takes data according to the index and starts 
    from the beginning of the array if the index is greater than the array length.
    
    Parameters
    ----------
    array : ndarray
        The source array
    count : int, optional
        The number of elements to take (default is 1)
    step : int, optional
        The stepwise movement along the array after each selected index
    shift : int, optional
        The starting index of the array element to take (default is 0)

    Returns
    -------
    ndarray
        an array of indices used in selecting elements from the source array
        an array of elements taken from the source array
    """

    _, index_set = jnp.divmod(shift + jnp.arange(0, count, dtype=int) * step, array.shape[0])

    return index_set, jnp.take(array, index_set, axis=0)


def grid_in_cube(spacing=(2, 2, 2), scale=2., center_shift=(0., 0., 0.)):
    """Samples points from a 3D cube according to a uniform grid
    
    Parameters
    ----------
    spacing : tuple, optional
        A tuple of ints of step-lengths of the grid (default is (2, 2, 2))
    scale : float, optional
        The scaling factor (default is 2.)
    center_shift : tuple, optional
        A tuple of ints of coordinates by which to modify the center of the cube (default is (0., 0., 0.))

    Returns
    -------
    ndarray
        mesh-grid of the sampled points from the 3D cube
    """

    center_shift_ = jnp.array(center_shift)
    cube = jnp.mgrid[0:1:spacing[0] * 1j,
           0:1:spacing[1] * 1j,
           0:1:spacing[2] * 1j].transpose((1, 2, 3, 0))

    return scale * (cube - 0.5) + center_shift_


def grid_in_cube2(spacing=(4, 4, 4), lower=(-2, -2, -2), upper=(2, 2, 2)):
    """Samples points from a 3D cube
    
    Parameters
    ----------
    spacing : tuple, optional
        A tuple of ints of step-lengths of the grid (default is (4, 4, 4))
    lower : tuple, optional
        A tuple of ints of start values of the grid (default is (-2, -2, -2))
    upper : tuple, optional
        A tuple of ints of stop values of the grid (default is (2, 2, 2))

    Returns
    -------
    ndarray
        multi-dimensional mesh-grid
    """
    cube = jnp.mgrid[lower[0]:upper[0]:spacing[0] * 1j,
           lower[1]:upper[1]:spacing[1] * 1j,
           lower[2]:upper[2]:spacing[2] * 1j].transpose((1, 2, 3, 0))

    return cube


def uniform_in_cube(rng_key: KeyArray, count=100, lower=(-2, -2, -2), upper=(2, 2, 2)):
    """Randomly samples points from a 3d cube and stacks the arrays horizontally
    
    Parameters
    ----------
    rng_key: KeyArray
        A PRNGKey used as the random key.
    count: int, optional
        Length of array of random points to sample (default is 100)
    lower: tuple, optional 
        tuple of ints broadcast-compatible with ``shape``, a minimum 
        (inclusive) value for the range (default is (-2, -2, -2)
    upper: tuple, optional
        tuple of ints broadcast-compatible with  ``shape``, a maximum
        (exclusive) value for the range (default is (2, 2, 2)

    Returns
    -------
    ndarray
        a horizontally stacked array (shape is (`count`, 3)) of random floating 
        points between the specified range of coordinates (lower and upper)
    """
    x = jax.random.uniform(rng_key, shape=(count, 1), minval=lower[0], maxval=upper[0])
    y = jax.random.uniform(rng_key, shape=(count, 1), minval=lower[1], maxval=upper[1])
    z = jax.random.uniform(rng_key, shape=(count, 1), minval=lower[2], maxval=upper[2])
    return jnp.hstack([x, y, z])


def sdf_primitive_sphere(center=(0., 0., 0.), radius=1.):
    def prim(x: float, y: float, z: float):
        sdf = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 - radius ** 2
        return sdf

    vec_prim = jax.vmap(prim)

    prim_x = jax.grad(prim, argnums=0)
    prim_y = jax.grad(prim, argnums=1)
    prim_z = jax.grad(prim, argnums=2)

    vec_prim_x = jax.vmap(prim_x)
    vec_prim_y = jax.vmap(prim_y)
    vec_prim_z = jax.vmap(prim_z)

    return vec_prim, vec_prim_x, vec_prim_y, vec_prim_z


def barycentric_grid(order: Sequence[Union[int, Sequence[int]]] = (1, -1),
                     spacing: Sequence[int] = (0, 3),
                     filter_negative = True):
    pass
