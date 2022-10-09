from typing import *

import jax
import jax.numpy as jnp
from jax.random import KeyArray


def take_each_n(array: jnp.ndarray, count=1, step=1, shift=0):
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
    """

    Parameters
    ----------
    center : tuple, optional
        Coordinates of center x, y, z (defaults is (0., 0., 0.))
    radius : float, optional
        Radius of sphere (defaults is 1.)
    
    Returns
    -------
    Set of jax.vmap
        description smth, x, y, z
    """
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


def help_barycentric_grid(order: Sequence[Union[int, Sequence[int]]] = (1, -1)):
    """Presents view of barycentric_grid formula with various parameters

    Parameters
    ----------
    order : Sequence[Union[int, Sequence[int]]], optional
        oder of params (defaults is (1, -1))

    Returns
    -------
    str
        view of formula
    """
    order_adv = [((v,) if isinstance(v, int) else v) for v in order]

    polynomial = ""
    polynomial_sub = ""

    for ind_code, code in enumerate(order_adv):
        expr = ""
        expr_sub = ""
        for ind_iter, iter_ in enumerate(code):
            if iter_ == 0:
                expr += f"X*"
            elif iter_ > 0:
                expr += f"l{iter_}*"
                expr_sub += f"l{iter_}+"
            elif iter_ < 0:
                expr += f"(1-l{-iter_})*"
                expr_sub += f"(1-l{-iter_})+"
            if ind_iter == len(code)-1:
                expr = expr[:-1]
        polynomial += f"{expr}*e{ind_code+1} + "
        polynomial_sub += f"{expr_sub}"
        if ind_code == len(order_adv) - 1:
            polynomial = polynomial[:-3]
            polynomial_sub = polynomial_sub[:-1]

    polynomial = polynomial.replace("X", f"(1-({polynomial_sub}))")

    return polynomial


def barycentric_grid(order: Sequence[Union[int, Sequence[int]]] = (1, -1),
                     spacing: Sequence[int] = (0, 3),
                     filter_negative: bool = True):
    """Makes a barycentyc grid

    Parameters
    ----------
    order : (Sequence[Union[int, Sequence[int]]], optional)
        order of parameters (defaults is (1, -1))
    spacing : (Sequence[int], optional)
        _description_. (defaults is (0, 3))
    filter_negative : (bool, optional)
        complite negative values (defaults is True)

    Returns
    -------
    jnp.array 
        _description_
    """
    assert ((len(order) >= 2),
            "The `order` parameter must include more than 1 iterator.")
    assert ((len(spacing) >= 2),
            "The `spacing` parameter must include more than 1 iterator.")
    assert (((spacing[0] == 0) or (spacing[0] is None)),
            "First value in spacing must be 0, because zero iterator is not used.")

    order_adv = [((v,) if isinstance(v, int) else v) for v in order]
    flat_flat_order = [element for x in order_adv for element in x]

    assert (jnp.max(jnp.abs(jnp.array(flat_flat_order))) > len(spacing),
            "Index of iterator in `order` overcomes the number of iterators in `spacing`.")
    assert (jnp.sum(jnp.array(flat_flat_order) == 0) > 1,
            "Only one 0 is possible in `order`. Zero shows replenished coefficient.")

    lin_spaces = [[0., 0.]] + [jnp.linspace(0, 1, s) for s in spacing[1:]]
    iter_list = [0]*len(spacing)
    ret = []

    while iter_list[0] < 1:

        # Collect cases from current iterator states
        case = []
        case_sub = 0.
        replace_ind = None
        for ord_ind, ord in enumerate(order_adv):
            val = 1.
            val_sub = 1.
            for ord_ind2, ord2 in enumerate(ord):
                if ord2 == 0:
                    val *= 0.
                    val_sub *= 0.
                    replace_ind = ord_ind
                elif ord2 > 0:
                    val *= lin_spaces[ord2][iter_list[ord2]]
                    val_sub *= lin_spaces[ord2][iter_list[ord2]]
                elif ord2 < 0:
                    val *= (1. - lin_spaces[-ord2][iter_list[-ord2]])
                    val_sub *= (1. - lin_spaces[-ord2][iter_list[-ord2]])

            case.append(float(val))
            case_sub += val_sub

        if replace_ind is not None:
            case[replace_ind] = float(1 - case_sub)

        # Add this case or filter if negative values are not allowed
        here_is_negative = False
        for i in case:
            if i < 0.:
                here_is_negative = True
        if (not here_is_negative) or (not filter_negative and here_is_negative):
            ret.append(case)

        # Update iterators from the last
        iter_list[-1] += 1
        for ind in reversed(range(len(iter_list))):
            if iter_list[ind] >= len(lin_spaces[ind]):
                iter_list[ind] = 0
                iter_list[ind-1] += 1

    ret = jnp.array(ret)
    return ret


def train_test_split(array: jnp.array,
                     rng: jax.random.PRNGKey,
                     test_size: float = 0.3) -> (list, list):
    indices = jnp.arange(len(array))

    test_index_list = [index for index in
                       jax.random.choice(key=rng,
                                         a=indices,
                                         replace=False,
                                         shape=[int(len(indices)*test_size)]).tolist()]
    train_index_list = [index for index in indices.tolist() if index not in test_index_list]

    return train_index_list, test_index_list




