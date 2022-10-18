from typing import *

import jax
import jax.numpy as jnp
from jax.random import KeyArray


def take_each_n(array: jnp.ndarray, count=1, step=1, shift=0) -> (jnp.ndarray, jnp.ndarray):
    """An advanced range iterator that iterates over data and selects elements according to their index.
    If during iteration the index becomes greater than the array length,
     the iteration continues from the beginning of the array.
    This function selects elements from an array along the axis zero, which is the first dimension.
    
    Parameters
    ----------
    array : ndarray
        The source array
    count : int, optional
        The number of elements to take (default is 1)
    step : int, optional
        The step of iterator
    shift : int, optional
        Index shift for the first index (default is 0)

    Returns
    -------
    (ndarray, ndarray)
        an array of indices of the elements taken from the source array
        an array of elements from the source array corresponding to the selected indices
    """

    _, index_set = jnp.divmod(shift + jnp.arange(0, count, dtype=int) * step, array.shape[0])

    return index_set, jnp.take(array, index_set, axis=0)


def grid_in_cube(spacing=(2, 2, 2), scale=2., center_shift=(0., 0., 0.)) -> jnp.ndarray:
    """Draw samples from the uniform grid that is defined inside a bounding box
    with center in the `center_shift` and size of `scale`
    
    Parameters
    ----------
    spacing : tuple, optional
        Number of sections along X, Y, and Z axes (default is (2, 2, 2))
    scale : float, optional
        The scaling factor which defines the size of bounding box (default is 2.)
    center_shift : tuple, optional
        A tuple of ints of coordinates by which to modify the center of the cube (default is (0., 0., 0.))

    Returns
    -------
    ndarray
        3D mesh-grid with shape (spacing[0], spacing[1], spacing[2], 3)
    """

    center_shift_ = jnp.array(center_shift)
    cube = jnp.mgrid[0:1:spacing[0] * 1j,
           0:1:spacing[1] * 1j,
           0:1:spacing[2] * 1j].transpose((1, 2, 3, 0))

    return scale * (cube - 0.5) + center_shift_


def grid_in_cube2(spacing=(4, 4, 4), lower=(-2, -2, -2), upper=(2, 2, 2)) -> jnp.ndarray:
    """Draw samples from the uniform grid that is defined inside a (lower, upper) bounding box
    
    Parameters
    ----------
    spacing : tuple, optional
        Number of sections along X, Y, and Z axes (default is (4, 4, 4))
    lower: tuple, optional
        position of lower point for the bounding box  (default is (-2, -2, -2)
    upper: tuple, optional
        position of upper point for the bounding box (default is (2, 2, 2)

    Returns
    -------
    ndarray
        3D mesh-grid with shape (spacing[0], spacing[1], spacing[2], 3)
    """
    cube = jnp.mgrid[lower[0]:upper[0]:spacing[0] * 1j,
           lower[1]:upper[1]:spacing[1] * 1j,
           lower[2]:upper[2]:spacing[2] * 1j].transpose((1, 2, 3, 0))

    return cube


def uniform_in_cube(rng_key: KeyArray, count=100, lower=(-2, -2, -2), upper=(2, 2, 2)) -> jnp.ndarray:
    """Draw samples from uniform distribution inside a (lower, upper) bounding box
    
    Parameters
    ----------
    rng_key: KeyArray
        Jax key for a random generator
    count: int, optional
        Size of sampling (default is 100)
    lower: tuple, optional 
        position of lower point for the bounding box  (default is (-2, -2, -2)
    upper: tuple, optional
        position of upper point for the bounding box (default is (2, 2, 2)

    Returns
    -------
    ndarray
        Array of random points (shape is (`count`, 3))
    """
    x = jax.random.uniform(rng_key, shape=(count, 1), minval=lower[0], maxval=upper[0])
    y = jax.random.uniform(rng_key, shape=(count, 1), minval=lower[1], maxval=upper[1])
    z = jax.random.uniform(rng_key, shape=(count, 1), minval=lower[2], maxval=upper[2])
    return jnp.hstack([x, y, z])


def help_barycentric_grid(order: Sequence[Union[int, Sequence[int]]] = (1, -1)):
    """Helper for 'barycentric_grid' function.
    This method prints an iteration polynomial for the barycentric coordinates.

    Parameters
    ----------
    order : Sequence[Union[int, Sequence[int]]], optional
        Order of iterators (defaults is (1, -1), as for the linear interpolation)

    Returns
    -------
    str
        Text representation of the polynomial
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
            if ind_iter == len(code) - 1:
                expr = expr[:-1]
        polynomial += f"{expr}*e{ind_code + 1} + "
        polynomial_sub += f"{expr_sub}"
        if ind_code == len(order_adv) - 1:
            polynomial = polynomial[:-3]
            polynomial_sub = polynomial_sub[:-1]

    polynomial = polynomial.replace("X", f"(1-({polynomial_sub}))")

    return polynomial


def barycentric_grid(order: Sequence[Union[int, Sequence[int]]] = (1, -1),
                     spacing: Sequence[int] = (0, 3),
                     filter_negative: bool = True):
    """Analog of nested `for` cycles in barycentric coordinates.
    In 1D case without free variable this is linear interpolation.
    In 2D case with free variable this is list of ternary plot points.
    In ND case this works like a uniform grid inside N-simplex.
    If this simplex is defined on the basis vectors of space.

    Parameters
    ----------
    order : (Sequence[Union[int, Sequence[int]]], optional)
        Order of iterator in the polynomial (defaults is (1, -1), as for the linear interpolation)
    spacing : (Sequence[int], optional)
        This is grid spacing for each iterated variable.
        N-value in some position is equivalent to jnp.linspace(0,1,N).
        Zero element must be zero, because this is a technical definition for free variable.
    filter_negative : (bool, optional)
        Filter values outside the simple (defaults is True)

    Returns
    -------
    jnp.ndarray
        List of vectors inside the simplex. All the vectors have len(spacing) components.
    """
    assert (len(order) >= 2), "The `order` parameter must include more than 1 iterator."
    assert (len(spacing) >= 2), "The `spacing` parameter must include more than 1 iterator."
    assert ((spacing[0] == 0) or (spacing[0] is None)), \
        "First value in spacing must be 0, because zero iterator is not used."

    order_adv = [((v,) if isinstance(v, int) else v) for v in order]
    flat_flat_order = [element for x in order_adv for element in x]

    assert float(jnp.max(jnp.abs(jnp.array(flat_flat_order)))) < len(spacing), \
        "Index of iterator in `order` overcomes the number of iterators in `spacing`."
    assert float(jnp.sum(jnp.array(flat_flat_order) == 0)) <= 1, \
        "Only one 0 is possible in `order`. Zero shows replenished coefficient."

    lin_spaces = [[0., 0.]] + [jnp.linspace(0, 1, s) for s in spacing[1:]]
    iter_list = [0] * len(spacing)
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
                iter_list[ind - 1] += 1

    ret = jnp.array(ret)
    return ret


def train_test_split(array: jnp.ndarray,
                     rng: KeyArray,
                     test_size: float = 0.3) -> (list, list):
    """
    Split array to test and train subset. This is analog of `model_selection.train_test_split` in sklearn.

    Parameters
    ----------
    array: jnp.ndarray :
        Array for split
    rng : KeyArray :
        Jax key for a random generator
    test_size: float:
        Percent of test subset in the array

    Returns
    ----------
    (list, list)
        List of indexes for test and train subsets
    """
    assert (0. <= test_size <= 1.)
    indices = jnp.arange(len(array))

    test_index_list = [index for index in
                       jax.random.choice(key=rng,
                                         a=indices,
                                         replace=False,
                                         shape=[int(len(indices) * test_size)]).tolist()]
    train_index_list = [index for index in indices.tolist() if index not in test_index_list]

    return train_index_list, test_index_list


def scale_xyz(xyz, scale=(1., 1., 1.)):
    """
    Scale array of points to the `scale` factor.

    Parameters
    ----------
    :param xyz: Array of points
    :param scale: The scale factor

    Returns
    -------
    :return: Scaled array of points with shape equal to shape of `xyz` array
    """
    assert(xyz.shape[-1] == 3)
    scale = jnp.array(scale)

    ret_shape = xyz.shape
    xyz = xyz.reshape((-1, 3))
    xyz = scale*xyz

    xyz = xyz.reshape(ret_shape)
    return xyz