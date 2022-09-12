import unittest

from nndt.math_core import *
import jax.numpy as jnp


class MathCoreTestCase(unittest.TestCase):

    def test_grid_in_cube(self):
        cube = grid_in_cube(spacing=(4, 4, 4), scale=4., center_shift=(2., 2., 2.))
        self.assertEqual(0., float(jnp.min(cube)))
        self.assertEqual(4., float(jnp.max(cube)))

    def test_grid_in_cube2(self):
        cube = grid_in_cube2(spacing=(4, 4, 4), lower=(-2, -2, -2), upper=(2, 2, 2))
        self.assertEqual((4, 4, 4, 3), cube.shape)
        self.assertEqual(-2, float(jnp.min(cube)))
        self.assertEqual(2, float(jnp.max(cube)))

    def test_uniform_in_cube(self):
        rng_key = jax.random.PRNGKey(42)
        cube = uniform_in_cube(rng_key,
                               count=100,
                               lower=(-2, -2, -2),
                               upper=(2, 2, 2))
        self.assertEqual((100, 3), cube.shape)
        self.assertLessEqual(-2, float(jnp.min(cube)))
        self.assertGreaterEqual(2, float(jnp.max(cube)))

    def test_take_each_n(self):
        rng_key = jax.random.PRNGKey(42)
        index_set, array = take_each_n(jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                       count=4,
                                       step=4,
                                       shift=1)
        self.assertEqual((4,), array.shape)
        self.assertEqual((4,), index_set.shape)
        self.assertEqual([1, 5, 9, 3], list(array))
        self.assertEqual([1, 5, 9, 3], list(index_set))


if __name__ == '__main__':
    unittest.main()
