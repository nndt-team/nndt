import unittest

import numpy as np

from nndt.math_core import *


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
        index_set, array = take_each_n(jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                       count=4,
                                       step=4,
                                       shift=1)
        self.assertEqual((4,), array.shape)
        self.assertEqual((4,), index_set.shape)
        self.assertEqual([1, 5, 9, 3], list(array))
        self.assertEqual([1, 5, 9, 3], list(index_set))

    def test_sdf_primitive_sphere(self):
        vec_prim, vec_prim_x, vec_prim_y, vec_prim_z = sdf_primitive_sphere()
        xyz = jnp.array([[0., 0., 0.], [0., 0., 1.], [0., 0., 2.]])

        self.assertTrue(bool(jnp.allclose(jnp.array([-1., 0., 3.]),
                                          vec_prim(xyz[:, 0], xyz[:, 1], xyz[:, 2]))))
        self.assertTrue(bool(jnp.allclose(jnp.array([0., 0., 0.]),
                                          vec_prim_x(xyz[:, 0], xyz[:, 1], xyz[:, 2]))))
        self.assertTrue(bool(jnp.allclose(jnp.array([0., 0., 0.]),
                                          vec_prim_y(xyz[:, 0], xyz[:, 1], xyz[:, 2]))))
        self.assertTrue(bool(jnp.allclose(jnp.array([0., 2., 4.]),
                                          vec_prim_z(xyz[:, 0], xyz[:, 1], xyz[:, 2]))))

    def test_sdf_primitive_sphere2(self):
        vec_prim, vec_prim_x, vec_prim_y, vec_prim_z = sdf_primitive_sphere(
                        center=(1., 1., 1.),
                        radius=float(jnp.sqrt(1**2 + 1**2 + 1**2)))
        xyz = jnp.array([[0., 0., 0.]])

        self.assertTrue(bool(jnp.allclose(jnp.array([0., 0., 0.]),
                                          vec_prim(xyz[:, 0], xyz[:, 1], xyz[:, 2]))))
        self.assertTrue(bool(jnp.allclose(jnp.array([-2., -2., -2.]),
                                          vec_prim_x(xyz[:, 0], xyz[:, 1], xyz[:, 2]))))
        self.assertTrue(bool(jnp.allclose(jnp.array([-2., -2., -2.]),
                                          vec_prim_y(xyz[:, 0], xyz[:, 1], xyz[:, 2]))))
        self.assertTrue(bool(jnp.allclose(jnp.array([-2., -2., -2.]),
                                          vec_prim_z(xyz[:, 0], xyz[:, 1], xyz[:, 2]))))

def assert_sum_equal_one(values):
    jnp.allclose(1., jnp.sum(values, axis=-1))

class BarycentricGridTestCase(unittest.TestCase):

    def test_linear(self):
        # l1*e1 + (1-l2)*e2
        coords = barycentric_grid(order= (1, -1),
                                  spacing= (0, 3))
        assert_sum_equal_one(coords)

        jnp.allclose([[0., 1.],
                      [0.5, 0.5],
                      [1., 0.]], coords)

    def test_linear_inverted(self):
        # (1-l1)*e1 + l2*e2
        coords = barycentric_grid(order = (-1, 1),
                                  spacing = (0, 3),
                                  filter_negative = True)

        assert_sum_equal_one(coords)

        jnp.allclose([[1., 0.],
                      [0.5, 0.5],
                      [0., 1.]], coords)

    def test_ternary(self):
        # l1*e1 + (1-(l1+l2))*e2 + l2*e3
        coords = barycentric_grid(order = (1, 0, 2),
                                  spacing = (0, 3, 3),
                                  filter_negative = True)

        assert_sum_equal_one(coords)

        jnp.allclose([[0.0, 1.0, 0.0],
                      [0.0, 0.5, 0.5],
                      [0.0, 0.0, 1.0],
                      [0.5, 0.5, 0.0],
                      [0.5, 0.0, 0.5],
                      [1.0, 0.0, 0.0]], coords)

    def test_ternary_without_negative_filter(self):
        # l1*e1 + (1-(l1+l2))*e2 + l2*e3
        coords = barycentric_grid(order = (1, 0, 2),
                                  spacing = (0, 3, 3),
                                  filter_negative = False)

        assert_sum_equal_one(coords)

        jnp.allclose([[0.0, 1.0, 0.0],
                      [0.0, 0.5, 0.5],
                      [0.0, 0.0, 1.0],
                      [0.5, 0.5, 0.0],
                      [0.5, 0.0, 0.5],
                      [0.5, -0.5, 1.0],
                      [1.0, 0.0, 0.0],
                      [1.0, -0.5, 0.5],
                      [1.0, -1.0, 1.0]], coords)

    def test_replenishment_without_negative_filter(self):
        # l1*e1 + (1-(l1+l2))*e2 + l2*e3
        coords = barycentric_grid(order = (1, (-1,-2), 2),
                                  spacing = (0, 3, 3),
                                  filter_negative = False)

        assert_sum_equal_one(coords)

        jnp.allclose([[0.0, 1.0, 0.0],
                      [0.0, 0.5, 0.5],
                      [0.0, 0.0, 1.0],
                      [0.5, 0.5, 0.0],
                      [0.5, 0.0, 0.5],
                      [0.5, -0.5, 1.0],
                      [1.0, 0.0, 0.0],
                      [1.0, -0.5, 0.5],
                      [1.0, -1.0, 1.0]], coords)

    def test_square(self):
        # (1-l1)*(1-l2)*e1 + (l1-1)*l2*e2 + l1*(1-l2)*e3 + l1*l2*e4
        coords = barycentric_grid(order = ((-1, -2), (-1, 2), (1, -2), (1, 2)),
                                  spacing = (0, 3, 3),
                                  filter_negative = True)

        assert_sum_equal_one(coords)

        jnp.allclose([[1.0, 0.0, 0.0, 0.0],
                      [0.5, 0.5, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.5, 0.0, 0.5, 0.0],
                      [0.25, 0.25, 0.25, 0.25],
                      [0.0, 0.5, 0.0, 0.5],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.5, 0.5],
                      [0.0, 0.0, 0.0, 1.0]], coords)



if __name__ == '__main__':
    unittest.main()
