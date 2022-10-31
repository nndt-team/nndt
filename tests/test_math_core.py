import unittest

from nndt.math_core import *
from nndt.primitive_sdf import sdf_primitive_sphere
from tests.base import BaseTestCase


class MathCoreTestCase(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_grid_in_cube(self):
        cube = grid_in_cube(spacing=(4, 4, 4), scale=4.0, center_shift=(2.0, 2.0, 2.0))
        self.assertEqual(0.0, float(jnp.min(cube)))
        self.assertEqual(4.0, float(jnp.max(cube)))

    def test_grid_in_cube2(self):
        cube = grid_in_cube2(spacing=(4, 4, 4), lower=(-2, -2, -2), upper=(2, 2, 2))
        self.assertEqual((4, 4, 4, 3), cube.shape)
        self.assertEqual(-2, float(jnp.min(cube)))
        self.assertEqual(2, float(jnp.max(cube)))

    def test_uniform_in_cube(self):
        rng_key = jax.random.PRNGKey(42)
        cube = uniform_in_cube(rng_key, count=100, lower=(-2, -2, -2), upper=(2, 2, 2))
        self.assertEqual((100, 3), cube.shape)
        self.assertLessEqual(-2, float(jnp.min(cube)))
        self.assertGreaterEqual(2, float(jnp.max(cube)))

    def test_take_each_n(self):
        index_set, array = take_each_n(
            jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), count=4, step=4, shift=1
        )
        self.assertEqual((4,), array.shape)
        self.assertEqual((4,), index_set.shape)
        self.assertEqual([1, 5, 9, 3], list(array))
        self.assertEqual([1, 5, 9, 3], list(index_set))

    def test_sdf_primitive_sphere(self):
        vec_prim, vec_prim_x, vec_prim_y, vec_prim_z = sdf_primitive_sphere()
        xyz = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])

        self.assertTrue(
            bool(
                jnp.allclose(
                    jnp.array([-1.0, 0.0, 3.0]),
                    vec_prim(xyz[:, 0], xyz[:, 1], xyz[:, 2]),
                )
            )
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    jnp.array([0.0, 0.0, 0.0]),
                    vec_prim_x(xyz[:, 0], xyz[:, 1], xyz[:, 2]),
                )
            )
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    jnp.array([0.0, 0.0, 0.0]),
                    vec_prim_y(xyz[:, 0], xyz[:, 1], xyz[:, 2]),
                )
            )
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    jnp.array([0.0, 2.0, 4.0]),
                    vec_prim_z(xyz[:, 0], xyz[:, 1], xyz[:, 2]),
                )
            )
        )

    def test_sdf_primitive_sphere2(self):
        vec_prim, vec_prim_x, vec_prim_y, vec_prim_z = sdf_primitive_sphere(
            center=(1.0, 1.0, 1.0), radius=float(jnp.sqrt(1**2 + 1**2 + 1**2))
        )
        xyz = jnp.array([[0.0, 0.0, 0.0]])

        self.assertTrue(
            bool(
                jnp.allclose(
                    jnp.array([0.0, 0.0, 0.0]),
                    vec_prim(xyz[:, 0], xyz[:, 1], xyz[:, 2]),
                )
            )
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    jnp.array([-2.0, -2.0, -2.0]),
                    vec_prim_x(xyz[:, 0], xyz[:, 1], xyz[:, 2]),
                )
            )
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    jnp.array([-2.0, -2.0, -2.0]),
                    vec_prim_y(xyz[:, 0], xyz[:, 1], xyz[:, 2]),
                )
            )
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    jnp.array([-2.0, -2.0, -2.0]),
                    vec_prim_z(xyz[:, 0], xyz[:, 1], xyz[:, 2]),
                )
            )
        )

    def test_train_test_split(self):
        array = jnp.array([i * i for i in range(10)])
        rng = jax.random.PRNGKey(2)
        test_size = 0.2
        train_data_indices, test_data_indices = train_test_split(array, rng, test_size)

        self.assertEqual(test_size, len(test_data_indices) / len(array))
        self.assertEqual(list, type(train_data_indices))
        self.assertEqual(list, type(test_data_indices))
        self.assertEqual(
            [ind for ind in train_data_indices if ind not in test_data_indices],
            train_data_indices,
        )
        self.assertEqual(
            [ind for ind in test_data_indices if ind not in train_data_indices],
            test_data_indices,
        )

        rng_2, _ = jax.random.split(rng)
        train_data_indices_2, test_data_indices_2 = train_test_split(
            array, rng_2, test_size
        )

        self.assertNotEqual(train_data_indices_2, train_data_indices)
        self.assertNotEqual(test_data_indices_2, test_data_indices)


def assert_sum_equal_one(values):
    jnp.allclose(1.0, jnp.sum(values, axis=-1))


class BarycentricGridTestCase(unittest.TestCase):
    def test_helper(self):
        val = help_barycentric_grid((1, -1))
        self.assertEqual("l1*e1 + (1-l1)*e2", val)
        val = help_barycentric_grid((-1, 1))
        self.assertEqual("(1-l1)*e1 + l1*e2", val)
        val = help_barycentric_grid((1, 0, 2))
        self.assertEqual("l1*e1 + (1-(l1+l2))*e2 + l2*e3", val)
        val = help_barycentric_grid((1, (-1, -2), 2))
        self.assertEqual("l1*e1 + (1-l1)*(1-l2)*e2 + l2*e3", val)
        val = help_barycentric_grid(((-1, -2), (-1, 2), (1, -2), (1, 2)))
        self.assertEqual(
            "(1-l1)*(1-l2)*e1 + (1-l1)*l2*e2 + l1*(1-l2)*e3 + l1*l2*e4", val
        )

    def test_linear(self):
        # l1*e1 + (1-l1)*e2
        coords = barycentric_grid(order=(1, -1), spacing=(0, 3))
        assert_sum_equal_one(coords)

        jnp.allclose(jnp.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]), coords)

    def test_linear_inverted(self):
        # (1-l1)*e1 + l1*e2
        coords = barycentric_grid(order=(-1, 1), spacing=(0, 3), filter_negative=True)

        assert_sum_equal_one(coords)

        jnp.allclose(jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]), coords)

    def test_ternary(self):
        # l1*e1 + (1-(l1+l2))*e2 + l2*e3
        coords = barycentric_grid(
            order=(1, 0, 2), spacing=(0, 3, 3), filter_negative=True
        )

        assert_sum_equal_one(coords)

        jnp.allclose(
            jnp.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.0, 0.0, 1.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [1.0, 0.0, 0.0],
                ]
            ),
            coords,
        )

    def test_ternary_without_negative_filter(self):
        # l1*e1 + (1-(l1+l2))*e2 + l2*e3
        coords = barycentric_grid(
            order=(1, 0, 2), spacing=(0, 3, 3), filter_negative=False
        )

        assert_sum_equal_one(coords)

        jnp.allclose(
            jnp.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.0, 0.0, 1.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.5, -0.5, 1.0],
                    [1.0, 0.0, 0.0],
                    [1.0, -0.5, 0.5],
                    [1.0, -1.0, 1.0],
                ]
            ),
            coords,
        )

    def test_replenishment_without_negative_filter(self):
        # l1*e1 + (1-l1)*(1-l2)*e2 + l2*e3
        coords = barycentric_grid(
            order=(1, (-1, -2), 2), spacing=(0, 3, 3), filter_negative=False
        )

        assert_sum_equal_one(coords)

        jnp.allclose(
            jnp.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.0, 0.0, 1.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.25, 0.5],
                    [0.5, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.5],
                    [1.0, 0.0, 1.0],
                ]
            ),
            coords,
        )

    def test_square(self):
        # (1-l1)*(1-l2)*e1 + (1-l1)*l2*e2 + l1*(1-l2)*e3 + l1*l2*e4
        coords = barycentric_grid(
            order=((-1, -2), (-1, 2), (1, -2), (1, 2)),
            spacing=(0, 3, 3),
            filter_negative=True,
        )

        assert_sum_equal_one(coords)

        jnp.allclose(
            jnp.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.5, 0.0, 0.5, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.0, 0.5, 0.0, 0.5],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.5, 0.5],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            coords,
        )

    def test_rotation_matrix(self):
        self.assertTrue(jnp.allclose(rotation_matrix(0.0, 0.0, 0.0), jnp.eye(3)))
        M = rotation_matrix(34.0, 34.0, 424.0)
        self.assertTrue(abs(float(jnp.dot(M[0], M[1]))) < 0.0000001)
        self.assertTrue(abs(float(jnp.dot(M[1], M[2]))) < 0.0000001)
        self.assertTrue(abs(float(jnp.dot(M[0], M[2]))) < 0.0000001)
        self.assertTrue(abs(float(jnp.linalg.norm(M[0])) - 1.0) < 0.000001)
        self.assertTrue(abs(float(jnp.linalg.norm(M[1])) - 1.0) < 0.000001)
        self.assertTrue(abs(float(jnp.linalg.norm(M[2])) - 1.0) < 0.000001)

    def test_scale_xyz(self):
        xyz = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(
            jnp.array([3.0, -8.0, 15.0]), scale_xyz(xyz, scale=(3.0, -4.0, 5.0))
        )
        xyz = jnp.array([0.0, 0.0, 0.0])
        assert jnp.allclose(
            jnp.array([0.0, 0.0, 0.0]), scale_xyz(xyz, scale=(3.0, -4.0, 5.0))
        )


if __name__ == "__main__":
    unittest.main()
