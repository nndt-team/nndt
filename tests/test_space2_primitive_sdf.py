import unittest

import nndt.space2 as spc
from nndt.primitive_sdf import *
from tests.base import PATH_TEST_ACDC, BaseTestCase


class PrimitiveSDFTestCase(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_sdf_primitive_sphere(self):
        ss = SphereSDF()
        vec_prim = ss.vec_fun
        vec_prim_x = ss.vec_fun_dx
        vec_prim_y = ss.vec_fun_dy
        vec_prim_z = ss.vec_fun_dz
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
        ss = SphereSDF(
            center=(1.0, 1.0, 1.0), radius=float(jnp.sqrt(1**2 + 1**2 + 1**2))
        )
        vec_prim = ss.vec_fun
        vec_prim_x = ss.vec_fun_dx
        vec_prim_y = ss.vec_fun_dy
        vec_prim_z = ss.vec_fun_dz
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

    def test_add_sphere(self):
        self.space = spc.load_from_path(PATH_TEST_ACDC)
        self.space.preload("shift_and_scale")
        space = spc.add_sphere(
            self.space, "for_test", center=(1.0, 1.0, 1.0), radius=1.0
        )
        space.for_test.sphere.unload_from_memory()
        self.assertTrue(
            bool(
                jnp.allclose(
                    jnp.array([0.0]),
                    space.for_test.surface_xyz2sdt((jnp.array([2.0, 1.0, 1.0]))),
                )
            )
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    jnp.array([0.0]),
                    space.for_test.surface_xyz2sdt((jnp.array([1.0, 2.0, 1.0]))),
                )
            )
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    jnp.array([0.0]),
                    space.for_test.surface_xyz2sdt((jnp.array([1.0, 1.0, 2.0]))),
                )
            )
        )
        self.assertTrue(((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)), space.for_test.bbox)
        print(space.print("full"))


if __name__ == "__main__":
    unittest.main()
