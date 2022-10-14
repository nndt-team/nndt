import unittest

from nndt.primitive_sdf import *
import jax.numpy as jnp

class PrimitiveSDFTestCase(unittest.TestCase):

    def test_sdf_primitive_sphere(self):
        ss = SphereSDF()
        vec_prim = ss.vec_fun
        vec_prim_x = ss.vec_fun_dx
        vec_prim_y = ss.vec_fun_dy
        vec_prim_z = ss.vec_fun_dz
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
        ss = SphereSDF(center=(1., 1., 1.), radius=float(jnp.sqrt(1 ** 2 + 1 ** 2 + 1 ** 2)))
        vec_prim = ss.vec_fun
        vec_prim_x = ss.vec_fun_dx
        vec_prim_y = ss.vec_fun_dy
        vec_prim_z = ss.vec_fun_dz
        xyz = jnp.array([[0., 0., 0.]])

        self.assertTrue(bool(jnp.allclose(jnp.array([0., 0., 0.]),
                                          vec_prim(xyz[:, 0], xyz[:, 1], xyz[:, 2]))))
        self.assertTrue(bool(jnp.allclose(jnp.array([-2., -2., -2.]),
                                          vec_prim_x(xyz[:, 0], xyz[:, 1], xyz[:, 2]))))
        self.assertTrue(bool(jnp.allclose(jnp.array([-2., -2., -2.]),
                                          vec_prim_y(xyz[:, 0], xyz[:, 1], xyz[:, 2]))))
        self.assertTrue(bool(jnp.allclose(jnp.array([-2., -2., -2.]),
                                          vec_prim_z(xyz[:, 0], xyz[:, 1], xyz[:, 2]))))


if __name__ == '__main__':
    unittest.main()
