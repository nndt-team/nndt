import unittest

import jax
import jax.numpy as jnp

from nndt.math_core import grid_in_cube2
from nndt.space2 import load_from_path

FILE_TMP = "./test_file.space"
FILE_TMP2 = "./test_file2.space"

PATH_TEST_STRUCTURE = './test_folder_tree'
PATH_TEST_ACDC = './acdc_for_test'


class MethodSetTestCase(unittest.TestCase):

    def cmp_array(self, arr0, arr1, atol=0.1):
        arr0_ = jnp.array(arr0)
        arr1_ = jnp.array(arr1)
        print(arr1_)
        self.assertTrue(bool(jnp.allclose(arr0_, arr1_, atol=atol, rtol=0.0)))
        return arr1_

    def helper_sampling_presence(self, path):
        space = load_from_path(path)
        space.preload()
        print(space.print('default'))
        space.init()
        print(space.print('full'))
        rng_key = jax.random.PRNGKey(42)
        ret1 = space.sampling.sampling_grid()
        ret2 = space.sampling.sampling_grid_with_noise(rng_key, sigma=0.0000001)
        ret3 = space.sampling.sampling_uniform(rng_key)
        jnp.allclose(jnp.zeros((2, 2, 2, 3)), ret1)
        jnp.allclose(jnp.zeros((2, 2, 2, 3)), ret2)
        jnp.allclose(jnp.zeros((100, 3)), ret3)

    def test_sampling_presence(self):
        self.helper_sampling_presence(PATH_TEST_ACDC)

    def helper_transform_load(self, mode="identity"):
        space = load_from_path(PATH_TEST_ACDC)
        space.preload(mode=mode, keep_in_memory=False)
        print(space.print("full"))

        return space

    def test_transform_identity(self):
        space = self.helper_transform_load(mode="identity")

        ps_grid = grid_in_cube2(spacing=(2, 2, 2), lower=(59.0, 112.0, 7.0), upper=(134.0, 183.0, 84.0))
        ns_grid = grid_in_cube2(spacing=(2, 2, 2), lower=(59.0, 112.0, 7.0), upper=(134.0, 183.0, 84.0))

        self.assertEqual(7.0, jnp.array(space.patient009.transform.transform_xyz_ps2ns(ps_grid)).min())
        self.assertEqual(183.0, jnp.array(space.patient009.transform.transform_xyz_ps2ns(ps_grid)).max())
        self.assertEqual(7.0, jnp.array(space.patient009.transform.transform_xyz_ns2ps(ns_grid)).min())
        self.assertEqual(183.0, jnp.array(space.patient009.transform.transform_xyz_ns2ps(ns_grid)).max())

    def test_transform_to_cube(self):
        space = self.helper_transform_load(mode="to_cube")

        ps_grid = grid_in_cube2(spacing=(2, 2, 2), lower=(59.0, 112.0, 7.0), upper=(134.0, 183.0, 84.0))
        ns_grid = grid_in_cube2(spacing=(2, 2, 2), lower=(-1, -1, -1), upper=(1, 1, 1))

        self.assertEqual(-1., float(jnp.array(space.patient009.transform.transform_xyz_ps2ns(ps_grid)).min()))
        self.assertEqual(1., float(jnp.array(space.patient009.transform.transform_xyz_ps2ns(ps_grid)).max()))
        self.assertEqual(7.0, float(jnp.array(space.patient009.transform.transform_xyz_ns2ps(ns_grid)).min()))
        self.assertEqual(183.0, float(jnp.array(space.patient009.transform.transform_xyz_ns2ps(ns_grid)).max()))

    def test_transform_shift_and_scale(self):
        space = self.helper_transform_load(mode="shift_and_scale")

        ps_grid = grid_in_cube2(spacing=(2, 2, 2), lower=(59.0, 112.0, 7.0), upper=(134.0, 183.0, 84.0))
        ns_grid = grid_in_cube2(spacing=(2, 2, 2), lower=(-0.75, -0.71, -0.77), upper=(0.75, 0.71, 0.77))

        self.assertAlmostEqual(-0.7699, float(jnp.array(space.patient009.transform.transform_xyz_ps2ns(ps_grid)).min()), places=2)
        self.assertAlmostEqual(0.7699, float(jnp.array(space.patient009.transform.transform_xyz_ps2ns(ps_grid)).max()), places=2)
        self.assertAlmostEqual(7.0, float(jnp.array(space.patient009.transform.transform_xyz_ns2ps(ns_grid)).min()), places=2)
        self.assertAlmostEqual(183.0, float(jnp.array(space.patient009.transform.transform_xyz_ns2ps(ns_grid)).max()), places=2)


if __name__ == '__main__':
    unittest.main()
