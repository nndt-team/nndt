import unittest

import jax
import jax.numpy as jnp

from nndt.space2 import load_from_path

FILE_TMP = "./test_file.space"
FILE_TMP2 = "./test_file2.space"

PATH_TEST_STRUCTURE = './test_folder_tree'
PATH_TEST_ACDC = './acdc_for_test'


class MethodSetTestCase(unittest.TestCase):

    def helper_sampling_presence(self, path):
        space = load_from_path(path)
        space.preload()
        rng_key = jax.random.PRNGKey(42)
        ret1 = space.sampling.sampling_grid()
        ret2 = space.sampling.sampling_grid_with_noise(rng_key, sigma=0.0000001)
        ret3 = space.sampling.sampling_uniform(rng_key)
        jnp.allclose(jnp.zeros((2, 2, 2, 3)), ret1)
        jnp.allclose(jnp.zeros((2, 2, 2, 3)), ret2)
        jnp.allclose(jnp.zeros((100, 3)), ret3)

        print(space.print('default'))
        print(space.print('full'))

    def test_sampling_exists_in_the_space(self):
        self.helper_sampling_presence(PATH_TEST_ACDC)
        self.helper_sampling_presence(PATH_TEST_STRUCTURE)


class CheckAllMethodsTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.space = load_from_path(PATH_TEST_ACDC)
        self.space.preload('shift_and_scale')
        print(self.space.print())
        self.rng_key = jax.random.PRNGKey(42)

    def test_sampling_eachN_from_mesh(self):
        ind, xyz = self.space.patient009.sampling_eachN_from_mesh(count=100, step=7, shift=3)
        self.assertEqual((100,), ind.shape)
        self.assertEqual((100, 3), xyz.shape)

    def test_sampling_grid(self):
        ret = self.space.patient009.sampling_grid()
        self.assertEqual((2, 2, 2, 3), ret.shape)
        ret = self.space.patient009.sampling_grid(spacing=(5, 5, 5))
        self.assertEqual((5, 5, 5, 3), ret.shape)
        ret = self.space.patient009.sampling_grid((5, 5, 5))
        self.assertEqual((5, 5, 5, 3), ret.shape)

    def test_sampling_grid_with_noise(self):
        ret = self.space.patient009.sampling_grid_with_noise(self.rng_key,
                                                             spacing=(5, 5, 5),
                                                             sigma=1.)
        self.assertEqual((5, 5, 5, 3), ret.shape)

    def test_sampling_uniform(self):
        ret = self.space.patient009.sampling_uniform(self.rng_key, 100)
        self.assertEqual((100, 3), ret.shape)

    def test_xyz2ind_ind2xyz(self):
        ns_dist, ind = self.space.patient009.surface_xyz2ind(jnp.array([[-0.75312406, -0.01604767, -0.69798934]]))
        point_xyz = self.space.patient009.surface_ind2xyz(ind)
        ns_dist2 = jnp.linalg.norm(point_xyz - jnp.array([-0.75312406, -0.01604767, -0.69798934]))
        self.assertAlmostEqual(float(ns_dist), float(ns_dist2))

        ns_dist, ind = self.space.patient009.surface_xyz2ind(jnp.array([[2., 2., 2.]]))
        point_xyz = self.space.patient009.surface_ind2xyz(ind)
        ns_dist2 = jnp.linalg.norm(point_xyz - jnp.array([2., 2., 2.]))
        self.assertAlmostEqual(float(ns_dist), float(ns_dist2))

    def test_surface_colors(self):
        rgba = self.space.patient009.surface_ind2rgba(jnp.array([0, 1, 2]))
        self.assertTrue(bool(jnp.allclose(jnp.array((0.937255, 0.937255, 0.937255)), rgba[:, 0])))
        self.assertTrue(bool(jnp.allclose(jnp.array((0.156863, 0.156863, 0.156863)), rgba[:, 1])))
        self.assertTrue(bool(jnp.allclose(jnp.array((0.156863, 0.156863, 0.156863)), rgba[:, 2])))
        self.assertTrue(bool(jnp.allclose(jnp.array((1., 1., 1.)), rgba[:, 3])))

    def test_surface_colors_xyz(self):
        rgba = self.space.patient009.surface_xyz2rgba(jnp.array([[-0.75312406, -0.01604767, -0.69798934]]))
        self.assertTrue(bool(jnp.allclose(jnp.array((0.937255,)), rgba[:, 0])))
        self.assertTrue(bool(jnp.allclose(jnp.array((0.156863,)), rgba[:, 1])))
        self.assertTrue(bool(jnp.allclose(jnp.array((0.156863,)), rgba[:, 2])))
        self.assertTrue(bool(jnp.allclose(jnp.array((1.,)), rgba[:, 3])))


if __name__ == '__main__':
    unittest.main()
