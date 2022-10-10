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
        ret1 = space.sampling.grid()
        ret2 = space.sampling.grid_with_shackle(rng_key, sigma=0.0000001)
        ret3 = space.sampling.uniform(rng_key)
        jnp.allclose(jnp.zeros((2, 2, 2, 3)), ret1)
        jnp.allclose(jnp.zeros((2, 2, 2, 3)), ret2)
        jnp.allclose(jnp.zeros((100, 3)), ret3)

        print(space.print('default'))
        print(space.print('full'))

    def test_sampling_exists_in_the_space(self):
        self.helper_sampling_presence(PATH_TEST_ACDC)
        self.helper_sampling_presence(PATH_TEST_STRUCTURE)

    def test_all_methods_in_Object3D(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.preload()
        rng_key = jax.random.PRNGKey(42)
        ret = space.patient009.grid()
        self.assertEqual((2, 2, 2, 3), ret.shape)
        ret = space.patient009.grid(spacing=(5, 5, 5))
        self.assertEqual((5, 5, 5, 3), ret.shape)
        ret = space.patient009.grid((5, 5, 5))
        self.assertEqual((5, 5, 5, 3), ret.shape)

if __name__ == '__main__':
    unittest.main()
