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
        rng_key = jax.random.PRNGKey(42)
        ret1 = space.sampling.grid()
        ret2 = space.sampling.grid_with_shackle(rng_key, sigma=0.0000001)
        ret3 = space.sampling.uniform(rng_key)
        jnp.allclose(jnp.zeros((2, 2, 2, 3)), ret1)
        jnp.allclose(jnp.zeros((2, 2, 2, 3)), ret2)
        jnp.allclose(jnp.zeros((100, 3)), ret3)

        print(space.print('default'))
        print(space.print('full'))

    def test_sampling_presence(self):
        self.helper_sampling_presence(PATH_TEST_ACDC)
        self.helper_sampling_presence(PATH_TEST_STRUCTURE)


if __name__ == '__main__':
    unittest.main()
