import unittest

import jax
import jax.numpy as jnp

from space2 import load_from_path

FILE_TMP = "./test_file.space"
FILE_TMP2 = "./test_file2.space"

PATH_TEST_STRUCTURE = './test_folder_tree'
PATH_TEST_ACDC = './acdc_for_test'

class LoadersTestCase(unittest.TestCase):

    def helper_initialization_call(self, path, keep_in_memory):
        space = load_from_path(path)
        space.initialization(keep_in_memory=keep_in_memory)
        print(space.explore('default'))
        print(space.explore('full'))

    def test_initialization_call(self):
        self.helper_initialization_call(PATH_TEST_ACDC, keep_in_memory=False)
        self.helper_initialization_call(PATH_TEST_STRUCTURE, keep_in_memory=False)

    def test_initialization_call_and_keep_in_memory(self):
        self.helper_initialization_call(PATH_TEST_ACDC, keep_in_memory=True)
        self.helper_initialization_call(PATH_TEST_STRUCTURE, keep_in_memory=True)


if __name__ == '__main__':
    unittest.main()
