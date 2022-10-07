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

        return space

    def test_initialization_call(self):
        self.helper_initialization_call(PATH_TEST_ACDC, keep_in_memory=False)
        self.helper_initialization_call(PATH_TEST_STRUCTURE, keep_in_memory=False)

    def test_initialization_call_and_keep_in_memory(self):
        self.helper_initialization_call(PATH_TEST_ACDC, keep_in_memory=True)
        self.helper_initialization_call(PATH_TEST_STRUCTURE, keep_in_memory=True)

    def test_initialization_check_access_to_field_mesh(self):
        space = self.helper_initialization_call(PATH_TEST_ACDC, keep_in_memory=False)

        self.assertNotIn('^', space.patient069.colored_obj.explore())
        self.assertIsNotNone(space.patient069.colored_obj._loader.mesh)
        self.assertIn('^', space.patient069.colored_obj.explore())

    def test_initialization_check_access_to_field_text(self):
        space = self.helper_initialization_call(PATH_TEST_STRUCTURE, keep_in_memory=False)

        self.assertNotIn('^', space.group1.patient11.organ110.data1100_txt.explore())
        self.assertIsNotNone(space.group1.patient11.organ110.data1100_txt._loader.text)
        self.assertIn('^', space.group1.patient11.organ110.data1100_txt.explore())


if __name__ == '__main__':
    unittest.main()
