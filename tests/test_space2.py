import unittest
import os

from nndt.space2 import *


class MyTestCase(unittest.TestCase):

    def test_load_from_path(self):
        space = load_from_path("./test_folder_tree")
        print(space.explore())

    def test_load_from_path2(self):
        space = load_from_path("./acdc_for_test")
        print(space.explore())

    def test_forbidden_name(self):
        with self.assertRaises(ValueError) as context:
            space = Space('children')


if __name__ == '__main__':
    unittest.main()
