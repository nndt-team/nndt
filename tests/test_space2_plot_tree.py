import os
import unittest

from nndt.space2 import load_from_path

FILE_TMP_PNG = "./test_file.png"

PATH_TEST_ACDC = './acdc_for_test'

class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        if os.path.exists(FILE_TMP_PNG):
            os.remove(FILE_TMP_PNG)

    def test_plot_colored_obj(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.preload(mode="ident", keep_in_memory=False)
        print(space.print("default"))

        space.patient009.colored_obj.plot(filepath=FILE_TMP_PNG)
        self.assertTrue(os.path.exists(FILE_TMP_PNG))

    def test_plot_sdt_npy(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.preload(mode="ident", keep_in_memory=False)
        print(space.print("default"))

        space.patient009.sdf_npy.plot(filepath=FILE_TMP_PNG)
        self.assertTrue(os.path.exists(FILE_TMP_PNG))

if __name__ == '__main__':
    unittest.main()
