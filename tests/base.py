import os
import unittest

from py7zr import py7zr

PATH_TEST_STRUCTURE = './test_folder_tree'
PATH_TEST_ACDC = './acdc_for_test'


class BaseTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("WORKING DIR: " + os.getcwd())

        if os.path.basename(os.getcwd()) == 'tests':
            prefix = './'
            iter_dir = [PATH_TEST_STRUCTURE, PATH_TEST_ACDC]
        elif os.path.basename(os.getcwd()) == 'nndt':
            prefix = './tests/'
            iter_dir = ['./tests/' + PATH_TEST_STRUCTURE, './tests/' + PATH_TEST_ACDC]
        else:
            raise NotImplementedError("Something goes wrong with path...")

        for path in iter_dir:
            if not os.path.exists(path):
                with py7zr.SevenZipFile(path + '.7z', mode='r') as z:
                    z.extractall(path=prefix)


if __name__ == '__main__':
    unittest.main()
