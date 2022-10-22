import os
import unittest

from py7zr import py7zr

PATH_TEST_STRUCTURE = './test_folder_tree'
PATH_TEST_ACDC = './acdc_for_test'

class BaseTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for path in [PATH_TEST_STRUCTURE, PATH_TEST_ACDC]:
            if not os.path.exists(path):
                with py7zr.SevenZipFile(path+'.7z', mode='r') as z:
                    z.extractall(path='./')

if __name__ == '__main__':
    unittest.main()
