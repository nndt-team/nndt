import os
import unittest

from py7zr import py7zr

PATH_TEST_STRUCTURE = './tree_for_test'
PATH_TEST_ACDC = './acdc_for_test'


class BaseTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("WORKING DIR: " + os.getcwd())

        src_target = [PATH_TEST_STRUCTURE, PATH_TEST_ACDC]
        if os.path.basename(os.getcwd()) == 'tests':
            src_file = [PATH_TEST_STRUCTURE + '.7z', PATH_TEST_ACDC + '.7z']
        elif os.getcwd() == '/home/runner/work/nndt/nndt':  # TODO this is shit!
            src_file = ['../tests/' + PATH_TEST_STRUCTURE + '.7z', '../tests/' + PATH_TEST_ACDC + '.7z']
        elif os.path.basename(os.getcwd()) == 'nndt':
            src_file = ['./tests/' + PATH_TEST_STRUCTURE + '.7z', './tests/' + PATH_TEST_ACDC + '.7z']
        else:
            raise NotImplementedError("Something goes wrong with path...")

        for src, target in zip(src_file, src_target):
            if not os.path.exists(target):
                with py7zr.SevenZipFile(src, mode='r') as z:
                    z.extractall(path='./')


if __name__ == '__main__':
    unittest.main()
