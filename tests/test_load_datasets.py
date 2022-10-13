from nndt.datasets import utils
import unittest
import shutil
import os

DATASETS_FOLDER = '.datasets'
DATA_FOLDER = DATASETS_FOLDER + '/ACDC_5'


class DatasetLoadingCase(unittest.TestCase):
    def tearDown(self) -> None:
        if os.path.exists(f'./{DATASETS_FOLDER}'):
            shutil.rmtree(f'./{DATASETS_FOLDER}')

    def test_download_tests(self):
        with self.assertRaises(ConnectionError):
            utils.load('test')

    def test_download_ACDC(self):
        utils.load('ACDC_5')
        self.assertTrue(
            len(os.listdir(f'./{DATA_FOLDER}/')) == 5
        )
        self.assertFalse(
            os.path.isfile(f'./{DATA_FOLDER}/temp.zip')
        )


    def test_download_from_dropbox(self):
        utils.load('test2')
        self.assertTrue(
            len(os.listdir(f'./{DATA_FOLDER}/')) == 5
        )
        self.assertFalse(
            os.path.isfile(f'./{DATA_FOLDER}/temp.zip')
        )


    def test_random_name(self):
        with self.assertRaises(ValueError):
            utils.load('abcds')

    def test_wrong_hash(self):
        with self.assertRaises(ConnectionError):
            utils.load('test_hash')
