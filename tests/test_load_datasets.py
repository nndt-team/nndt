from datasets import utils  # TODO: replace utils with datasets.py
import unittest
import shutil
import os

DATA_FOLDER = 'ACDC_5'


class DatasetUnavailableCase(unittest.TestCase):
    def test_download_tests(self):
        with self.assertRaises(ConnectionError):
            utils.load('test')


class DatasetLoadingCase(unittest.TestCase):
    def tearDown(self) -> None:
        shutil.rmtree(f'./{DATA_FOLDER}')

    def test_download_ACDC(self):
        utils.load('ACDC_5')
        self.assertTrue(os.path.exists(f'./{DATA_FOLDER}/patient009'))
        self.assertTrue(os.path.exists(f'./{DATA_FOLDER}/patient029'))
        self.assertTrue(os.path.exists(f'./{DATA_FOLDER}/patient049'))
        self.assertTrue(os.path.exists(f'./{DATA_FOLDER}/patient069'))
        self.assertTrue(os.path.exists(f'./{DATA_FOLDER}/patient089'))
        self.assertFalse(os.path.isfile(f'./temp.zip'))

    def test_download_from_dropbox(self):
        utils.load('test2')
        self.assertTrue(os.path.exists(f'./{DATA_FOLDER}/patient009'))
        self.assertTrue(os.path.exists(f'./{DATA_FOLDER}/patient029'))
        self.assertTrue(os.path.exists(f'./{DATA_FOLDER}/patient049'))
        self.assertTrue(os.path.exists(f'./{DATA_FOLDER}/patient069'))
        self.assertTrue(os.path.exists(f'./{DATA_FOLDER}/patient089'))
        self.assertFalse(os.path.isfile(f'./temp.zip'))
