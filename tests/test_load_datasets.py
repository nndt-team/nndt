import os
import shutil

from nndt.datasets import ACDC, MICCAI2015
from tests.base import BaseTestCase

DATASETS_FOLDER = '.datasets'
DATA_FOLDER = DATASETS_FOLDER + '/ACDC_5'


class DatasetLoadingCase(BaseTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def tearDown(self) -> None:
        if os.path.exists(f'./{DATASETS_FOLDER}'):
            shutil.rmtree(f'./{DATASETS_FOLDER}')

    def test_download_ACDC(self):
        ACDC().load()
        self.assertTrue(
            len(os.listdir(f'./{DATA_FOLDER}/')) == 5
        )
        self.assertFalse(
            os.path.isfile(f'./{DATA_FOLDER}/temp.zip')
        )

    def test_download_from_dropbox(self):
        ACDC('dropbox_test').load()
        self.assertTrue(
            len(os.listdir(f'./{DATA_FOLDER}/')) == 5
        )
        self.assertFalse(
            os.path.isfile(f'./{DATA_FOLDER}/temp.zip')
        )

    def test_wrong_url(self):
        with self.assertRaises(ConnectionError):
            ACDC('wrong_url_test').load()

    def test_random_name(self):
        with self.assertRaises(ValueError):
            ACDC('random_name').load()

    def test_wrong_hash(self):
        with self.assertRaises(ConnectionError):
            ACDC('wrong_hash_test').load()

    def test_dataset_list(self):
        _list = MICCAI2015().dataset_list()
        self.assertTrue(_list is not None)
        self.assertTrue(len(_list) > 0)
        self.assertTrue(len([item for item in _list if '_test' in item]) == 0)
