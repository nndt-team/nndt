import unittest

from nndt.space2 import load_from_path, load_implicit_ir1
from tests.base import PATH_TEST_ACDC, BaseTestCase

FILE_TMP_IR1 = "./test_file.ir1"


class TrainTaskSetNodeTestCase(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_transform_inversions(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.preload("shift_and_scale")
        space.patient009.train_task_sdt2sdf(FILE_TMP_IR1, epochs=1)

    def test_load_ir1(self):
        space = load_implicit_ir1(FILE_TMP_IR1)
        space.preload("shift_and_scale")
        print(space.default.test_file_ir1)
        print(space.print())


if __name__ == "__main__":
    unittest.main()
