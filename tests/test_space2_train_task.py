import unittest

import jax
import jax.numpy as jnp

from nndt.math_core import grid_in_cube2
from nndt.space2 import load_from_path
from tests.base import BaseTestCase, PATH_TEST_ACDC

FILE_TMP_IR1 = "./test_file.ir1"


class TrainTaskSetNodeTestCase(BaseTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_transform_inversions(self):
        space = load_from_path(PATH_TEST_ACDC)
        space.preload('shift_and_scale')
        space.patient009.train_task_sdt2sdf(FILE_TMP_IR1, epochs=1)


if __name__ == '__main__':
    unittest.main()
