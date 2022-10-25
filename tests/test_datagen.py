from tests.base import BaseTestCase
import nndt.space2 as spc
from nndt.datagen import DataGenForSegmentation
from tests.base import BaseTestCase, PATH_TEST_STRUCTURE, PATH_TEST_ACDC
from jax.random import PRNGKey, split
import jax.numpy as jnp


class test_datagen(BaseTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_generate_data_for_segmentation(self):
        space = spc.load_from_path(PATH_TEST_ACDC)
        space.preload("shift_and_scale")
        gen = DataGenForSegmentation(space, augment=False)
        X, Y = [], []
        key = PRNGKey(42)

        for i in range(10):
            key, subkey = split(key)
            res = gen.get(subkey, i)
            X.append(res[0])
            Y.append(res[1])

        X = jnp.stack(X, axis=0)
        Y = jnp.stack(Y, axis=0)
        self.assertEqual(X.shape, (10, 165, 16, 16, 16, 1))
        self.assertEqual(Y.shape, (10, 165))
