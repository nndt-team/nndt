import jax.numpy as jnp
from jax.random import PRNGKey, split

import nndt.space2 as spc
from nndt.datagen import DataGenForSegmentation, DataGenForShapeRegression
from tests.base import PATH_TEST_ACDC, BaseTestCase


class DatagenTestCase(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_generate_data_for_segmentation(self):
        space = spc.load_from_path(PATH_TEST_ACDC)
        space.preload("shift_and_scale")
        gen = DataGenForSegmentation(space, augment=False)
        x, y = [], []
        key = PRNGKey(42)

        for i in range(10):
            key, subkey = split(key)
            res = gen.get(subkey, i)
            x.append(res[0])
            y.append(res[1])

        x = jnp.stack(x, axis=0)
        y = jnp.stack(y, axis=0)
        self.assertEqual(x.shape, (10, 165, 16, 16, 16, 1))
        self.assertEqual(y.shape, (10, 165))

    def test_generate_data_for_shape_regression(self):
        space = spc.load_from_path(PATH_TEST_ACDC)
        space.preload("shift_and_scale")
        gen = DataGenForShapeRegression(space, augment=False)
        key = PRNGKey(42)
        key, subkey = split(key)
        res = gen.get(subkey, 0)
        self.assertEqual(res.X.shape, (5 * 4 * 4 * 4,))
        self.assertEqual(res.Y.shape, (5 * 4 * 4 * 4,))
        self.assertEqual(res.Z.shape, (5 * 4 * 4 * 4,))
        self.assertEqual(res.T.shape, (5 * 4 * 4 * 4,))
        self.assertEqual(res.P.shape, (5 * 4 * 4 * 4, 5))
        self.assertEqual(res.SDF.shape, (5 * 4 * 4 * 4,))
