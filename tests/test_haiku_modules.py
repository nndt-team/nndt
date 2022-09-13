import unittest
from typing import *

import haiku as hk
import jax
import jax.numpy as jnp

from haiku_modules import DescConv


class HaikuModulesTestCase(unittest.TestCase):

    def test_DescConv_init(self):
        def init(X):
            dc = DescConv(n_layers=1, kernels_in_first_layer=1,
                          kernel_shape=(2, 2, 2),
                          stride=(2, 2, 2),
                          activation=jax.nn.relu)
            return dc(X)

        rng_key = jax.random.PRNGKey(42)
        init, nn = hk.transform(init)
        params = init(rng_key,
                      jnp.zeros((100, 16, 16, 16, 1)))
        result = nn(params, rng_key, jnp.zeros((200, 16, 16, 16, 1)))

        self.assertEqual((200, 8, 8, 8, 1), result.shape)

    def test_DescConv_compress(self):
        def init(X):
            dc = DescConv(n_layers=3, kernels_in_first_layer=3,
                          kernel_shape=(2, 2, 2),
                          stride=(2, 2, 2),
                          activation=jax.nn.relu)
            return dc(X)

        rng_key = jax.random.PRNGKey(42)
        init, nn = hk.transform(init)
        params = init(rng_key,
                      jnp.zeros((100, 16, 16, 16, 1)))
        result = nn(params, rng_key, jnp.zeros((200, 16, 16, 16, 1)))

        self.assertEqual((200, 2, 2, 2, 12), result.shape)

    def test_DescConv_relu_on_output(self):
        def init(X):
            dc = DescConv(n_layers=3, kernels_in_first_layer=3,
                          kernel_shape=(2, 2, 2),
                          stride=(2, 2, 2),
                          activation=jax.nn.relu)
            return dc(X)

        rng_key = jax.random.PRNGKey(42)
        init, nn = hk.transform(init)
        params = init(rng_key,
                      jnp.zeros((100, 16, 16, 16, 1)))
        result = nn(params, rng_key, jnp.ones((200, 16, 16, 16, 1)))
        print(result)

        self.assertTrue(bool(jnp.all(result >= 0.)))
        self.assertTrue(bool(jnp.any(result >= 0.1)))

if __name__ == '__main__':
    unittest.main()
