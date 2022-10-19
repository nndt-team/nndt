
import jax
import jax.numpy as jnp

from timeit import default_timer as timer


if __name__=="__main__":
    start = timer()
    for i in range(10):
        key = jax.random.PRNGKey(42)
    end = timer()
    print("Key generation: ", (end - start)/10)

    start = timer()
    for i in range(10):
        #####
        key = jax.random.PRNGKey(42)
        #####
    end = timer()
    print("Key generation: ", (end - start) / 10)

    key = jax.random.PRNGKey(42)
    start = timer()
    N = 50
    for i in range(N):
        #####
        jax.random.uniform(key, shape=(1000*16*16*16, 3), minval=1. - 0.3, maxval=1. + 0.3)
        #####
    end = timer()
    print("Uniform random without subkey split: ", (end - start) / N)

    key = jax.random.PRNGKey(42)
    start = timer()
    N = 50
    for i in range(N):
        #####
        key, subkey = jax.random.split(key)
        jax.random.uniform(subkey, shape=(1000 * 16 * 16 * 16, 3), minval=1. - 0.3, maxval=1. + 0.3)
        #####
    end = timer()
    print("Uniform random with subkey split: ", (end - start) / N)

