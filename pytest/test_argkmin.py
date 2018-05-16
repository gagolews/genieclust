import numpy as np
import time
from genieclust.internal import argkmin


# np.random.seed(123)
def test_argkmin():
    for n in [1, 5, 100_000]:
        for k in [k for k in [1, 2, 5, 10, 25, 50, 100] if k < n]:

            x = np.arange(n)
            t0 = time.time()
            y1a = argkmin(x, k)
            print("(ascending)  n=%10d, k=%4d: %10.3fs" % (n, k, time.time()-t0), end="\t")
            t0 = time.time()
            y2a = np.argsort(x)[k]
            print("%10.3fs" % (time.time()-t0,))
            assert y1a == y2a

            x = np.arange(n)[::-1]
            t0 = time.time()
            y1b = argkmin(x, k)
            print("(descending)                       %10.3fs" % (time.time()-t0,), end="\t")
            t0 = time.time()
            y2b = np.argsort(x)[k]
            print("%10.3fs" % (time.time()-t0,))
            assert y1b == y2b

            x = np.random.rand(n)
            t0 = time.time()
            y1c = argkmin(x, k)
            print("(random)                           %10.3fs" % (time.time()-t0,), end="\t")
            t0 = time.time()
            y2c = np.argsort(x)[k]
            print("%10.3fs" % (time.time()-t0,))
            assert y1c == y2c

if __name__ == "__main__":
    test_argkmin()
