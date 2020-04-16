import numpy as np
import time
from genieclust.tools import *

# np.random.seed(123)
def test_argsort():
    for n in [1, 10, 1_000]:

        x = np.arange(n)
        t0 = time.time()
        y1a = argsort(x, False)
        print("(ascending)  n=%18d: %10.3fs" % (n, time.time()-t0), end="\t")
        t0 = time.time()
        y2a = np.argsort(x)
        print("%10.3fs" % (time.time()-t0,))
        assert np.all(y1a == y2a)

        x = np.arange(n)[::-1].copy()
        t0 = time.time()
        y1b = argsort(x, False)
        print("(descending)                       %10.3fs" % (time.time()-t0,), end="\t")
        t0 = time.time()
        y2b = np.argsort(x)
        print("%10.3fs" % (time.time()-t0,))
        assert np.all(y1b == y2b)

        x = np.round(np.random.rand(n), 5)
        t0 = time.time()
        y1c = argsort(x, True)
        print("(random-stable)                    %10.3fs" % (time.time()-t0,), end="\t")
        t0 = time.time()
        y2c = np.argsort(x, kind="mergesort")
        print("%10.3fs" % (time.time()-t0,))
        assert np.all(y1c == y2c)


def test_argkmin():
    for n in [1, 5, 1_000]:
        for k in [k for k in [0, 1, 2, 4, 10, 25, 50, 100] if k < n]:
            x = np.arange(n)
            t0 = time.time()
            y1a = argkmin(x, k)
            print("(ascending)  n=%10d, k=%4d: %10.3fs" % (n, k, time.time()-t0), end="\t")
            t0 = time.time()
            y2a = np.argsort(x, kind='mergesort')[k]
            print("%10.3fs" % (time.time()-t0,))
            assert y1a == y2a

            x = np.arange(n)[::-1].copy()
            t0 = time.time()
            y1b = argkmin(x, k)
            print("(descending)                       %10.3fs" % (time.time()-t0,), end="\t")
            t0 = time.time()
            y2b = np.argsort(x, kind='mergesort')[k]
            print("%10.3fs" % (time.time()-t0,))
            assert y1b == y2b

            x = np.round(np.random.rand(n), 5)
            t0 = time.time()
            y1c = argkmin(x, k)
            print("(random)                           %10.3fs" % (time.time()-t0,), end="\t")
            t0 = time.time()
            y2c = np.argsort(x, kind='mergesort')[k]
            print("%10.3fs" % (time.time()-t0,))
            assert y1c == y2c


def test_cummin():
    for n in [1, 5, 1_000]:
        x = np.random.rand(n)
        cx = cummin(x)
        for i in range(n):
            assert cx[i] == min(x[:(i+1)])

        x = np.sort(x)
        cx = cummin(x)
        for i in range(n):
            assert cx[i] == min(x[:(i+1)])

        x = x[::-1]
        cx = cummin(x)
        for i in range(n):
            assert cx[i] == min(x[:(i+1)])


def test_cummax():
    for n in [1, 5, 1_000]:
        x = np.random.rand(n)
        cx = cummax(x)
        for i in range(n):
            assert cx[i] == max(x[:(i+1)])

        x = np.sort(x)
        cx = cummax(x)
        for i in range(n):
            assert cx[i] == max(x[:(i+1)])

        x = x[::-1]
        cx = cummax(x)
        for i in range(n):
            assert cx[i] == max(x[:(i+1)])



if __name__ == "__main__":
    print("\n\nargsort\n")
    test_argsort()
    print("\n\nargkmin\n")
    test_argkmin()
    print("\n\ncummin\n")
    test_cummin()
    print("\n\ncummax\n")
    test_cummax()

