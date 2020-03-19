import numpy as np
import sklearn.metrics
from genieclust.compare_partitions import *


# np.random.seed(123)
def test_compare_partitions():
    for n in [10, 25, 50, 100, 250, 500, 1000]:
        for k in range(2, 11):

            x = np.random.choice(np.arange(k), n)
            y = np.random.choice(np.arange(k), n)

            ari1 = adjusted_rand_score(x, y)
            ari2 = sklearn.metrics.adjusted_rand_score(x, y)
            assert abs(ari1-ari2)<1e-9

            fm1 = fm_score(x, y)
            fm2 = sklearn.metrics.fowlkes_mallows_score(x, y)
            assert abs(fm1-fm2)<1e-9

            y = x.copy()
            c = np.random.permutation(np.arange(k+1))
            for i in range(n): y[i] = c[y[i]]
            assert adjusted_rand_score(x, y)>1.0-1e-9
            assert rand_score(x, y)>1.0-1e-9
            assert adjusted_fm_score(x, y)>1.0-1e-9
            assert fm_score(x, y)>1.0-1e-9

            # TODO: more tests...



if __name__ == "__main__":
    test_compare_partitions()
