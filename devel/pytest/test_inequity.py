import numpy as np
from genieclust.inequity import *

# np.random.seed(123)
def test_inequity():
    def gini_ref(x):
        n = len(x)
        s = 0.0
        for i in range(n-1):
            for j in range(i+1,n):
                s += np.abs(x[i]-x[j])
        return s / (n-1) / np.sum(x)

    def bonferroni_ref(x):
        n = len(x)
        x = np.sort(x)[::-1]
        s = 0.0
        for i in range(1,n+1):
            for j in range(i,n+1):
                s += x[j-1]/(n-i+1)
        return n  * (1.0- s / np.sum(x)) / (n-1)


    for n in [2, 5, 100]:
        for i in range(10):
            if i == 0:
                x = np.r_[[1]*n]
            elif i == 1:
                x = np.r_[1, [0]*(n-1)]
            else:
                x = np.random.random(n)*8 + 3

            xg1 = gini_index(np.array(x))
            xg2 = gini_index(np.array(x, dtype=np.float_))
            xg3 = gini_ref(x)
            assert abs(xg1 - xg2) < 1e-9
            assert abs(xg1 - xg3) < 1e-9

            xb1 = bonferroni_index(np.array(x))
            xb2 = bonferroni_index(np.array(x, dtype=np.float_))
            xb3 = bonferroni_ref(x)
            assert abs(xb1 - xb2) < 1e-9
            assert abs(xb1 - xb3) < 1e-9

            if n > 2:
                x = np.sort(x)
                assert gini_index(x[::2],True) == gini_index(np.array(x[::2]))
                assert bonferroni_index(x[::2], True) == bonferroni_index(np.array(x[::2]))

if __name__ == "__main__":
    test_inequity()
