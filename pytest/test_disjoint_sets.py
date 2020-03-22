import numpy as np
from genieclust.inequity import *
from genieclust.internal import DisjointSets
from genieclust.internal import GiniDisjointSets
import time
import gc

np.random.seed(666)

def test_DisjointSets():
    print("test_DisjointSets")
    for n in [5, 10, 25, 100, 250, 1000, 10000]:
        d = DisjointSets(n)
        assert all([i==d.find(i) for i in range(n)])

        for k in range(int(np.random.randint(0, n-2, 1))):
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if d.find(i) == d.find(j): continue
            d.union(i, j)
            assert d.find(i) == d.find(j)

        assert len(np.unique(d.to_list())) == d.get_k()
        assert max(d.to_list_normalized()) == d.get_k()-1
        assert len(d.to_lists()) == d.get_k()


def test_GiniDisjointSets():
    print("test_GiniDisjointSets")
    for n in [5, 10, 25, 100, 250, 1000, 10000]:
        d = GiniDisjointSets(n)
        assert all([i==d.find(i) for i in range(n)])

        for k in range(int(np.random.randint(0, n-2, 1))):
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if d.find(i) == d.find(j): continue
            d.union(i, j)
            assert d.find(i) == d.find(j)
            c1 = d.get_counts()
            assert np.sum(c1) == d.get_n()
            assert min(c1) == d.get_smallest_count()


            # c2 = np.sort([len(x) for x in d.to_lists()])
            # c1 = np.sort(c1)
            # assert np.all(c1 == c2)


            g1 = d.get_gini()
            g2 = gini(np.array(c1), True)
            assert np.allclose(g1, g2)

        assert len(np.unique(d.to_list())) == d.get_k()
        assert max(d.to_list_normalized()) == d.get_k()-1
        assert len(d.to_lists()) == d.get_k()


if __name__ == "__main__":
    test_DisjointSets()
    test_GiniDisjointSets()
