import numpy as np
from genieclust.genie import *
from genieclust.inequity import*
from genieclust.mst import *
from genieclust.internal import GiniDisjointSets
import time
import gc

#np.random.seed(123)
def test_GiniDisjointSets():
    for n in [5, 10, 25, 100, 250, 1000, 10000]:
        d = GiniDisjointSets(n)
        assert all([i==d.find(i) for i in range(n)])

        for k in range(int(np.random.randint(0, n-2, 1))):
            i,j = np.random.randint(0, n, 2)
            if d.find(i) == d.find(j): continue
            d.union(i, j)
            g1 = d.get_gini()
            c1 = d.get_counts()
            #c1 = sorted(c1)
            #c2 = sorted([len(x) for x in d.to_lists()])
            #assert(c1 == c2)
            assert min(c1) == d.get_smallest_count()
            g2 = gini(np.array(c1), True)
            assert abs(g1-g2)<1e-9






import scipy.spatial.distance
from rpy2.robjects.packages import importr
stats = importr("stats")
genie = importr("genie")
import numpy as np
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
path = "benchmark_data"

def test_genie():
    for dataset in ["unbalance", "s1", "pathbased", "a1", "Aggregation", "WUT_Smile"]: #, "bigger"
        for g in [0.01, 0.1, 0.3, 0.5, 1.0]:
            gc.collect()
            if dataset == "bigger":
                np.random.seed(123)
                n = 25000
                X = np.random.normal(size=(n,2))
                labels = np.random.choice(np.r_[1,2], n)
            else:
                X = np.loadtxt("%s/%s.data.gz" % (path,dataset), ndmin=2)
                labels = np.loadtxt("%s/%s.labels0.gz" % (path,dataset), dtype='int')
            label_counts = np.unique(labels,return_counts=True)[1]
            k = len(label_counts)
            D = scipy.spatial.distance.pdist(X)
            D = scipy.spatial.distance.squareform(D)

            t0 = time.time()
            M = MST_pair(D)
            res1 = Genie(k, g).fit_predict_from_mst(M)+1
            print("%-20s g=%.2f\t t_py=%.3f" % (dataset, g, time.time()-t0), end="\t")

            D = stats.dist(X)
            t0 = time.time()
            res2 = stats.cutree(genie.hclust2(D, thresholdGini=g), k)
            print("t_r=%.3f" % (time.time()-t0))
            res2 = np.array(res2)
            assert np.all(res1 == res2)


if __name__ == "__main__":
    test_GiniDisjointSets()
    test_genie()

