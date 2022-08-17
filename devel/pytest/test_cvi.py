import numpy as np
import genieclust.cluster_validity

import scipy.spatial.distance
import numpy as np

# cluster validity measures are thoroughly tested
# in the corresponding R version's unit tests


import os
if os.path.exists("devel/benchmark_data"):
    path = "devel/benchmark_data"
elif os.path.exists("benchmark_data"):
    path = "benchmark_data"
else:
    path = "../benchmark_data"

def test_cvi():
    try:
        import rpy2
        from rpy2.robjects.packages import importr
        import rpy2.robjects.numpy2ri
        rpy2.robjects.numpy2ri.activate()

        r_base = importr("base")
        lib_loc = r_base.Sys_getenv("R_LIBS_USER")[0]
        print(lib_loc)

        r_genieclust = importr("genieclust", lib_loc=lib_loc)

        for dataset in ["s1", "Aggregation", "unbalance", "h2mg_64_50"]:
            X = np.loadtxt("%s/%s.data.gz" % (path,dataset), ndmin=2)
            labels = np.loadtxt("%s/%s.labels0.gz" % (path,dataset), dtype=np.intp)

            X = (X-X.mean(axis=0))/X.std(axis=None, ddof=1)
            X += np.random.normal(0, 0.0001, X.shape)

            i1 = r_genieclust.calinski_harabasz_index(X, labels)[0]
            i2 = genieclust.cluster_validity.calinski_harabasz_index(X, labels-1)
            assert i1 == i2


    except ImportError:
        pass


if __name__ == "__main__":
    test_cvi()
