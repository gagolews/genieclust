import numpy as np
import genieclust.cluster_validity

import scipy.spatial.distance
import numpy as np

# cluster validity measures are thoroughly tested
# in the corresponding R version's unit tests


import os
if os.path.exists(".devel/benchmark_data"):
    path = ".devel/benchmark_data"
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

    except:
        print("ImportError")
        return

    for dataset in ["s1", "Aggregation", "unbalance", "h2mg_64_50"]:
        X = np.loadtxt("%s/%s.data.gz" % (path,dataset), ndmin=2)
        labels = np.loadtxt("%s/%s.labels0.gz" % (path,dataset), dtype=np.intp)

        X = (X-X.mean(axis=0))/X.std(axis=None, ddof=1)
        X += np.random.normal(0, 0.0001, X.shape)

        i1 = r_genieclust.calinski_harabasz_index(X, labels)[0]
        i2 = genieclust.cluster_validity.calinski_harabasz_index(X, labels-1)
        assert i1 == i2

        i1 = r_genieclust.negated_ball_hall_index(X, labels)[0]
        i2 = genieclust.cluster_validity.negated_ball_hall_index(X, labels-1)
        assert i1 == i2

        i1 = r_genieclust.negated_davies_bouldin_index(X, labels)[0]
        i2 = genieclust.cluster_validity.negated_davies_bouldin_index(X, labels-1)
        assert i1 == i2

        i1 = r_genieclust.negated_wcss_index(X, labels)[0]
        i2 = genieclust.cluster_validity.negated_wcss_index(X, labels-1)
        assert i1 == i2

        i1 = r_genieclust.silhouette_index(X, labels)[0]
        i2 = genieclust.cluster_validity.silhouette_index(X, labels-1)
        assert i1 == i2

        i1 = r_genieclust.silhouette_w_index(X, labels)[0]
        i2 = genieclust.cluster_validity.silhouette_w_index(X, labels-1)
        assert i1 == i2

        i1 = r_genieclust.generalised_dunn_index(X, labels, 4, 2)[0]
        i2 = genieclust.cluster_validity.generalised_dunn_index(X, labels-1, 4, 2)
        assert i1 == i2

        i1 = r_genieclust.wcnn_index(X, labels, 5)[0]
        i2 = genieclust.cluster_validity.wcnn_index(X, labels-1, 5)
        assert i1 == i2

        i1 = r_genieclust.dunnowa_index(X, labels, 10, "SMin:5", "Max")[0]
        i2 = genieclust.cluster_validity.dunnowa_index(X, labels-1, 10, "SMin:5", "Max")
        assert i1 == i2

        #double c_dunnowa_index

        #double c_generalised_dunn_index

        #double c_wcnn_index



if __name__ == "__main__":
    test_cvi()
