import warnings
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import sklearn.metrics


from genieclust.compare_partitions import *

# compare our C++ implementation (that we can make available in R, Julia etc.!)
# with the sklearn one (written in Python)

def compare_with_sklearn(x, y):
    ari1 = adjusted_rand_score(x, y)
    ari2 = sklearn.metrics.adjusted_rand_score(x, y)
    assert abs(ari1-ari2)<1e-9

    fm1 = fm_score(x, y)
    fm2 = sklearn.metrics.fowlkes_mallows_score(x, y)
    assert abs(fm1-fm2)<1e-9

    mi1 = mi_score(x, y)
    mi2 = sklearn.metrics.mutual_info_score(x, y)
    assert abs(mi1-mi2)<1e-9

    nmi1 = normalized_mi_score(x, y)
    nmi2 = sklearn.metrics.normalized_mutual_info_score(x, y)
    assert abs(nmi1-nmi2)<1e-9

    ami1 = adjusted_mi_score(x, y)
    ami2 = sklearn.metrics.adjusted_mutual_info_score(x, y)
    assert abs(ami1-ami2)<1e-9


# np.random.seed(123)
def test_compare_partitions():
    for n in [15, 25, 50, 100, 250, 500, 1000]:
        for k in range(2, 11):

            x = np.random.permutation(np.r_[np.arange(k), np.random.choice(np.arange(k), n-k)])
            y = np.random.permutation(np.r_[np.arange(k), np.random.choice(np.arange(k), n-k)])
            compare_with_sklearn(x, y)
            assert -1e-9<normalized_pivoted_accuracy(x, y)<1.0+1e-9
            assert -1e-9<normalized_clustering_accuracy(x, y)<1.0+1e-9
            assert -1e-9<pair_sets_index(x, y)<1.0+1e-9
            assert -1e-9<pair_sets_index(x, y, True)<1.0+1e-9

            assert normalized_clustering_accuracy(x, y) == normalized_clustering_accuracy(confusion_matrix(x, y))

            y = x.copy()
            y[:5] = 1
            compare_with_sklearn(x, y)
            assert -1e-9<normalized_pivoted_accuracy(x, y)<1.0+1e-9
            assert -1e-9<normalized_clustering_accuracy(x, y)<1.0+1e-9
            assert -1e-9<pair_sets_index(x, y)<1.0+1e-9
            assert -1e-9<pair_sets_index(x, y, True)<1.0+1e-9

            y = x.copy()
            y[::2] = 1
            compare_with_sklearn(x, y)
            assert -1e-9<normalized_pivoted_accuracy(x, y)<1.0+1e-9
            assert -1e-9<normalized_clustering_accuracy(x, y)<1.0+1e-9
            assert -1e-9<pair_sets_index(x, y)<1.0+1e-9
            assert -1e-9<pair_sets_index(x, y, True)<1.0+1e-9

            y = x.copy()
            c = np.random.permutation(np.arange(k))
            for i in range(n): y[i] = c[x[i]]

            assert 1.0+1e-9>adjusted_rand_score(x, y)>1.0-1e-9
            assert 1.0+1e-9>rand_score(x, y)>1.0-1e-9
            assert 1.0+1e-9>adjusted_fm_score(x, y)>1.0-1e-9
            assert 1.0+1e-9>fm_score(x, y)>1.0-1e-9
            assert          mi_score(x, y)>-1e-9
            assert 1.0+1e-9>normalized_mi_score(x, y)>1.0-1e-9
            assert 1.0+1e-9>adjusted_mi_score(x, y)>1.0-1e-9
            assert 1.0+1e-9>normalized_pivoted_accuracy(x, y)>1.0-1e-9
            assert 1.0+1e-9>normalized_clustering_accuracy(x, y)>1.0-1e-9
            assert 1.0+1e-9>pair_sets_index(x, y)>1.0-1e-9
            assert 1.0+1e-9>pair_sets_index(x, y, True)>1.0-1e-9

            assert confusion_matrix(x, y).sum() == normalized_confusion_matrix(x, y).sum()

            # TODO: more tests...



if __name__ == "__main__":
    test_compare_partitions()
