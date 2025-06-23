import numpy as np
import genieclust
import time
import gc

import scipy.spatial.distance
import numpy as np

import matplotlib.pyplot as plt
import pytest


def test_plot():
    np.random.seed(123)

    n = 100
    X = np.random.rand(n, 2)
    genieclust.plots.plot_scatter(X)
    genieclust.plots.plot_scatter(X[:,0], X[:,1])
    genieclust.plots.plot_scatter(X, labels=np.random.choice(np.arange(10), n))
    mst_d, mst_i = genieclust.fastmst.mst_euclid(X)
    genieclust.plots.plot_segments(mst_i, X)
    genieclust.plots.plot_segments(mst_i, X[:,0], X[:,1])

    with pytest.raises(Exception):
        genieclust.plots.plot_scatter(X.reshape(50,2,2))

    with pytest.raises(Exception):
        genieclust.plots.plot_scatter(X.reshape(50,4))

    with pytest.raises(Exception):
        genieclust.plots.plot_scatter(X, labels=np.r_[1,2])

    with pytest.raises(Exception):
        genieclust.plots.plot_scatter(X[:,1])

    with pytest.raises(Exception):
        genieclust.plots.plot_scatter(X[:,1], X)

    with pytest.raises(Exception):
        genieclust.plots.plot_scatter(X, X[:,1])

    with pytest.raises(Exception):
        genieclust.plots.plot_scatter(X[:,0], X[5:,1])

    with pytest.raises(Exception):
        genieclust.plots.plot_segments(mst_d, X)


if __name__ == "__main__":
    test_plot()
