"""
The Genie Clustering Algorithm

Copyright (C) 2018 Marek.Gagolewski.com
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import scipy.spatial.distance
import warnings
from sklearn.base import BaseEstimator, ClusterMixin

from . import internal
from . import mst

class Genie(BaseEstimator, ClusterMixin):
    """
    The Genie+ Clustering Algorithm with noise point detection (for M>1)

    Based on: Gagolewski M., Bartoszuk M., Cena A.,
    Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
    Information Sciences 363, 2016, pp. 8-23. doi:10.1016/j.ins.2016.05.003

    A new hierarchical clustering linkage criterion: the Genie algorithm
    links two clusters in such a way that a chosen economic inequity measure
    (here, the Gini index) of the cluster sizes does not increase drastically
    above a given threshold. Benchmarks indicate a high practical
    usefulness of the introduced method: it most often outperforms
    the Ward or average linkage, k-means, spectral clustering,
    DBSCAN, Birch, and others in terms of the clustering
    quality while retaining the single linkage speed.

    This is a new implementation of an O(n sqrt(n))-time algorithm
    (provided that the MST /minimum spanning tree of the
    pairwise distance graph/ has already been computed,
    see genieclust.internal.genie_from_mst()).

    @TODO@: The current implementation of the fit() method runs in O(n**2) and
    uses O(n**2) memory.

    The clustering can also be computed with respect to the
    mutual reachability distance (based, e.g., on the Euclidean metric),
    which is used in the definition of the HDBSCAN* algorithm, see
    R. Campello, D. Moulavi, A. Zimek, J. Sander, Hierarchical density
    estimates for data clustering, visualization, and outlier detection,
    ACM Transactions on Knowledge Discovery from Data 10(1):5:1–5:51, 2015.
    doi:10.1145/2733381.

    The Genie correction together with the smoothing factor M>2 (note that
    M==2 corresponds to the original distance) gives a robustified version of
    the HDBSCAN algorithm that yields a specific number of clusters, and hence
    is not dependent on the original DBSCAN's somehow magical
    parameter eps or the HDBSCAN Python package's min_cluster_size one.

    If M>1, some points may be marked as noise. If some postprocessing
    is required, you may wish to merge all noise points with their
    nearest clusters, see genieclust.internal.merge_leaves_with_nearest_clusters().
    This will yield a classical n_clusters-partition of a data set
    (with no notion of noise).


    Parameters:
    ----------

    n_clusters : int, default=2
        Number of clusters the data is split into.

    gini_threshold : float, default=0.3
        The threshold for the Genie correction.

    M : int, default=1
        Smoothing factor. M=1 gives the original Genie algorithm,
        which mark no points as noise.

    metric : str or function, default="euclidean"
        See scipy.spatial.distance.pdist()


    Attributes:
    --------

    labels_ : ndarray, shape (n_samples,)
        Cluster labels for each point in the dataset given to fit():
        an integer vector c with c[i] denoting the cluster id
        (in {0, ..., n_clusters-1}) of the i-th object.
        If M>1, noise points are labeled -1.
    """

    def __init__(self,
                  n_clusters=2,
                  gini_threshold=0.3,
                  M=1,
                  metric="euclidean"):

        n_clusters = int(n_clusters)
        gini_threshold = float(gini_threshold)
        M = int(M)
        if n_clusters <= 1:
            raise Exception("n_clusters must be > 1")
        if not (0.0 <= gini_threshold <= 1.0):
            raise Exception("gini_threshold not in [0,1]")
        if M < 1:
            raise Exception("M must be >= 1")

        self.n_clusters = n_clusters
        self.gini_threshold = gini_threshold
        self.metric = metric
        self.M = M
        self.labels_ = None


    def fit_predict(self, X, y=None):
        """
        Compute a k-partition and return the predicted labels.

        @TODO@: do not compute the whole distance matrix.
        The current version requires O(n**2) memory.


        Parameters:
        ----------

        X : ndarray, shape (n,d)
            A matrix defining n points in a d-dimensional vector space.

        y : None
            Ignored.


        Returns:
        -------

        labels_ : ndarray, shape (n,)
            Predicted labels, representing a partition of X.
            labels_[i] gives the cluster id of the i-th input point.
        """
        self.fit(X)
        return self.labels_


    def fit(self, X, y=None):
        """
        Compute a k-partition.

        @TODO@: do not compute the whole distance matrix.
        The current version requires O(n**2) memory.


        Parameters:
        ----------

        X : ndarray, shape (n,d)
            A matrix defining n points in a d-dimensional vector space.

        y : None
            Ignored.
        """
        D = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(X, self.metric)
        )

        if self.M > 2:
            D = internal.mutual_reachability_distance(D, self.M)

        MST = mst.MST_pair(D)
        self.labels_ = internal.genie_from_mst(MST,
            self.n_clusters, self.gini_threshold, self.M>1)


    def __repr__(self):
        """
        Return repr(self).
        """
        return "Genie(n_clusters=%r, gini_threshold=%r, M=%r, metric=%r)" % (
            self.n_clusters,
            self.gini_threshold,
            self.M,
            self.metric
        )
