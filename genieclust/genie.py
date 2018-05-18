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
    The Genie Clustering Algorithm

    Gagolewski M., Bartoszuk M., Cena A.,
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


    Parameters:
    ----------

    n_clusters : int, default=2
        Number of clusters the data is split into.

    gini_threshold : float, default=0.3
        The threshold for the Genie correction

    metric : str or function, default="euclidean"
        See scipy.spatial.distance.pdist()


    Attributes:
    --------

    labels_ : ndarray, shape (n_samples,)
        Cluster labels for each point in the dataset given to fit():
        an integer vector c with c[i] denoting the cluster id
        (in {0, ..., n_clusters-1}) of the i-th object.
    """

    def __init__(self,
                  n_clusters=2,
                  gini_threshold=0.3,
                  metric="euclidean"):

        n_clusters = int(n_clusters)
        gini_threshold = float(gini_threshold)
        if n_clusters <= 1:
            raise Exception("n_clusters must be > 1")
        if not (0.0 <= gini_threshold <= 1.0):
            raise Exception("gini_threshold not in [0,1]")

        self.n_clusters = n_clusters
        self.gini_threshold = gini_threshold
        self.metric = metric
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


    def fit_predict_from_mst(self, MST):
        """
        Compute a k-partition based on a precomputed MST
        and return the predicted labels.

        This method ignores self.metric.


        The MST may, for example, be determined as follows:

        MST = genieclust.mst.MST_pair(
            scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(X, "euclidean")),
        )


        Parameters:
        ----------

        mst : tuple
            See genieclust.mst.MST_pair()


        Returns:
        -------

        labels_ : ndarray, shape (n,)
            Predicted labels, representing a partition of X.
            labels_[i] gives the cluster id of the i-th input point.
        """
        self.fit_from_mst(MST)
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
        MST = mst.MST_pair(
            scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(X, self.metric)),
        )
        self.fit_from_mst(MST)


    def fit_from_mst(self, MST):
        """
        Compute a k-partition based on a precomputed MST.

        This method ignores self.metric.


        The MST may, for example, be determined as follows:

        MST = genieclust.mst.MST_pair(
            scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(X, "euclidean")),



        Parameters:
        ----------

        MST : tuple
            See genieclust.mst.MST_pair()
        """
        self.labels_ = internal.genie_from_mst(MST,
            self.n_clusters, self.gini_threshold)


    def __repr__(self):
        """
        Return repr(self).
        """
        return "Genie(n_clusters=%r, gini_threshold=%r, metric=%r)" % (
            self.n_clusters,
            self.gini_threshold,
            self.metric
        )
