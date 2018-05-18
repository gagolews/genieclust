"""
HDBSCAN* Clustering Algorithm

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


class HDBSCAN(BaseEstimator, ClusterMixin):
    """
    An implementation of the HDBSCAN* Clustering Algorithm,
    that yields a specific number of clusters, and hence
    is not dependent on the original DBSCAN's somehow magical
    parameter eps.

    @TODO@: The current implementation of the fit() method runs in O(n**2) and
    uses O(n**2) memory.


    See R. Campello, D. Moulavi, A. Zimek, J. Sander, Hierarchical density
    estimates for data clustering, visualization, and outlier detection,
    ACM Transactions on Knowledge Discovery from Data 10(1):5:1–5:51, 2015.
    doi:10.1145/2733381.

    Basically this is the single linkage algorithm that marks all
    leaves in the corresponding minimum spanning tree as noise points.
    The genieclust.internal.mutual_reachability_distance() function returns
    a pairwise distance matrix that enables to take the smoothing
    factor M into account.

    Note that for smoothing factor M == 1, you should use the ordinary
    single linkage algorithm, i.e., mark no points as noise.

    The authors of the original manuscript suggest some post-processing
    of the results, as in practice the number of noise points tends
    to be very large. For instance, "cluster boundary points"
    can be merged back with the nearest clusters.

    Another option is just to merge all noise points with their
    nearest clusters, see genieclust.internal.merge_leaves_with_nearest_clusters().
    This yields a classical n_clusters-partition of a data set (with no notion
    of noise).


    Parameters:
    ----------

    n_clusters : int, default=2
        Number of clusters the data is split into.

    M : int, default=4
        Smoothing factor.

    metric : str or function, default="euclidean"
        See scipy.spatial.distance.pdist()


    Attributes:
    --------

    labels_ : ndarray, shape (n_samples,)
        Cluster labels for each point in the dataset given to fit():
        an integer vector c with c[i] denoting the cluster id
        (in {0, ..., n_clusters-1}) of the i-th object.
        -1 denotes that a noise point.
    """

    def __init__(self,
                 n_clusters=2,
                 M=4,
                 metric="euclidean"):
        n_clusters = int(n_clusters)
        M = int(M)
        if n_clusters <= 1:
            raise Exception("n_clusters must be > 1")
        if M < 2:
            raise Exception("M must be > 1")

        self.n_clusters = n_clusters
        self.M = M
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
        (w.r.t. the mutual reachability distance)
        and return the predicted labels.

        This method ignores self.M and self.metric.


        The MST may, for example, be determined as follows:

        MST = genieclust.mst.MST_pair(
            genieclust.internal.mutual_reachability_distance(
                scipy.spatial.distance.squareform(
                    scipy.spatial.distance.pdist(X, metric)),
                M)
        )


        Parameters:
        ----------

        MST : tuple
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
            genieclust.internal.mutual_reachability_distance(
                scipy.spatial.distance.squareform(
                    scipy.spatial.distance.pdist(X, self.metric)),
                self.M)
        )
        self.fit_from_mst(MST)


    def fit_from_mst(self, MST):
        """
        Compute a k-partition based on a precomputed MST.

        This method ignores self.M and self.metric.


        The MST may, for example, be determined as follows:

        MST = genieclust.mst.MST_pair(
            mutual_reachability_distance(
                scipy.spatial.distance.squareform(
                    scipy.spatial.distance.pdist(X, metric)),
                M)
        )


        Parameters:
        ----------

        MST : tuple
            See genieclust.mst.MST_pair()
        """
        self.labels_ = internal.hdbscan_from_mst(MST,
            self.n_clusters)


    def __repr__(self):
        """
        Return repr(self).
        """
        return "HDBSCAN(n_clusters=%r, M=%r, metric=%r)" % (
            self.n_clusters,
            self.M,
            self.metric
        )
