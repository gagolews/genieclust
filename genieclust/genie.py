"""The Genie+ Clustering Algorithm

Copyright (C) 2018-2020 Marek Gagolewski (https://www.gagolewski.com)
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
from . import internal
from . import mst
import scipy.spatial.distance
from sklearn.base import BaseEstimator, ClusterMixin
import sklearn.neighbors
import warnings
import math

# @TODO: delme
import time

#
try:
    import faiss
except ImportError:
    pass



class Genie(BaseEstimator, ClusterMixin):
    """The Genie+ Clustering Algorithm with optional smoothing and
    noise point detection (for M>1)

    Based on: Gagolewski M., Bartoszuk M., Cena A.,
    Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
    Information Sciences 363, 2016, pp. 8-23. doi:10.1016/j.ins.2016.05.003

    A new hierarchical clustering linkage criterion: the Genie algorithm
    links two clusters in such a way that an inequity measure
    (namely, the Gini index) of the cluster sizes doesn't go far beyond
    some threshold. The introduced method most often outperforms
    the Ward or average linkage, k-means, spectral clustering,
    DBSCAN, Birch, and many others in terms of the clustering
    quality while - at the same time - it retains the speed of
    the single linkage algorithm.

    This is a new implementation of the Genie algorithm that requires
    O(n_samples*sqrt(n_samples))-time given a minimum spanning tree
    of the pairwise distance graph.
    The clustering can also be computed with respect to the
    mutual reachability distance (based, e.g., on the Euclidean metric),
    which is used in the definition of the HDBSCAN* algorithm, see
    R. Campello, D. Moulavi, A. Zimek, J. Sander, Hierarchical density
    estimates for data clustering, visualization, and outlier detection,
    ACM Transactions on Knowledge Discovery from Data 10(1):5:1–5:51, 2015.
    doi:10.1145/2733381.

    The Genie correction together with the smoothing factor M>2 (note that
    M==2 corresponds to the original distance) gives a robustified version of
    the HDBSCAN algorithm that is able to yield a predefined number of clusters,
    and hence  not dependent on the original DBSCAN's somehow magical
    `eps` parameter or the HDBSCAN Python package's `min_cluster_size` one.


    Parameters
    ----------

    n_clusters : int, default=2
        Number of clusters to detect.
    gini_threshold : float in [0,1], default=0.3
        The threshold for the Genie correction, i.e.,
        the Gini index of the cluster size distribution.
        Threshold of 1.0 disables the correction.
        Low thresholds highly penalise the formation of small clusters.
    M : int, default=1
        Smoothing factor. M=1 gives the original Genie algorithm.
    n_neighbors : int, default=-1
        Number of nearest neighbours to compute for each data point.
        n_neighbors < 0 picks the default one, typically several dozen,
        but no less than M. Note that the algorithm's memory
        consumption is proportional to n_samples*n_neighbors.
    postprocess : str, one of "boundary" [default], "none", "all"
        In effect only if M>1. By default, only "boundary" points are merged
        with their nearest "core" points. To force a classical
        n_clusters-partition of a data set (with no notion of noise),
        choose "all".
    exact : bool, default=False
        If False, the minimum spanning tree shall be approximated
        based on the nearest neighbours graph. Finding nearest neighbours
        in low dimensional spaces is usually fast. Otherwise,
        the algorithm will need to inspect all pairwise distances,
        which gives the time complexity of O(n_samples*n_samples*n_features).
    allow_cast_float32 : bool, default=True
        Allow casting input data to float32 (for efficiency reasons, however,
        increases total memory usage). Note that some nearest neighbour search
        methods require float32 data anyway.  This also normalises
        the input coordinates so that the method is guaranteed to be translation
        and scale invariant.
    nn_params: dict, optional (default=None)
        Arguments to the sklearn.neighbors.NearestNeighbors class
        constructor, e.g., the metric to use (default='euclidean').


    Attributes
    ----------

    labels_ : ndarray, shape (n_samples,)
        Detected cluster labels for each point in the dataset given to fit():
        an integer vector c with c[i] denoting the cluster id
        (in {0, ..., n_clusters-1}) of the i-th object.
        If M>1, noise points are labelled -1.
    """

    def __init__(self,
            n_clusters=2,
            gini_threshold=0.3,
            M=1,
            n_neighbors=-1,
            postprocess="boundary",
            exact=True,
            allow_cast_float32=True,
            nn_params=None
        ):
        self.n_clusters = n_clusters
        self.gini_threshold = gini_threshold
        self.M = M
        self.n_neighbors = n_neighbors
        self.postprocess = postprocess
        self.exact = exact
        self.allow_cast_float32 = allow_cast_float32
        self.nn_params = nn_params

        self.labels_ = None
        # self.__last_state = dict()
        # self.__last_X
        # self.__last_mst
        # self.__last_nn_dist
        # self.__last_nn_ind


    def fit(self, X, y=None, cache=False):
        """Perform clustering on X.
        The resulting partition shall be given by self.labels_.


        Parameters
        ----------

        X : ndarray, shape (n_samples, n_features)
            A matrix defining n_samples points in
            a n_features-dimensional vector space.
        y : None
            Ignored.
        cache : bool, default=True
            Store auxiliary results to speed up further calls
            to fit() on the same data matrix, but with different params.


        Returns
        -------

        self
        """
        n = X.shape[0]
        #d = X.shape[0]

        if cache:
            raise NotImplementedError("cache not implemented yet")


        cur_state = dict()

        if self.nn_params is None:
            cur_state["nn_params"] = dict()
        else:
            cur_state["nn_params"] = self.nn_params

        cur_state["metric"] = cur_state["nn_params"].get("metric", "euclidean")
        cur_state["metric_params"] = cur_state["nn_params"].get("metric_params", None)
        if cur_state["metric_params"] is None:
            cur_state["metric_params"] = dict()

        cur_state["n_clusters"] = int(self.n_clusters)
        if cur_state["n_clusters"] <= 1:
            raise ValueError("n_clusters must be > 1")

        cur_state["gini_threshold"] = float(self.gini_threshold)
        if not (0.0 <= cur_state["gini_threshold"] <= 1.0):
            raise ValueError("gini_threshold not in [0,1]")

        cur_state["M"] = int(self.M)
        if not 1 <= cur_state["M"] <= n:
            raise ValueError("M must be in [1, n_samples]")

        cur_state["postprocess"] = self.postprocess
        if cur_state["postprocess"] not in ("boundary", "none", "all"):
            raise ValueError('postprocess should be one of ("boundary", "none", "all")')

        cur_state["n_neighbors"] = int(self.n_neighbors)
        if 0 <= cur_state["n_neighbors"] < max(1, cur_state["M"]-1):
            raise ValueError("n_neighbors should be >= M-1")

        cur_state["exact"] = bool(self.exact)
        cur_state["allow_cast_float32"] = bool(self.allow_cast_float32)

        if cur_state["allow_cast_float32"]:
            X = X.astype(np.float32, order="C", copy=False) # faiss supports float32 only # warning if sparse!!
            # center X + scale (NOT: standardize!)
            X = (X-X.mean(axis=0))/X.std(axis=None, ddof=1) # we don't want this for sparse X

        nn_dist       = None
        nn_ind        = None
        spanning_tree = None
        if not cur_state["exact"]:
            #raise NotImplementedError("approximate method not implemented yet")

            actual_n_neighbors = cur_state["n_neighbors"]
            if actual_n_neighbors < 0:
                actual_n_neighbors = min(32, int(math.ceil(math.sqrt(n))))
                actual_n_neighbors = max(actual_n_neighbors, cur_state["M"]-1)
                actual_n_neighbors = min(n-1, actual_n_neighbors)

            # t0 = time.time()
            #nn = sklearn.neighbors.NearestNeighbors(n_neighbors=actual_n_neighbors, **cur_state["nn_params"])
            #nn_dist, nn_ind = nn.fit(X).kneighbors()
            # print("T=%.3f" % (time.time()-t0), end="\t")

            # FAISS - `euclidean` and `cosine` only!




            nn = faiss.IndexFlatL2(X.shape[1])
            nn.add(X)
            nn_dist, nn_ind = nn.search(X, actual_n_neighbors+1)

            # @TODO:::::
            #nn_bad_where = np.where((nn_ind[:,0]!=np.arange(n)))[0]
            #print(nn_bad_where)
            #print(nn_ind[nn_bad_where,:5])
            #print(X[nn_bad_where,:])
            #assert nn_bad_where.shape[0] == 0

            nn_dist = nn_dist[:,1:].astype(X.dtype, order="C")
            nn_ind  = nn_ind[:,1:].astype(np.intp, order="C")

            if cur_state["M"] > 1:
                # d_core = nn_dist[:,cur_state["M"]-2].astype(X.dtype, order="C")
                raise NotImplementedError("approximate method not implemented yet")

            # t0 = time.time()
            mst_dist, mst_ind = mst.mst_from_nn(nn_dist, nn_ind, stop_disconnected=True)
            # print("T=%.3f" % (time.time()-t0), end="\t")

        else: # cur_state["exact"]
            if cur_state["M"] == 1:
                # the original Genie algorithm

                # 1. Use Prim's algorithm to determine the MST
                #  w.r.t. the distances computed on the fly

                # t0 = time.time()

                mst_dist, mst_ind = mst.mst_from_distance(X,
                    metric=cur_state["metric"],
                    metric_params=cur_state["metric_params"])
                # print("T=%.3f" % (time.time()-t0), end="\t")
            else:
                # Genie+HDBSCAN

                # 1. Use sklearn to determine d_core distance
                nn = sklearn.neighbors.NearestNeighbors(
                    n_neighbors=cur_state["M"]-1, **cur_state["nn_params"])
                nn_dist, nn_ind = nn.fit(X).kneighbors()
                d_core = nn_dist[:,cur_state["M"]-2].astype(X.dtype, order="C")

                # 2. Use Prim's algorithm to determine the MST
                #  w.r.t. the distances computed on the fly
                mst_dist, mst_ind = mst.mst_from_distance(X,
                    metric=cur_state["metric"],
                    metric_params=dict(**cur_state["metric_params"],
                    d_core=d_core)
                )

        # apply the Genie+ algorithm
        labels = internal.genie_from_mst(mst_dist, mst_ind,
            n_clusters=cur_state["n_clusters"],
            gini_threshold=cur_state["gini_threshold"],
            noise_leaves=(cur_state["M"]>1))

        # postprocess labels, if requested to do so
        if cur_state["M"] == 1 or cur_state["postprocess"] == "none":
            pass
        elif cur_state["postprocess"] == "boundary":
            labels = postprocess.merge_boundary_points(mst_ind, labels, nn_ind, cur_state["M"])
        elif cur_state["postprocess"] == "all":
            labels = postprocess.merge_noise_points(mst_ind, labels)

        self.labels_ = labels

        # # save state
        # self.__last_state    = cur_state

        # self.__last_X        = X
        # self.__last_mst      = mst
        # self.__last_nn_dist  = nn_dist
        # self.__last_nn_ind   = nn_ind

        self._mst_dist = mst_dist
        self._mst_ind  = mst_ind

        return self


    # not needed - inherited from ClusterMixin
    def fit_predict(self, X, y=None, cache=False):
        """Compute a k-partition and return the predicted labels,
        see fit().


        Parameters
        ----------

        X : ndarray
            see fit()
        y : None
            see fit()
        cache : bool
            see fit()



        Returns
        -------

        labels_ : ndarray, shape (n_samples,)
            Predicted labels, representing a partition of X.
            labels_[i] gives the cluster id of the i-th input point.
            negative labels_ correspond to noise points.
        """
        self.fit(X)
        return self.labels_


    # not needed - inherited from BaseEstimator
    # def __repr__(self):
    #     """
    #     Return repr(self).
    #     """
    #     return "Genie(%s)" % (
    #         ", ".join(["%s=%r"%(k,v) for (k,v) in self.get_params().items()])
    #     )

    #
    # def get_params(self, deep=False):
    #     """
    #     Get the parameters for this estimator.
    #
    #     Parameters:
    #     -----------
    #
    #     deep: bool
    #         Ignored
    #
    #     Returns:
    #     --------
    #
    #     params: dict
    #     """
    #     return dict(
    #         n_clusters = self.__n_clusters,
    #         gini_threshold = self.__gini_threshold,
    #         M = self.__M,
    #         postprocess = self.__postprocess,
    #         n_neighbors = self.__n_neighbors,
    #         **self.__NearestNeighbors_params
    #     )

    # not needed - inherited from BaseEstimator
    # def set_params(self, **params):
    #     """
    #     Set the parameters for this estimator.
    #
    #
    #     Parameters:
    #     -----------
    #
    #     params
    #
    #
    #     Returns:
    #     --------
    #
    #     self
    #     """
    #     ################## @TODO
    #     return self
