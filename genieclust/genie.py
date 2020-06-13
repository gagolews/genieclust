"""The Genie++ Clustering Algorithm

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

import math
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from . import internal
# import scipy.spatial.distance
# import sklearn.neighbors
# import warnings


try:
    import faiss
except ImportError:
    faiss = None


try:
    import mlpack
except ImportError:
    mlpack = None


class GenieBase(BaseEstimator, ClusterMixin):
    """Base class for Genie and GIc"""

    def __init__(self,
            M,
            affinity,
            exact,
            cast_float32,
            use_mlpack
        ):
        super().__init__()
        self.M = M
        self.affinity = affinity
        self.cast_float32 = cast_float32
        self.exact = exact
        self.use_mlpack = use_mlpack

        self.n_samples_   = None
        self.n_features_  = None
        self._mst_dist_   = None
        self._mst_ind_    = None
        self._nn_dist_    = None
        self._nn_ind_     = None
        self._d_core_     = None
        self._last_state_ = None


    def _postprocess(self, M, postprocess):
        """(internal)
        updates self.labels_ and self.is_noise_
        """
        reshaped = False
        if self.labels_.ndim == 1:
            reshaped = True
            # promote it to a matrix with 1 row
            self.labels_.shape = (1, self.labels_.shape[0])
            start_partition = 0
        else:
            # duplicate the 1st row (create the "0"-partition that will
            # not be postprocessed):
            self.labels_ = np.vstack((self.labels_[0,:], self.labels_))
            start_partition = 1 # do not postprocess the "0"-partition

        self.is_noise_    = (self.labels_[0,:] < 0)

        # postprocess labels, if requested to do so
        if M == 1 or postprocess == "none":
            pass
        elif postprocess == "boundary":
            for i in range(start_partition, self.labels_.shape[0]):
                self.labels_[i,:] = internal.merge_boundary_points(
                    self._mst_ind_, self.labels_[i,:],
                    self._nn_ind_, M)
        elif postprocess == "all":
            for i in range(start_partition, self.labels_.shape[0]):
                self.labels_[i,:] = internal.merge_noise_points(
                    self._mst_ind_, self.labels_[i,:])

        if reshaped:
            self.labels_.shape = (self.labels_.shape[1],)



    def fit(self, X, y=None):
        cur_state = dict()
        cur_state["X"] = id(X)

        _affinity_options = ("euclidean", "l2", "manhattan", "l1",
                             "cityblock", "cosine", "precomputed")
        cur_state["affinity"] = str(self.affinity).lower()
        if cur_state["affinity"] not in _affinity_options:
            raise ValueError("affinity should be one of %r"%_affinity_options)

        if cur_state["affinity"] == "l2":
            cur_state["affinity"] = "euclidean"
        if cur_state["affinity"] in ["l1", "cityblock"]:
            cur_state["affinity"] = "manhattan"

        n_samples  = X.shape[0]
        if cur_state["affinity"] == "precomputed":
            n_features = self.n_features_ # the user must set it manually
            X = X.reshape(X.shape[0], -1)
            if X.shape[1] not in [1, X.shape[0]]:
                raise ValueError("X must be distance vector \
                    or a square-form distance matrix, \
                    see scipy.spatial.distance.pdist or \
                    scipy.spatial.distance.squareform")
            if X.shape[1] == 1:
                # from a quadratic equation:
                n_samples = int(round((math.sqrt(1.0+8.0*n_samples)+1.0)/2.0))
                assert n_samples*(n_samples-1)//2 == X.shape[0]

        else:
            n_features = X.shape[1]

        cur_state["M"] = int(self.M)
        if not 1 <= cur_state["M"] <= n_samples:
            raise ValueError("M must be in [1, n_samples]")

        cur_state["exact"] = bool(self.exact)
        cur_state["cast_float32"] = bool(self.cast_float32)

        if self.use_mlpack == "auto":
            if mlpack is not None and \
                    cur_state["affinity"] == "euclidean" and \
                    n_features <= 6 and \
                    cur_state["M"] <= 2:
                cur_state["use_mlpack"] = True
            else:
                cur_state["use_mlpack"] = False

        else:
            cur_state["use_mlpack"] = bool(self.use_mlpack)

        if cur_state["use_mlpack"] and mlpack is None:
            raise ValueError("package mlpack is not available")
        if cur_state["use_mlpack"] and cur_state["affinity"] != "euclidean":
            raise ValueError("mlpack can only be used with affinity=='euclidean'")
        if cur_state["use_mlpack"] and cur_state["M"] not in [1, 2]:
            raise ValueError("mlpack can only be used with M of 1 or 2")

        mst_dist = None
        mst_ind  = None
        nn_dist  = None
        nn_ind   = None
        d_core   = None

        if cur_state["cast_float32"] and cur_state["affinity"] != "precomputed":
            # faiss supports float32 only
            # warning if sparse!!
            X = X.astype(np.float32, order="C", copy=False)


        if  self._last_state_ is not None and \
                cur_state["X"]            == self._last_state_["X"] and \
                cur_state["affinity"]     == self._last_state_["affinity"] and \
                cur_state["exact"]        == self._last_state_["exact"] and \
                cur_state["cast_float32"] == self._last_state_["cast_float32"]:

            if cur_state["M"] == self._last_state_["M"]:
                mst_dist = self._mst_dist_
                mst_ind  = self._mst_ind_
                nn_dist  = self._nn_dist_
                nn_ind   = self._nn_ind_
            elif cur_state["M"] < self._last_state_["M"]:
                nn_dist  = self._nn_dist_
                nn_ind   = self._nn_ind_
            else:
                pass

        if not cur_state["exact"]:
            #raise NotImplementedError("approximate method not implemented yet")

            if cur_state["affinity"] == "precomputed":
                raise ValueError('exact==False with affinity=="precomputed"')


            assert cur_state["affinity"] == "euclidean"

            actual_n_neighbors = min(32, int(math.ceil(math.sqrt(n_samples))))
            actual_n_neighbors = max(actual_n_neighbors, cur_state["M"]-1)
            actual_n_neighbors = min(n_samples-1, actual_n_neighbors)

            # t0 = time.time()
            ##nn = sklearn.neighbors.NearestNeighbors(
            ##n_neighbors=actual_n_neighbors, ....**cur_state["nn_params"])
            ##nn_dist, nn_ind = nn.fit(X).kneighbors()
            #nn_dist, nn_ind = internal.knn_from_distance(
            #X, k=actual_n_neighbors, ...metric=metric)
            # print("T=%.3f" % (time.time()-t0), end="\t")

            # FAISS - `euclidean` and `cosine` only!



            # TODO:  cur_state["metric"], cur_state["metric_params"]
            #t0 = time.time()
            # the slow part:
            nn = faiss.IndexFlatL2(n_features)
            nn.add(X)
            nn_dist, nn_ind = nn.search(X, actual_n_neighbors+1)
            #print("T=%.3f" % (time.time()-t0), end="\t")



            # @TODO:::::
            #nn_bad_where = np.where((nn_ind[:,0]!=np.arange(n_samples)))[0]
            #print(nn_bad_where)
            #print(nn_ind[nn_bad_where,:5])
            #print(X[nn_bad_where,:])
            #assert nn_bad_where.shape[0] == 0

            # TODO: check cache if rebuild needed
            nn_dist = nn_dist[:,1:].astype(X.dtype, order="C")
            nn_ind  = nn_ind[:,1:].astype(np.intp, order="C")

            if cur_state["M"] > 1:
                # d_core = nn_dist[:,cur_state["M"]-2].astype(X.dtype, order="C")
                raise NotImplementedError("approximate method not implemented yet")

            #t0 = time.time()
            # the fast part:
            mst_dist, mst_ind = internal.mst_from_nn(nn_dist, nn_ind,
                stop_disconnected=False, # TODO: test this!!!!
                stop_inexact=False)
            #print("T=%.3f" % (time.time()-t0), end="\t")

        else: # cur_state["exact"]
            if cur_state["use_mlpack"]:
                assert cur_state["M"] in [1, 2]
                assert cur_state["affinity"] == "euclidean"

                if mst_dist is None or mst_ind is None:
                    _res = mlpack.emst(input=X)["output"]
                    mst_dist = _res[:,2].astype(np.double, order="C")
                    mst_ind  = _res[:,:2].astype(np.intp, order="C")
            else:
                if cur_state["M"] > 2: # else d_core   = None
                    # Genie+HDBSCAN --- determine d_core
                    # TODO: mlpack for k-nns?
                    if nn_dist is None or nn_ind is None:
                        nn_dist, nn_ind = internal.knn_from_distance(
                            X, k=cur_state["M"]-1,
                            metric=cur_state["affinity"]) # supports "precomputed"

                    assert cur_state["M"]-2 < nn_dist.shape[1]
                    d_core = nn_dist[:,cur_state["M"]-2].astype(X.dtype, order="C")

                # Use Prim's algorithm to determine the MST
                # w.r.t. the distances computed on the fly
                if mst_dist is None or mst_ind is None:
                    mst_dist, mst_ind = internal.mst_from_distance(X,
                        metric=cur_state["affinity"],
                        d_core=d_core
                    )

        self.n_samples_  = n_samples
        self.n_features_ = n_features
        self._mst_dist_  = mst_dist
        self._mst_ind_   = mst_ind
        self._nn_dist_   = nn_dist
        self._nn_ind_    = nn_ind
        self._d_core_    = d_core
        self._last_state_= cur_state

        return self


    def fit_predict(self, X, y=None):
        """Compute a k-partition and return the predicted labels,
        see fit().


        Parameters
        ----------

        X : ndarray
            see fit()
        y : None
            see fit()


        Returns
        -------

        labels_ : ndarray, shape (n_samples,)
            Predicted labels, representing a partition of X.
            labels_[i] gives the cluster id of the i-th input point.
            negative labels_ correspond to noise points.
            Note that the determined number of clusters
            might be larger than the requested one.
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
    #def set_params(self, **params):
        #"""
        #Set the parameters for this estimator.


        #Parameters:
        #-----------

        #params


        #Returns:
        #--------

        #self
        #"""
        ################### @TODO
        #print(params)
        #super().set_params(**params)
        #return self








class Genie(GenieBase):
    """The Genie++ Clustering Algorithm with optional smoothing and
    noise point detection (for M>1)

    The Genie algorithm [1]
    links two clusters in such a way that an inequity measure
    (namely, the Gini index) of the cluster sizes doesn't go far beyond
    some threshold. The introduced method most often outperforms
    the Ward or average linkage, k-means, spectral clustering,
    DBSCAN, Birch, and many others in terms of the clustering
    quality while - at the same time - it retains the speed of
    the single linkage algorithm.

    This is a reimplementation (with extras) of the original Genie
    algorithm as implemented in the R package `genie` that requires
    O(n_samples*sqrt(n_samples))-time given a minimum spanning tree
    of the pairwise distance graph.

    The clustering can also be computed with respect to the
    mutual reachability distance (based, e.g., on the Euclidean metric),
    which is used in the definition of the HDBSCAN* algorithm, see [2].

    The Genie correction together with the smoothing factor M>1 (note that
    M==2 corresponds to the original distance) gives a robustified version of
    the HDBSCAN* algorithm that is able to yield a predefined number of
    clusters. Hence it does not dependent on the DBSCAN's somehow magical
    `eps` parameter or the HDBSCAN Python package's `min_cluster_size` one.

    Note according to the algorithm's original definition,
    the resulting partition tree (dendrogram) might violate
    the ultrametricity property (merges might occur at levels that
    are not increasing w.r.t. a between-cluster distance).
    Departures from ultrametricity are corrected by applying
    `Z[:,2] = genieclust.tools.cummin(Z[::-1,2])[::-1]`.


    References
    ==========

    [1] Gagolewski M., Bartoszuk M., Cena A.,
    Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
    Information Sciences 363, 2016, pp. 8-23. doi:10.1016/j.ins.2016.05.003

    [2] Campello R., Moulavi D., Zimek A., Sander J.,
    Hierarchical density estimates for data clustering, visualization,
    and outlier detection,
    ACM Transactions on Knowledge Discovery from Data 10(1), 2015, 5:1–5:51.
    doi:10.1145/2733381.


    Parameters
    ----------

    n_clusters : int >= 0, default=2
        Number of clusters to detect. Note that depending on the dataset
        and approximations used (see parameter `exact`), the actual
        partition cardinality can be smaller.
        n_clusters==1 can act as a noise point/outlier detector (if M>1
        and postprocess is not "all").
        n_clusters==0 computes the whole dendrogram but doesn't generate
        any particular cuts.
    gini_threshold : float in [0,1], default=0.3
        The threshold for the Genie correction, i.e.,
        the Gini index of the cluster size distribution.
        Threshold of 1.0 disables the correction.
        Low thresholds highly penalise the formation of small clusters.
    M : int, default=1
        Smoothing factor. M=1 gives the original Genie algorithm.
    affinity : str, default="euclidean"
        Metric used to compute the linkage. One of: "euclidean" (synonym: "l2"),
        "manhattan" (a.k.a. "l1" and "cityblock"), "cosine" or "precomputed".
        If "precomputed", a n_samples*(n_samples-1)/2 distance vector
        or a square-form distance
        matrix is needed on input (argument X) for the fit() method,
        see `scipy.spatial.distance.pdist()` or
        `scipy.spatial.distance.squareform()`, amongst others.
    compute_full_tree : bool, default=True
        If True, only a partial hierarchy is determined so that
        at most n_clusters are generated. Saves some time if you think you know
        how many clusters are there, but are you *really* sure about that?
    compute_all_cuts : bool, default=False
        If True, n_clusters-partition and all the more coarse-grained
        ones will be determined; in such a case, the labels_ attribute
        will be a matrix
    postprocess : str, one of "boundary" (default), "none", "all"
        In effect only if M>1. By default, only "boundary" points are merged
        with their nearest "core" points. To force a classical
        n_clusters-partition of a data set (with no notion of noise),
        choose "all".
    exact : bool, default=True
        TODO: Not yet implemented.
        If False, the minimum spanning tree is approximated
        based on the nearest neighbours graph. Finding nearest neighbours
        in low dimensional spaces is usually fast. Otherwise,
        the algorithm will need to inspect all pairwise distances,
        which gives the time complexity of O(n_samples*n_samples*n_features).
    cast_float32 : bool, default=True
        Allow casting input data to a float32 dense matrix
        (for efficiency reasons; decreases the run-time ~2x times
        at a cost of greater memory usage).
        TODO: Note that some nearest neighbour search
        methods require float32 data anyway.
        TODO: Might be a problem if the input matrix is sparse, but
        we don't support this yet.
    use_mlpack : bool or "auto", default="auto"
        Use mlpack.emst() for computing the Euclidean minimum spanning tree?
        Might be faster for lower-dimensional spaces. As the name suggests,
        only affinity='euclidean' is supported (and M<=2).
        By default, we rely on mlpack if it is installed and n_features <= 6.


    Attributes
    ----------

    labels_ : ndarray, shape (n_samples,) or (<=n_clusters+1, n_samples), or None
        If n_clusters==0, no labels_ are generated (None).
        If compute_all_cuts==True (the default), these are the detected
        cluster labels of each point: an integer vector with labels_[i]
        denoting the cluster id (in {0, ..., n_clusters-1}) of the i-th object.
        If M>1, noise points are labelled -1 (unless taken care of in the
        postprocessing stage).
        Otherwise, i.e., if compute_all_cuts==False,
        all partitions of cardinality down to n_clusters (if n_samples
        and the number of noise points allows) are determined.
        In such a case, labels_[j,i] denotes the cluster id of the i-th
        point in a j-partition.
        We assume that a 0- and 1- partition only distinguishes between
        noise- and non-noise points, however, no postprocessing
        is conducted on the 0-partition (there might be points with
        labels -1 even if postprocess=="all").
    n_clusters_ : int
        The number of clusters detected by the algorithm.
        If 0, then labels_ are not set.
        Note that the actual number might be larger than the n_clusters
        requested, for instance, if there are many noise points.
    n_samples_ : int
        The number of points in the fitted dataset.
    n_features_ : int or None
        The number of features in the fitted dataset.
    is_noise_ : ndarray, shape (n_samples,) or None
        is_noise_[i] is True iff the i-th point is a noise one;
        For M=1, all points are no-noise ones.
        Points are marked as noise even if postprocess=="all".
        Note that boundary points are also marked as noise points.
    children_ : ndarray, shape (n_samples-1, 2)
        The i-th row provides the information on the clusters merged at
        the i-th iteration. Noise points are merged first, with
        the corresponding distances_[i] of 0.
        See the description of Z[i,0] and Z[i,1] in
        scipy.cluster.hierarchy.linkage. Together with distances_ and
        counts_, this forms the linkage matrix that can be used for
        plotting the dendrogram.
        Only available if compute_full_tree==True.
    distances_ : ndarray, shape (n_samples-1,)
        Distance between the two clusters merged at the i-th iteration.
        As Genie does not guarantee that that distances are
        ordered increasingly (do not panic, there are some other hierarchical
        clustering linkages that also violate the ultrametricity property),
        these are corrected by applying
        `distances_ = genieclust.tools.cummin(distances_[::-1])[::-1]`.
        See the description of Z[i,2] in scipy.cluster.hierarchy.linkage.
        Only available if compute_full_tree==True.
    counts_ : ndarray, shape (n_samples-1,)
        Number of elements in a cluster created at the i-th iteration.
        See the description of Z[i,3] in scipy.cluster.hierarchy.linkage.
        Only available if compute_full_tree==True.
    """

    def __init__(self,
            n_clusters=2,
            gini_threshold=0.3,
            M=1,
            affinity="euclidean",
            compute_full_tree=True,
            compute_all_cuts=False,
            postprocess="boundary",
            exact=True,
            cast_float32=True,
            use_mlpack="auto"
        ):
        super().__init__(M, affinity, exact, cast_float32, use_mlpack)

        self.n_clusters = n_clusters
        self.gini_threshold = gini_threshold
        self.compute_full_tree = compute_full_tree
        self.compute_all_cuts = compute_all_cuts
        self.postprocess = postprocess

        self.n_clusters_  = 0 # should not be confused with self.n_clusters
        self.labels_      = None
        self.is_noise_    = None
        self.children_    = None
        self.distances_   = None
        self.counts_      = None
        self._links_      = None
        self._iters_      = None



    def fit(self, X, y=None):
        """Perform clustering of the X dataset.
        See the labels_ and n_clusters_ attributes for the clustering result.


        Parameters
        ----------

        X : ndarray, shape (n_samples, n_features)  or
                (n_samples*(n_samples-1)/2, ) or (n_samples, n_samples)
            A matrix defining n_samples in a vector space with n_features.
            Hint: it might be a good idea to normalise the coordinates of the
            input data points by calling
            `X = ((X-X.mean(axis=0))/X.std(axis=None, ddof=1)).astype(np.float32, order="C", copy=False)`
            so that the dataset is centred at 0 and
            has total variance of 1. This way the method becomes
            translation and scale invariant.
            However, if affinity="precomputed", then X is assumed to define
            all pairwise distances between n_samples
            (either in form of a distance vector or square distance matrix).
        y : None
            Ignored.


        Returns
        -------

        self
        """
        super().fit(X, y)
        cur_state = self._last_state_

        cur_state["n_clusters"] = int(self.n_clusters)
        if cur_state["n_clusters"] < 0:
            raise ValueError("n_clusters must be >= 0")

        cur_state["gini_threshold"] = float(self.gini_threshold)
        if not (0.0 <= cur_state["gini_threshold"] <= 1.0):
            raise ValueError("gini_threshold not in [0,1]")

        _postprocess_options = ("boundary", "none", "all")
        cur_state["postprocess"] = str(self.postprocess).lower()
        if cur_state["postprocess"] not in _postprocess_options:
            raise ValueError("postprocess should be one of %r"%_postprocess_options)

        cur_state["compute_full_tree"] = bool(self.compute_full_tree)
        cur_state["compute_all_cuts"] = bool(self.compute_all_cuts)


        # apply the Genie++ algorithm (the fast part):
        res = internal.genie_from_mst(self._mst_dist_, self._mst_ind_,
            n_clusters=cur_state["n_clusters"],
            gini_threshold=cur_state["gini_threshold"],
            noise_leaves=(cur_state["M"]>1),
            compute_full_tree=cur_state["compute_full_tree"],
            compute_all_cuts=cur_state["compute_all_cuts"])

        self.n_clusters_ = res["n_clusters"]
        self.labels_     = res["labels"]
        self._links_     = res["links"]
        self._iters_     = res["iters"]

        if self.labels_ is not None:
            self._postprocess(cur_state["M"], cur_state["postprocess"])

        if cur_state["compute_full_tree"]:
            Z = internal.get_linkage_matrix(self._links_,
                self._mst_dist_, self._mst_ind_)
            self.children_    = Z["children"]
            self.distances_   = Z["distances"]
            self.counts_      = Z["counts"]

        return self




class GIc(GenieBase):
    """GIc (Genie+Information Criterion) Information-Theoretic
    Hierarchical Clustering Algorithm

    Computes a k-partition based on a pre-computed MST
    maximising (heuristically) the information criterion [2].

    GIc has been proposed by Anna Cena in [1] and was inspired
    by Mueller's (et al.) ITM [2] and Gagolewski's (et al.) Genie [3]

    GIc uses a bottom-up, agglomerative approach (as opposed to the ITM,
    which follows a divisive scheme). It greedily selects for merging
    a pair of clusters that maximises the information criterion [2].
    By default, the initial partition is determined by considering
    the intersection of clusterings found by the Genie methods with
    thresholds 0.1, 0.3, 0.5 and 0.7.


    References
    ==========

    [1] Cena A., Adaptive hierarchical clustering algorithms based on
    data aggregation methods, PhD Thesis, Systems Research Institute,
    Polish Academy of Sciences 2018.

    [2] Mueller A., Nowozin S., Lampert C.H., Information Theoretic
    Clustering using Minimum Spanning Trees, DAGM-OAGM 2012.

    [3] Gagolewski M., Bartoszuk M., Cena A.,
    Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
    Information Sciences 363, 2016, pp. 8-23. doi:10.1016/j.ins.2016.05.003


    Parameters
    ----------

    n_clusters : int >= 0, default=2
        see `Genie`
    gini_thresholds : float in [0,1], default=[0.1, 0.3, 0.5, 0.7]
        The GIc algorithm optimises the information criterion
        in an agglomerative way, starting from the intersection
        of the clusterings returned by
        Genie(n_clusters=n_clusters+add_clusters, gini_threshold=gini_thresholds[i]),
        for all i=0,...,len(gini_thresholds)-1.
    add_clusters : int, default=0
        Number of additional clusters to work with internally.
    n_features : float or None, default None
        Dataset (intrinsic) dimensionality, if None, it will be set based on
        the shape of the input matrix.
    M : int, default=1
        see `Genie`
    affinity : str, default="euclidean"
        see `Genie`
    compute_full_tree : bool, default=True
        see `Genie`
    compute_all_cuts : bool, default=False
        see `Genie`
        Note that for GIc if compute_all_cuts==True,
        then the i-th cut in the hierarchy behaves as if
        add_clusters=n_clusters-i. In other words, the returned cuts
        will not be the same as those obtained by calling
        GIc numerous times, each time with different n_clusters requested.
    postprocess : str, one of "boundary" (default), "none", "all"
        see `Genie`
    exact : bool, default=True
        see `Genie`
    cast_float32 : bool, default=True
        see `Genie`
    use_mlpack : bool or "auto", default="auto"
        see `Genie`


    Attributes
    ----------

    see `Genie`
    """
    def __init__(self,
            n_clusters=2,
            gini_thresholds=[0.1, 0.3, 0.5, 0.7],
            add_clusters=0,
            n_features=None,
            M=1,
            affinity="euclidean",
            compute_full_tree=True,
            compute_all_cuts=False,
            postprocess="boundary",
            exact=True,
            cast_float32=True,
            use_mlpack="auto"
        ):
        super().__init__(M, affinity, exact, cast_float32, use_mlpack)

        self.n_clusters = n_clusters
        self.n_features = n_features
        self.add_clusters = add_clusters
        self.gini_thresholds = gini_thresholds
        self.compute_full_tree = compute_full_tree
        self.compute_all_cuts = compute_all_cuts
        self.postprocess = postprocess

        self.n_clusters_  = 0 # should not be confused with self.n_clusters
        self.labels_      = None
        self.is_noise_    = None
        self.children_    = None
        self.distances_   = None
        self.counts_      = None
        self._links_      = None
        self._iters_      = None



    def fit(self, X, y=None):
        """Perform clustering of the X dataset.
        See the labels_ and n_clusters_ attributes for the clustering result.


        Parameters
        ----------

        X : ndarray, shape (n_samples, n_features) or
                (n_samples*(n_samples-1)/2, ) or (n_samples, n_samples)
            see `Genie.fit()`
        y : None
            Ignored.


        Returns
        -------

        self
        """
        super().fit(X, y)
        cur_state = self._last_state_

        cur_state["n_clusters"] = int(self.n_clusters)
        if cur_state["n_clusters"] < 0:
            raise ValueError("n_clusters must be >= 0")

        cur_state["add_clusters"] = int(self.add_clusters)
        if cur_state["add_clusters"] < 0:
            raise ValueError("add_clusters must be >= 0")

        cur_state["gini_thresholds"] = np.array(self.gini_thresholds)

        _postprocess_options = ("boundary", "none", "all")
        cur_state["postprocess"] = str(self.postprocess).lower()
        if cur_state["postprocess"] not in _postprocess_options:
            raise ValueError("postprocess should be one of %r"%_postprocess_options)

        cur_state["compute_full_tree"] = bool(self.compute_full_tree)
        cur_state["compute_all_cuts"] = bool(self.compute_all_cuts)

        if self.n_features is None:
            if self.n_features_ is None:
                raise ValueError("The n_features attribute must be set manually.")
            else:
                # X.shape[1], set by GenieBase.fit()
                cur_state["n_features"] = self.n_features_
        else:
            cur_state["n_features"] = self.n_features

        cur_state["n_features"] = max(1.0, cur_state["n_features"])

        # apply the Genie+Ic algorithm:
        res = internal.gic_from_mst(self._mst_dist_, self._mst_ind_,
            n_features=cur_state["n_features"],
            n_clusters=cur_state["n_clusters"],
            add_clusters=cur_state["add_clusters"],
            gini_thresholds=cur_state["gini_thresholds"],
            noise_leaves=(cur_state["M"]>1),
            compute_full_tree=cur_state["compute_full_tree"],
            compute_all_cuts=cur_state["compute_all_cuts"])

        self.n_clusters_ = res["n_clusters"]
        self.labels_     = res["labels"]
        self._links_     = res["links"]
        self._iters_     = res["iters"]

        if self.labels_ is not None:
            self._postprocess(cur_state["M"], cur_state["postprocess"])


        if cur_state["compute_full_tree"]:
            Z = internal.get_linkage_matrix(self._links_,
                self._mst_dist_, self._mst_ind_)
            self.children_    = Z["children"]
            self.distances_   = Z["distances"]
            self.counts_      = Z["counts"]

        return self
