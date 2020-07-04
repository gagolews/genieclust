"""
The Genie++ Clustering Algorithm
"""


# Copyright (C) 2018-2020 Marek Gagolewski (https://www.gagolewski.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License
# Version 3, 19 November 2007, published by the Free Software Foundation.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License Version 3 for more details.
# You should have received a copy of the License along with this program.
# If not, see <https://www.gnu.org/licenses/>.


import os
import sys
import math
import numpy as np
import scipy.sparse
from sklearn.base import BaseEstimator, ClusterMixin
from . import internal
import warnings


try:
    import nmslib
except ImportError:
    nmslib = None


try:
    import mlpack
except ImportError:
    mlpack = None


###############################################################################
###############################################################################
###############################################################################


class GenieBase(BaseEstimator, ClusterMixin):
    """
    Base class for Genie and GIc
    """

    def __init__(
            self,
            *,
            n_clusters,
            M,
            affinity,
            exact,
            compute_full_tree,
            compute_all_cuts,
            postprocess,
            cast_float32,
            mlpack_enabled,
            verbose):
        # # # # # # # # # # # #
        super().__init__()
        self._n_clusters        = n_clusters
        self._n_features        = None  # can be overwritten by GIc
        self._M                 = M
        self._affinity          = affinity
        self._exact             = exact
        self._compute_full_tree = compute_full_tree
        self._compute_all_cuts  = compute_all_cuts
        self._postprocess       = postprocess
        self._cast_float32      = cast_float32
        self._mlpack_enabled    = mlpack_enabled
        self._verbose           = verbose

        self.n_samples_        = None
        self.n_features_       = None
        self.n_clusters_       = 0  # should not be confused with self.n_clusters
        self.labels_           = None
        self.is_noise_         = None
        self.children_         = None
        self.distances_        = None
        self.counts_           = None

        self._mst_dist_        = None
        self._mst_ind_         = None
        self._nn_dist_         = None
        self._nn_ind_          = None
        self._d_core_          = None
        self._links_           = None
        self._iters_           = None

        self._last_state_      = None



    def _postprocess_outputs(self, res, cur_state):
        """
        (internal) Updates `self.labels_` and `self.is_noise_`
        """
        if cur_state["verbose"]:
            print("[genieclust] Postprocessing the outputs.", file=sys.stderr)

        self.n_clusters_ = res["n_clusters"]
        self.labels_     = res["labels"]
        self._links_     = res["links"]
        self._iters_     = res["iters"]

        if self.labels_ is not None:
            reshaped = False
            if self.labels_.ndim == 1:
                reshaped = True
                # promote it to a matrix with 1 row
                self.labels_.shape = (1, self.labels_.shape[0])
                start_partition = 0
            else:
                # duplicate the 1st row (create the "0"-partition that will
                # not be postprocessed):
                self.labels_ = np.vstack((self.labels_[0, :], self.labels_))
                start_partition = 1  # do not postprocess the "0"-partition

            self.is_noise_    = (self.labels_[0, :] < 0)

            # postprocess labels, if requested to do so
            if cur_state["M"] < 2 or cur_state["postprocess"] == "none":
                pass
            elif cur_state["postprocess"] == "boundary":
                assert self._nn_ind_ is not None
                assert self._nn_ind_.shape[1] >= cur_state["M"] - 1
                for i in range(start_partition, self.labels_.shape[0]):
                    self.labels_[i, :] = internal.merge_boundary_points(
                        self._mst_ind_, self.labels_[i, :],
                        self._nn_ind_, cur_state["M"])
            elif cur_state["postprocess"] == "all":
                for i in range(start_partition, self.labels_.shape[0]):
                    self.labels_[i, :] = internal.merge_noise_points(
                        self._mst_ind_, self.labels_[i, :])

        if reshaped:
            self.labels_.shape = (self.labels_.shape[1],)

        if cur_state["compute_full_tree"] and cur_state["M"] == 1:
            Z = internal.get_linkage_matrix(self._links_,
                self._mst_dist_, self._mst_ind_)
            self.children_    = Z["children"]
            self.distances_   = Z["distances"]
            self.counts_      = Z["counts"]

        return cur_state


    def _check_params(self, cur_state=None):
        if cur_state is None:
            cur_state = dict()

        cur_state["compute_full_tree"] = bool(self._compute_full_tree)
        cur_state["compute_all_cuts"]  = bool(self._compute_all_cuts)
        cur_state["exact"]             = bool(self._exact)
        cur_state["cast_float32"]      = bool(self._cast_float32)
        cur_state["verbose"]           = bool(self._verbose)

        cur_state["n_clusters"] = int(self._n_clusters)
        if cur_state["n_clusters"] < 0:
            raise ValueError("n_clusters must be >= 0")

        _postprocess_options = ("boundary", "none", "all")
        cur_state["postprocess"] = str(self._postprocess).lower()
        if cur_state["postprocess"] not in _postprocess_options:
            raise ValueError("`postprocess` should be one of %r" % _postprocess_options)

        _affinity_options = ("l2", "euclidean", "l1", "manhattan",
                             "cityblock", "cosine", "cosinesimil", "precomputed")
        cur_state["affinity"] = str(self._affinity).lower()
        if cur_state["affinity"] not in _affinity_options:
            raise ValueError("`affinity` should be one of %r" % _affinity_options)

        if cur_state["affinity"] in ["euclidean"]:
            cur_state["affinity"] = "l2"
        elif cur_state["affinity"] in ["manhattan", "cityblock"]:
            cur_state["affinity"] = "l1"
        elif cur_state["affinity"] in ["chebyshev", "maximum"]:
            cur_state["affinity"] = "linf"
        elif cur_state["affinity"] in ["cosine"]:
            cur_state["affinity"] = "cosinesimil"

        cur_state["M"] = int(self._M)
        if cur_state["M"] < 1:
            raise ValueError("`M` must be > 0.")

        if type(self._mlpack_enabled) is str:
            cur_state["mlpack_enabled"] = str(self._mlpack_enabled).lower()
            if cur_state["mlpack_enabled"] != "auto":
                raise ValueError("`mlpack_enabled` must be one of: 'auto', True, False.")
        else:
            cur_state["mlpack_enabled"] = bool(self._mlpack_enabled)

        # this is more like an inherent dimensionality for GIc
        cur_state["n_features"] = self._n_features   # users can set this manually
        if cur_state["n_features"] is not None:      # only GIc needs this
            cur_state["n_features"] = max(1.0, float(cur_state["n_features"]))
        else:
            cur_state["n_features"] = -1

        return cur_state




    def _get_mst(self, X, cur_state):
        cur_state["X"] = id(X)

        n_samples  = X.shape[0]
        if cur_state["affinity"] == "precomputed":
            X = X.reshape(X.shape[0], -1)
            if X.shape[1] not in [1, X.shape[0]]:
                raise ValueError("`X` must be distance vector "
                    "or a square-form distance matrix, "
                    "see `scipy.spatial.distance.pdist` or "
                    "`scipy.spatial.distance.squareform`.")
            if X.shape[1] == 1:
                # from a very advanced and sophisticated quadratic equation:
                n_samples = int(round((math.sqrt(1.0+8.0*n_samples)+1.0)/2.0))
                assert n_samples*(n_samples-1)//2 == X.shape[0]

        if cur_state["n_features"] < 0 and cur_state["affinity"] != "precomputed":
            cur_state["n_features"] = X.shape[1]

        if cur_state["mlpack_enabled"] == "auto":
            if mlpack is not None and \
                    cur_state["affinity"] == "l2" and \
                    X.shape[1] <= 6 and \
                    cur_state["M"] == 1:
                cur_state["mlpack_enabled"] = True
            else:
                cur_state["mlpack_enabled"] = False


        if cur_state["mlpack_enabled"] and mlpack is None:
            raise ValueError("Package `mlpack` is not available.")
        if cur_state["mlpack_enabled"] and cur_state["affinity"] != "l2":
            raise ValueError("`mlpack` can only be used with `affinity` = 'l2'.")
        if cur_state["mlpack_enabled"] and cur_state["M"] != 1:
            raise ValueError("`mlpack` can only be used with `M` = 1.")

        if cur_state["verbose"]:
            print("[genieclust] Initialising data.", file=sys.stderr)

        mst_dist = None
        mst_ind  = None
        nn_dist  = None
        nn_ind   = None
        d_core   = None

        if cur_state["cast_float32"] and cur_state["affinity"] != "precomputed":
            # faiss and nmslib support float32 only
            # warning if sparse!!
            # this is not needed if cache is used!
            X = X.astype(np.float32, order="C", copy=False)


        if self._last_state_ is not None and \
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
            raise NotImplementedError("Approximate method not implemented yet.")
            # TODO: warn if warnings.warn("The number of connected components......") #53
            #  if cur_state["affinity"] == "precomputed":
            #      raise ValueError('exact==False with affinity=="precomputed"')
            #
            #
            #  assert cur_state["affinity"] == "l2"
            #
            #  actual_n_neighbors = min(32, int(math.ceil(math.sqrt(n_samples))))
            #  actual_n_neighbors = max(actual_n_neighbors, cur_state["M"]-1)
            #  actual_n_neighbors = min(n_samples-1, actual_n_neighbors)
            #
            #  # t0 = time.time()
            #  ##nn = sklearn.neighbors.NearestNeighbors(
            #  ##n_neighbors=actual_n_neighbors, ....**cur_state["nn_params"])
            #  ##nn_dist, nn_ind = nn.fit(X).kneighbors()
            #  #nn_dist, nn_ind = internal.knn_from_distance(
            #  #X, k=actual_n_neighbors, ...metric=metric)
            #  # print("T=%.3f" % (time.time()-t0), end="\t")
            #
            #  # FAISS - `l2` and `cosinesimil` only!
            #
            #
            #
            #  # TODO:  cur_state["metric"], cur_state["metric_params"]
            #  #t0 = time.time()
            #  # the slow part:
            #  nn = faiss.IndexFlatL2(cur_state["n_features"])
            #  nn.add(X)
            #  nn_dist, nn_ind = nn.search(X, actual_n_neighbors+1) # TODO: , verbose=cur_state["verbose"]
            #  #print("T=%.3f" % (time.time()-t0), end="\t")
            #
            #
            #
            #  # @TODO:::::
            #  #nn_bad_where = np.where((nn_ind[:, 0]!=np.arange(n_samples)))[0]
            #  #print(nn_bad_where)
            #  #print(nn_ind[nn_bad_where, :5])
            #  #print(X[nn_bad_where, :])
            #  #assert nn_bad_where.shape[0] == 0
            #
            #  # TODO: check cache if rebuild needed
            #  nn_dist = nn_dist[:, 1:].astype(X.dtype, order="C")
            #  nn_ind  = nn_ind[:, 1:].astype(np.intp, order="C")
            #
            #  if cur_state["M"] > 1:
            #      # d_core = nn_dist[:, cur_state["M"]-2].astype(X.dtype, order="C")
            #      raise NotImplementedError("approximate method not implemented yet")
            #
            #  #t0 = time.time()
            #  # the fast part:
            #  mst_dist, mst_ind = internal.mst_from_nn(nn_dist, nn_ind,
            #      stop_disconnected=False, # TODO: test this!!!!
            #      stop_inexact=False,
            #      verbose=cur_state["verbose"])
            #  #print("T=%.3f" % (time.time()-t0), end="\t")

        else:  # cur_state["exact"]
            if cur_state["mlpack_enabled"]:
                assert cur_state["M"] == 1
                assert cur_state["affinity"] == "l2"

                if mst_dist is None or mst_ind is None:
                    _res = mlpack.emst(
                        input=X,
                        #leaf_size=...,
                        #naive=False,
                        copy_all_inputs=False,
                        verbose=cur_state["verbose"])["output"]
                    mst_dist = _res[:,  2].astype(X.dtype, order="C")
                    mst_ind  = _res[:, :2].astype(np.intp, order="C")
            else:
                if cur_state["M"] >= 2:  # else d_core   = None
                    # Genie+HDBSCAN --- determine d_core
                    # TODO: mlpack for k-nns?
                    if nn_dist is None or nn_ind is None:
                        nn_dist, nn_ind = internal.knn_from_distance(
                            X, # if not c_contiguous, raises an error
                            k=cur_state["M"]-1,
                            metric=cur_state["affinity"], # supports "precomputed"
                            verbose=cur_state["verbose"])

                    assert nn_dist.shape[1] >= cur_state["M"]-1
                    d_core = nn_dist[:, cur_state["M"]-2].astype(X.dtype, order="C")

                # Use Prim's algorithm to determine the MST
                # w.r.t. the distances computed on the fly
                if mst_dist is None or mst_ind is None:
                    mst_dist, mst_ind = internal.mst_from_distance(
                        X, # if not c_contiguous, raises an error
                        metric=cur_state["affinity"],
                        d_core=d_core,
                        verbose=cur_state["verbose"])

        # this might be an "intristic" dimensionality:
        self.n_features_  = cur_state["n_features"]
        self.n_samples_   = n_samples
        self._mst_dist_   = mst_dist
        self._mst_ind_    = mst_ind
        self._nn_dist_    = nn_dist
        self._nn_ind_     = nn_ind
        self._d_core_     = d_core
        self._last_state_ = cur_state  # will be modified in-place further on

        return cur_state


    def fit_predict(self, X, y=None):
        """
        Perform cluster analysis of a dataset and return the predicted labels.


        Parameters
        ----------

        X : object
            See `genieclust.Genie.fit`.
        y : None
            See `genieclust.Genie.fit`.


        Returns
        -------

        labels_ : ndarray
            `self.labels_` attribute.


        See also
        --------

        genieclust.Genie.fit

        """
        self.fit(X)
        return self.labels_





###############################################################################
###############################################################################
###############################################################################



class Genie(GenieBase):
    """
    Genie++ hierarchical clustering algorithm


    Parameters
    ----------

    n_clusters : int
        Number of clusters to detect. Note that depending on the dataset
        and approximations used (see parameter `exact`), the actual
        partition cardinality can be smaller.
        `n_clusters` equal to 1 can act as a noise point/outlier detector
        (if `M` > 1 and `postprocess` is not ``"all"``).
        `n_clusters` equal to 0 computes the whole dendrogram but doesn't
        generate any particular cuts.
    M : int
        Smoothing factor. `M` = 1 gives the original Genie algorithm.
    affinity : str
        Metric used to compute the linkage. One of:
        ``"l2"`` (or ``"euclidean"``),
        ``"l1"`` (or ``"manhattan"``, ``"cityblock"``),
        ``"cosinesimil"`` (or ``"cosine"``), or ``"precomputed"``.
        If ``"precomputed"``, a :math:`n (n-1)/2` distance vector
        or a square-form distance matrix is needed on input (argument `X`)
        for the `fit` method, see `scipy.spatial.distance.pdist` or
        `scipy.spatial.distance.squareform`, amongst others.
    compute_full_tree : bool
        If True, a complete hierarchy is determined
        TODO: makes sense only if `M` = 1, for plotting of dendrograms
        TODO: only if exact
        TODO: `children_`, `distances_`, `counts_`
    compute_all_cuts : bool
        If True, `n_clusters`-partition and all the more coarse-grained
        ones will be determined; in such a case, the `labels_` attribute
        will be a matrix.
        TODO: for approximate method the obtained clusterings maybe more fine-grained
    postprocess : {"boundary", "none", "all"}
        In effect only if `M` > 1. By default, only "boundary" points are merged
        with their nearest "core" points (A point is a boundary point if it is
        a noise point and it's amongst its adjacent vertex's
        `M` - 1 nearest neighbours). To force a classical
        n_clusters-partition of a data set (with no notion of noise),
        choose "all".
    exact : bool
        TODO: Not yet implemented.
        If ``False``, the minimum spanning tree is approximated
        based on an approximate nearest neighbours graph found by
        `nmslib`. Otherwise, the algorithm will need to inspect all pairwise distances,
        which gives the time complexity of :math:`O(d n^2)`.
        Otherwise [3]_
    cast_float32 : bool
        Allow casting input data to a ``float32`` dense matrix
        (for efficiency reasons; decreases the run-time ~2x times
        at a cost of greater memory usage)? Note that `nmslib`
        (used when `exact` is ``False``) requires ``float32`` data anyway.
    mlpack_enabled : "auto" or bool
        Use `mlpack.emst` for computing the Euclidean minimum spanning tree?
        Might be faster for lower-dimensional spaces. As the name suggests,
        only affinity='l2' is supported (and M=1).
        By default, we rely on `mlpack` if it's installed and `n_features` <= 6.
        Otherwise, we use our own parallelised implementation of Prim's
        algorithm (environment variable ``OMP_NUM_THREADS`` controls
        the number of threads used).
    verbose : bool
        Whether to print diagnostic messages and progress information on ``stderr``.
    gini_threshold : float
        The threshold for the Genie correction in [0,1], i.e.,
        the Gini index of the cluster size distribution.
        Threshold of 1.0 disables the correction.
        Low thresholds highly penalise the formation of small clusters.


    Attributes
    ----------

    labels_ : ndarray
        shape (n_samples,) or (<=n_clusters+1, n_samples), or None
        If `n_clusters` = 0, no `labels_` are generated (``None``).
        If `compute_all_cuts` = ``True`` (the default), these are the detected
        cluster labels of each point: an integer vector with ``labels_[i]``
        denoting the cluster id (in {0, ..., `n_clusters` - 1}) of the i-th object.
        If `M` > 1, noise points are labelled -1 (unless taken care of in the
        postprocessing stage).
        Otherwise, i.e., if `compute_all_cuts` = ``False``,
        all partitions of cardinality down to n_clusters (if `n_samples`
        and the number of noise points allows) are determined.
        In such a case, ``labels_[j,i]`` denotes the cluster id of the i-th
        point in a j-partition.
        We assume that a 0- and 1- partition only distinguishes between
        noise- and non-noise points, however, no postprocessing
        is conducted on the 0-partition (there might be points with
        labels -1 even if `postprocess` = ``"all"``).
        Note that the approximate method (`exact` = ``False``) might fail
        to determine the fine-grained clusters (if the approximate
        neighbour graph is disconnected) and then the first few rows
        might be identical.
    n_clusters_ : int
        The number of clusters detected by the algorithm.
        If 0, then `labels_` are not set.
        Note that the actual number might be greater than the requested one,
        for instance, due to the presence of too many noise points.
        TODO when exact=False
    n_samples_ : int
        The number of points in the fitted dataset.
    n_features_ : int or None
        The number of features in the fitted dataset.
    is_noise_ : ndarray, shape (n_samples,) or None
        ``is_noise_[i]`` is True iff the i-th point is a noise one;
        For `M` = 1, all points are no-noise ones.
        Points are marked as noise even if `postprocess` equals ``"all"``.
        Note that boundary points are also marked as noise points.
    children_ : ndarray, shape (n_samples-1, 2)
        The i-th row provides the information on the clusters merged at
        the i-th iteration. Noise points are merged first, with
        the corresponding ``distances_[i]`` of 0.
        See the description of ``Z[i,0]`` and ``Z[i,1]`` in
        `scipy.cluster.hierarchy.linkage`. Together with `distances_` and
        `counts_`, this forms the linkage matrix that can be used for
        plotting the dendrogram.
        Only available if `compute_full_tree` = ``True``.
        `M` = 1  only
    distances_ : ndarray, shape (n_samples-1,)
        Distance between the two clusters merged at the *i*-th iteration.
        As Genie does not guarantee that that distances are
        ordered increasingly (do not panic, there are some other hierarchical
        clustering linkages that also violate the ultrametricity property),
        these are corrected by applying
        ``distances_ = genieclust.tools.cummin(distances_[::-1])[::-1]``.
        See the description of ``Z[i,2]`` in `scipy.cluster.hierarchy.linkage`.
        Only available if `compute_full_tree` is ``True``.
        `M` = 1 only
        According to the algorithm's original definition,
        the resulting partition tree (dendrogram) might violate
        the ultrametricity property (merges might occur at levels that
        are not increasing w.r.t. a between-cluster distance).
        Departures from ultrametricity are corrected by applying
        ``Z[:, 2] = genieclust.tools.cummin(Z[::-1,2])[::-1]``.
    counts_ : ndarray, shape (n_samples-1,)
        Number of elements in a cluster created at the i-th iteration.
        See the description of ``Z[i,3]`` in `scipy.cluster.hierarchy.linkage`.
        Only available if `compute_full_tree` is ``True``.
        `M` = 1 only



    Notes
    -----

    A reimplementation of **Genie** - a robust and outlier resistant
    hierarchical clustering algorithm [1]_, originally published
    as an R package ``genie``. Features optional smoothing and
    noise point detection (if `M` > 1).

    The Genie algorithm is based on a minimum spanning tree (MST) of the
    pairwise distance graph of a given point set.
    Just like the single linkage, it consumes the edges
    of the MST in increasing order of weights. However, it prevents
    the formation of clusters of highly imbalanced sizes; once the Gini index
    of the cluster size distribution raises above an assumed threshold,
    a forced merge of a point group of the smallest size is performed.
    Its appealing simplicity goes hand in hand with its usability;
    Genie often outperforms other clustering approaches on benchmark data.

    The Genie algorithm itself has :math:`O(n \\sqrt{n})` time complexity
    given a minimum spanning tree of the pairwise distance graph.
    Generally, our parallelised implementation of a Jarník (Prim/Dijkstra)-like
    method will be called to compute an MST, which takes :math:`O(d n^2)` time.
    However, `MLPACK` (see Python package `mlpack`) [4]_ provides a very fast
    alternative in the case of Euclidean spaces of (very) low dimensionality
    and `M` = 1, see [5]_ and the `mlpack_enabled` parameter.
    Moreover, in the approximate method (`exact` = ``False``) we apply
    the Kruskal algorithm on the nearest neighbour graph determined
    by `nmslib` [3]_. Albeit this only gives *some* sort of a spanning *forest*,
    such a data structure turns out to be very suitable for our clustering task
    (note that the set of connected components will determine the top
    level of the identified cluster hierarchy).



    The Genie correction together with the smoothing factor `M` > 1
    gives a robustified version of the HDBSCAN\\* [2]_ algorithm that,
    contrary to its predecessor, is able to detect a *predefined* number of
    clusters. Hence, it is independent of the *DBSCAN*'s somehow magical
    ``eps`` parameter or the *HDBSCAN*'s ``min_cluster_size`` one.
    If `M` > 1, then the minimum spanning tree is computed with respect to the
    mutual reachability distance (based, e.g., on the Euclidean metric).
    Formally, the distance :math:`m(i,j)` is used instead of the
    chosen "raw" distance, :math:`d(i,j)`. It holds
    :math:`m(i,j)=\\max(d(i,j), c(i), c(j))`,
    where the "core" distance :math:`c(i)` is given by
    :math:`d(i,k)` with :math:`k` being the (`M` - 1)-th nearest neighbour
    of :math:`i`. This makes "noise" and "boundary" points being "pulled away"
    from each other, however, note that `M` = 2 corresponds to the
    original distance. During the clustering procedure, all leaves of the MST
    do not take part in the clustering process. They may be merged
    with the nearest clusters during the postprocessing stage,
    or left marked as "noise"
    observations.







    :Environment variables:
        OMP_NUM_THREADS
            Controls the number of threads used for computing the minimum
            spanning tree (not supported by `mlpack`).


    References
    ----------

    .. [1]
        Gagolewski M., Bartoszuk M., Cena A.,
        Genie: A new, fast, and outlier-resistant hierarchical
        clustering algorithm, *Information Sciences* 363, 2016, 8-23.
        doi:10.1016/j.ins.2016.05.003.

    .. [2]
        Campello R., Moulavi D., Zimek A., Sander J.,
        Hierarchical density estimates for data clustering, visualization,
        and outlier detection,
        *ACM Transactions on Knowledge Discovery from Data* 10(1),
        2015, 5:1–5:51. doi:10.1145/2733381.

    .. [3]
        Naidan B., Boytsov L., Malkov Y.,  Novak D.,
        *Non-metric space library (NMSLIB) manual*, version 2.0, 2019.
        https://github.com/nmslib/nmslib/blob/master/manual/latex/manual.pdf.

    .. [4]
        Curtin R.R., Edel M., Lozhnikov M., Mentekidis Y., Ghaisas S., Zhang S.,
        mlpack 3: A fast, flexible machine learning library,
        *Journal of Open Source Software* 3(26), 726, 2018.
        doi:10.21105/joss.00726.

    .. [5]
        March W.B., Ram P., Gray A.G.,
        Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis, and Applications,
        *Proc. ACM SIGKDD'10*, 2010, 603-611.

    """

    def __init__(
            self,
            n_clusters=2,
            *,
            gini_threshold=0.3,
            M=1,
            affinity="l2",
            exact=True,
            compute_full_tree=False,
            compute_all_cuts=False,
            postprocess="boundary",
            cast_float32=True,
            mlpack_enabled="auto",
            #nmslib_n_neighbors="auto",
            #mlpack_leaf_size=1 Leaf size in the kd-tree. According to the mlpack manual, leaves of size 1 give the best performance at the cost of greater memory requirements.
            #nmslib_params_init=dict(method="hnsw") #`space` forbidden
            #nmslib_params_index=dict(post=2) #`indexThreadQty` forbidden
            #nmslib_params_query=dict()
            verbose=False):
        # # # # # # # # # # # #
        super().__init__(
            n_clusters=n_clusters,
            M=M,
            affinity=affinity,
            exact=exact,
            compute_full_tree=compute_full_tree,
            compute_all_cuts=compute_all_cuts,
            postprocess=postprocess,
            cast_float32=cast_float32,
            mlpack_enabled=mlpack_enabled,
            verbose=verbose)

        self._gini_threshold = gini_threshold

        self._check_params()




    def _check_params(self, cur_state=None):
        cur_state = super()._check_params(cur_state)

        cur_state["gini_threshold"] = float(self._gini_threshold)
        if not (0.0 <= cur_state["gini_threshold"] <= 1.0):
            raise ValueError("`gini_threshold` not in [0,1].")

        return cur_state


    def fit(self, X, y=None):
        """
        Perform cluster analysis of a dataset.


        Parameters
        ----------

        X : object
            Typically a matrix with ``n_samples`` rows and ``n_features``
            columns, see below for more details and options.
        y : None
            Ignored.


        Returns
        -------

        self : genieclust.Genie
            The object that the method was called on.


        See also
        --------

        genieclust.Genie.fit_predict


        Notes
        -----

        Refer to the `labels_` and `n_clusters_` attributes for the result.


        Acceptable `X` types depend whether we use the exact or the approximate
        method.

        `X` when `exact` = ``True``
            For `affinity` of ``"precomputed"``, `X` should either
            be a distance vector of length ``n_samples*(n_samples-1)/2``
            (see `scipy.spatial.distance.pdist`) or a square distance matrix
            of shape ``(n_samples, n_samples)``
            (see `scipy.spatial.distance.squareform`).

            Otherwise, `X` should be real-valued matrix (dense, ``numpy.ndarray``)
            with ``n_samples`` rows and ``n_features`` columns.

            In the latter case, it might be a good idea to standardise
            or at least somehow preprocess the coordinates of the input data
            points by calling, for instance,
            ``X = (X-X.mean(axis=0))/X.std(axis=None, ddof=1)``
            so that the dataset is centred at 0 and has total variance of 1.
            This way the method becomes translation and scale invariant.
            What's more, if data are recorded with small precision (say, up
            to few decimal digits), adding a tiny bit of Gaussian noise will
            assure the solution is unique (note that this generally applies
            to other distance-based clustering algorithms as well).

        `X` when `exact` = ``False``
            The approximate method relies on `nmslib` for locating the
            nearest neighbours. Therefore, it supports all datatypes
            described in https://github.com/nmslib/nmslib/blob/master/manual/spaces.md.
            Depending on the chosen `affinity`, `X` may hence be
            a real-valued ``numpy.ndarray`` matrix with ``n_samples`` rows
            and ``n_features`` columns,
            a ``scipy.sparse.csr_matrix`` object, or
            an array of ASCII strings.



        References
        ----------

        .. [1]
            Naidan B., Boytsov L., Malkov Y.,  Novak D.,
            *Non-metric space library (NMSLIB) manual*, version 2.0, 2019.
            https://github.com/nmslib/nmslib/blob/master/manual/latex/manual.pdf.


        """
        cur_state = self._check_params()  # re-check, they might have changed

        cur_state = self._get_mst(X, cur_state)

        if cur_state["verbose"]:
            print("[genieclust] Determining clusters with GIc.", file=sys.stderr)

        # apply the Genie++ algorithm (the fast part):
        res = internal.genie_from_mst(self._mst_dist_, self._mst_ind_,
            n_clusters=cur_state["n_clusters"],
            gini_threshold=cur_state["gini_threshold"],
            noise_leaves=(cur_state["M"] > 1),
            compute_full_tree=cur_state["compute_full_tree"],
            compute_all_cuts=cur_state["compute_all_cuts"])

        cur_state = self._postprocess_outputs(res, cur_state)

        if cur_state["verbose"]:
            print("[genieclust] Done.", file=sys.stderr)

        return self




###############################################################################
###############################################################################
###############################################################################



class GIc(GenieBase):
    """
    (**EXPERIMENTAL**) GIc hierarchical clustering algorithm



    Parameters
    ----------

    n_clusters : int
        See `genieclust.Genie`.
    M : int
        See `genieclust.Genie`.
    affinity : str
        See `genieclust.Genie`.
    compute_full_tree : bool
        See `genieclust.Genie`.
    compute_all_cuts : bool
        See `genieclust.Genie`.
        Note that if `compute_all_cuts` is ``True``,
        then the i-th cut in the hierarchy behaves as if
        `add_clusters` was equal to ``n_clusters-i``.
        In other words, the returned cuts will not be the same
        as those obtained by multiple calls to GIc, each time
        with different `n_clusters` and constant `add_clusters` requested.
    postprocess : {"boundary", "none", "all"}
        See `genieclust.Genie`.
    exact : bool, default=True
        See `genieclust.Genie`.
    cast_float32 : bool
        See `genieclust.Genie`.
    mlpack_enabled : "auto" or bool
        See `genieclust.Genie`.
    verbose : bool
        See `genieclust.Genie`.
    gini_thresholds : array_like
        A list of Gini index thresholds between 0 and 1.
        The GIc algorithm optimises the information criterion
        in an agglomerative way, starting from the intersection
        of the clusterings returned by
        ``Genie(n_clusters=n_clusters+add_clusters, gini_threshold=gini_thresholds[i])``,
        for all ``i`` from ``0`` to ``len(gini_thresholds)-1``.
    add_clusters : int
        Number of additional clusters to work with internally.
    n_features : float or None
        Dataset's (intrinsic) dimensionality;
        if ``None``, it will be set based on the shape of the input matrix.
        `affinity` of ``"precomputed"`` needs this to be set manually.

    Attributes
    ----------

    See class `genieclust.Genie`.



    See also
    --------

    genieclust.Genie



    Notes
    -----

    GIc (Genie+Information Criterion) is an Information-Theoretic
    Hierarchical Clustering Algorithm. It computes an `n_clusters`-partition
    based on a pre-computed minimum spanning tree. Clusters are merged
    so as to maximise (heuristically) the information
    criterion discussed in [2]_.

    GIc was proposed by Anna Cena in [1]_ and was inspired
    by Mueller's (et al.) ITM [2]_ and Gagolewski's (et al.) Genie [3]_.

    GIc uses a bottom-up, agglomerative approach (as opposed to the ITM,
    which follows a divisive scheme). It greedily selects for merging
    a pair of clusters that maximises the information criterion [2]_.
    By default, the initial partition is determined by considering
    the intersection of the clusterings found by multiple runs of
    the Genie++ method with thresholds [0.1, 0.3, 0.5, 0.7].


    References
    ----------

    .. [1]
        Cena A., *Adaptive hierarchical clustering algorithms based on
        data aggregation methods*, PhD Thesis, Systems Research Institute,
        Polish Academy of Sciences 2018.

    .. [2]
        Mueller A., Nowozin S., Lampert C.H., Information Theoretic
        Clustering using Minimum Spanning Trees, *DAGM-OAGM*, 2012.

    .. [3]
        Gagolewski M., Bartoszuk M., Cena A.,
        Genie: A new, fast, and outlier-resistant hierarchical clustering
        algorithm, *Information Sciences* 363, 2016, 8-23.
        doi:10.1016/j.ins.2016.05.003.


    """
    def __init__(
            self,
            n_clusters=2,
            *,
            gini_thresholds=[0.1, 0.3, 0.5, 0.7],
            M=1,
            affinity="l2",
            exact=True,
            compute_full_tree=False,
            compute_all_cuts=False,
            postprocess="boundary",
            add_clusters=0,
            n_features=None,
            cast_float32=True,
            mlpack_enabled="auto",
            # TODO: Genie..............
            verbose=False):
        # # # # # # # # # # # #
        super().__init__(
            n_clusters=n_clusters,
            M=M,
            affinity=affinity,
            exact=exact,
            compute_full_tree=compute_full_tree,
            compute_all_cuts=compute_all_cuts,
            postprocess=postprocess,
            cast_float32=cast_float32,
            mlpack_enabled=mlpack_enabled,
            verbose=verbose)

        self._gini_thresholds = gini_thresholds
        self._n_features      = n_features
        self._add_clusters    = add_clusters

        self._check_params()



    def _check_params(self, cur_state=None):
        cur_state = super()._check_params(cur_state)

        cur_state["add_clusters"] = int(self._add_clusters)
        if cur_state["add_clusters"] < 0:
            raise ValueError("`add_clusters` must be non-negative.")

        cur_state["gini_thresholds"] = np.array(self._gini_thresholds)
        for g in cur_state["gini_thresholds"]:
            if not (0.0 <= g <= 1.0):
                raise ValueError("All elements in `gini_thresholds` "
                                 "must be in [0,1].")

        return cur_state


    def fit(self, X, y=None):
        """
        Perform cluster analysis of a dataset.


        Parameters
        ----------

        X : object
            See `genieclust.Genie.fit`.
        y : None
            Ignored.


        Returns
        -------

        self : genieclust.GIc
            The object that the method was called on.


        See also
        --------

        genieclust.Genie.fit

        genieclust.GIc.fit_predict


        Notes
        -----

        Refer to the `labels_` and `n_clusters_` attributes for the result.

        Note that for `affinity` of ``"precomputed"``, the `n_features`
        parameter must be set explicitly.

        """
        cur_state = self._check_params()  # re-check, they might have changed

        cur_state = self._get_mst(X, cur_state)

        if cur_state["n_features"] < 1:  # _get_mst sets this
            # this shouldn't happen in normal use
            raise ValueError("Please set the `_n_features` attribute manually.")

        if cur_state["verbose"]:
            print("[genieclust] Determining clusters with Genie++.", file=sys.stderr)

        # apply the Genie+Ic algorithm:
        res = internal.gic_from_mst(self._mst_dist_, self._mst_ind_,
            n_features=cur_state["n_features"],
            n_clusters=cur_state["n_clusters"],
            add_clusters=cur_state["add_clusters"],
            gini_thresholds=cur_state["gini_thresholds"],
            noise_leaves=(cur_state["M"] > 1),
            compute_full_tree=cur_state["compute_full_tree"],
            compute_all_cuts=cur_state["compute_all_cuts"])

        cur_state = self._postprocess_outputs(res, cur_state)

        if cur_state["verbose"]:
            print("[genieclust] Done.", file=sys.stderr)

        return self
