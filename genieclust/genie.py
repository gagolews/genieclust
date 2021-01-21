"""
The Genie++ Clustering Algorithm
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2018-2021, Marek Gagolewski <https://www.gagolewski.com>      #
#                                                                              #
#                                                                              #
#   This program is free software: you can redistribute it and/or modify       #
#   it under the terms of the GNU Affero General Public License                #
#   Version 3, 19 November 2007, published by the Free Software Foundation.    #
#   This program is distributed in the hope that it will be useful,            #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the               #
#   GNU Affero General Public License Version 3 for more details.              #
#   You should have received a copy of the License along with this program.    #
#   If this is not the case, refer to <https://www.gnu.org/licenses/>.         #
#                                                                              #
# ############################################################################ #


import os
import sys
import math
import numpy as np
import scipy.sparse
from sklearn.base import BaseEstimator, ClusterMixin
from . import internal
import warnings

import nmslib



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

    For detailed description of the parameters and attributes,
    see `genieclust.Genie`.
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
            mlpack_leaf_size,
            nmslib_n_neighbors,
            nmslib_params_init,
            nmslib_params_index,
            nmslib_params_query,
            verbose):
        # # # # # # # # # # # #
        super().__init__()
        self.n_clusters          = n_clusters
        self.n_features          = None  # can be overwritten by GIc
        self.M                   = M
        self.affinity            = affinity
        self.exact               = exact
        self.compute_full_tree   = compute_full_tree
        self.compute_all_cuts    = compute_all_cuts
        self.postprocess         = postprocess
        self.cast_float32        = cast_float32
        self.mlpack_enabled      = mlpack_enabled
        self.mlpack_leaf_size    = mlpack_leaf_size
        self.nmslib_n_neighbors  = nmslib_n_neighbors
        self.nmslib_params_init  = nmslib_params_init
        self.nmslib_params_index = nmslib_params_index
        self.nmslib_params_query = nmslib_params_query
        self.verbose             = verbose

        self.n_samples_           = None
        self.n_features_          = None
        self.n_clusters_          = 0  # should not be confused with self.n_clusters
        self.labels_              = None
        # self.is_noise_            = None
        self.children_            = None
        self.distances_           = None
        self.counts_              = None

        self._mst_dist_           = None
        self._mst_ind_            = None
        self._nn_dist_            = None
        self._nn_ind_             = None
        self._d_core_             = None
        self._links_              = None
        self._iters_              = None

        self._last_state_         = None



    def _postprocess_outputs(self, res, cur_state):
        """
        (internal) Updates `self.labels_`
        """
        if cur_state["verbose"]:
            print("[genieclust] Postprocessing outputs.", file=sys.stderr)

        self.labels_     = res["labels"]
        self._links_     = res["links"]
        self._iters_     = res["iters"]

        if res["n_clusters"] != cur_state["n_clusters"]:
            warnings.warn("The number of clusters detected (%d) is "
                          "different than the requested one (%d)." % (
                            res["n_clusters"],
                            cur_state["n_clusters"]))
        self.n_clusters_ = res["n_clusters"]

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

            #self.is_noise_    = (self.labels_[0, :] < 0)

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

        if cur_state["compute_full_tree"]:
            assert cur_state["exact"] and cur_state["M"] == 1
            Z = internal.get_linkage_matrix(
                self._links_,
                self._mst_dist_,
                self._mst_ind_)
            self.children_    = Z["children"]
            self.distances_   = Z["distances"]
            self.counts_      = Z["counts"]

        return cur_state


    def _check_params(self, cur_state=None):
        if cur_state is None:
            cur_state = dict()

        cur_state["M"] = int(self.M)
        if cur_state["M"] < 1:
            raise ValueError("`M` must be > 0.")

        cur_state["exact"]             = bool(self.exact)

        cur_state["compute_full_tree"] = bool(self.compute_full_tree)
        if cur_state["compute_full_tree"] and \
                not (cur_state["M"] == 1 and cur_state["exact"]):
            cur_state["compute_full_tree"] = False
            warnings.warn("`compute_full_tree` is only available when `M` = 1 "
                          "and `exact` is True")

        cur_state["compute_all_cuts"]  = bool(self.compute_all_cuts)
        cur_state["cast_float32"]      = bool(self.cast_float32)
        cur_state["verbose"]           = bool(self.verbose)

        cur_state["n_clusters"] = int(self.n_clusters)
        if cur_state["n_clusters"] < 0:
            raise ValueError("n_clusters must be >= 0")

        _postprocess_options = ("boundary", "none", "all")
        cur_state["postprocess"] = str(self.postprocess).lower()
        if cur_state["postprocess"] not in _postprocess_options:
            raise ValueError("`postprocess` should be one of %s" % repr(_postprocess_options))

        cur_state["affinity"] = str(self.affinity).lower()
        if cur_state["affinity"] in ["euclidean", "lp:p=2"]:
            cur_state["affinity"] = "l2"
        elif cur_state["affinity"] in ["euclidean_sparse"]:
            cur_state["affinity"] = "l2_sparse"
        elif cur_state["affinity"] in ["manhattan", "cityblock", "lp:p=1"]:
            cur_state["affinity"] = "l1"
        elif cur_state["affinity"] in ["manhattan_sparse", "cityblock_sparse"]:
            cur_state["affinity"] = "l1_sparse"
        elif cur_state["affinity"] in ["chebyshev", "maximum", "lp:p=inf"]:
            cur_state["affinity"] = "linf"
        elif cur_state["affinity"] in ["chebyshev_sparse", "maximum_sparse"]:
            cur_state["affinity"] = "linf_sparse"
        elif cur_state["affinity"] in ["cosine"]:
            cur_state["affinity"] = "cosinesimil"
        elif cur_state["affinity"] in ["cosine_sparse"]:
            cur_state["affinity"] = "cosinesimil_sparse"
        elif cur_state["affinity"] in ["cosine_sparse_fast"]:
            cur_state["affinity"] = "cosinesimil_sparse_fast"

        _affinity_exact_options = (
            "l2", "l1", "cosinesimil", "precomputed")
        if cur_state["exact"] and cur_state["affinity"] not in _affinity_exact_options:
            raise ValueError("`affinity` should be one of %s" % repr(_affinity_exact_options))

        if type(self.mlpack_enabled) is str:
            cur_state["mlpack_enabled"] = str(self.mlpack_enabled).lower()
            if cur_state["mlpack_enabled"] != "auto":
                raise ValueError("`mlpack_enabled` must be one of: 'auto', True, False.")
        else:
            cur_state["mlpack_enabled"] = bool(self.mlpack_enabled)

        cur_state["mlpack_leaf_size"] = int(self.mlpack_leaf_size)  # mlpack will check this


        cur_state["nmslib_n_neighbors"] = int(self.nmslib_n_neighbors)

        if type(self.nmslib_params_init) is not dict:
            raise ValueError("`nmslib_params_init` must be a `dict`.")
        cur_state["nmslib_params_init"] = self.nmslib_params_init.copy()

        if type(self.nmslib_params_index) is not dict:
            raise ValueError("`nmslib_params_index` must be a `dict`.")
        cur_state["nmslib_params_index"] = self.nmslib_params_index.copy()

        if type(self.nmslib_params_query) is not dict:
            raise ValueError("`nmslib_params_query` must be a `dict`.")
        cur_state["nmslib_params_query"] = self.nmslib_params_query.copy()

        # this is more like an inherent dimensionality for GIc
        cur_state["n_features"] = self.n_features   # users can set this manually
        if cur_state["n_features"] is not None:      # only GIc needs this
            cur_state["n_features"] = max(1.0, float(cur_state["n_features"]))
        else:
            cur_state["n_features"] = -1

        return cur_state


    def _get_mst_exact(self, X, cur_state):
        if cur_state["affinity"] == "precomputed":
            X = X.reshape(X.shape[0], -1)
            if X.shape[1] not in [1, X.shape[0]]:
                raise ValueError(
                    "`X` must be distance vector "
                    "or a square-form distance matrix, "
                    "see `scipy.spatial.distance.pdist` or "
                    "`scipy.spatial.distance.squareform`.")
            if X.shape[1] == 1:
                # from a very advanced and sophisticated quadratic equation:
                n_samples = int(round((math.sqrt(1.0+8.0*X.shape[0])+1.0)/2.0))
                assert n_samples*(n_samples-1)//2 == X.shape[0]
            else:
                n_samples  = X.shape[0]
        else:
            if cur_state["cast_float32"]:
                if scipy.sparse.isspmatrix(X):
                    raise ValueError("Sparse matrices are (currently) only "
                                     "supported when `exact` is False")
                X = np.array(X, dtype=np.float32, order="C", copy=False, ndmin=2)

            n_samples  = X.shape[0]
            if cur_state["n_features"] < 0:
                cur_state["n_features"] = X.shape[1]

        if cur_state["mlpack_enabled"] == "auto":
            cur_state["mlpack_enabled"] = mlpack is not None and \
                    cur_state["affinity"] == "l2" and \
                    X.shape[1] <= 6 and \
                    cur_state["M"] == 1

        if cur_state["mlpack_enabled"]:
            if mlpack is None:
                raise ValueError("Package `mlpack` is not available.")
            elif cur_state["affinity"] != "l2":
                raise ValueError("`mlpack` can only be used with `affinity` = 'l2'.")
            elif cur_state["M"] != 1:
                raise ValueError("`mlpack` can only be used with `M` = 1.")

        mst_dist = None
        mst_ind  = None
        nn_dist  = None
        nn_ind   = None
        d_core   = None

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

        if cur_state["mlpack_enabled"]:
            assert cur_state["M"] == 1
            assert cur_state["affinity"] == "l2"

            if mst_dist is None or mst_ind is None:
                _res = mlpack.emst(
                    input=X,
                    leaf_size=cur_state["mlpack_leaf_size"],
                    naive=False,
                    copy_all_inputs=False,
                    verbose=cur_state["verbose"])["output"]
                mst_dist = _res[:,  2].astype(X.dtype, order="C")
                mst_ind  = _res[:, :2].astype(np.intp, order="C")
        else:
            if cur_state["M"] >= 2:  # else d_core   = None
                # Genie+HDBSCAN --- determine d_core
                # TODO: mlpack for k-nns?

                if cur_state["M"]-1 >= X.shape[0]:
                    raise ValueError("`M` is too large")

                if nn_dist is None or nn_ind is None:
                    nn_dist, nn_ind = internal.knn_from_distance(
                        X,  # if not c_contiguous, raises an error
                        k=cur_state["M"]-1,
                        metric=cur_state["affinity"],  # supports "precomputed"
                        verbose=cur_state["verbose"])

                d_core = internal.get_d_core(nn_dist, nn_ind, cur_state["M"])

            # Use Prim's algorithm to determine the MST
            # w.r.t. the distances computed on the fly
            if mst_dist is None or mst_ind is None:
                mst_dist, mst_ind = internal.mst_from_distance(
                    X,  # if not c_contiguous, raises an error
                    metric=cur_state["affinity"],
                    d_core=d_core,
                    verbose=cur_state["verbose"])

        self.n_samples_   = n_samples
        self._mst_dist_   = mst_dist
        self._mst_ind_    = mst_ind
        self._nn_dist_    = nn_dist
        self._nn_ind_     = nn_ind
        self._d_core_     = d_core

        return cur_state


    def _get_mst_approx(self, X, cur_state):
        #if nmslib is None:
        #    raise ValueError("Package `nmslib` is not available.")

        if cur_state["affinity"] == "precomputed":
            raise ValueError(
                "`affinity` of 'precomputed' can only be used "
                "with `exact` = True.")

        if cur_state["cast_float32"]:
            if cur_state["affinity"] in ["leven", "normleven", "jaccard_sparse",
                                         "bit_jaccard", "bit_hamming"]:
                raise ValueError("`cast_float32` cannot be used with this `affinity`.")

            if scipy.sparse.isspmatrix(X):
                X = scipy.sparse.csr_matrix(X, dtype=np.float32, copy=False)
            else:
                X = np.array(X, dtype=np.float32, order="C", copy=False, ndmin=2)

        n_samples  = np.shape(X)[0]

        if cur_state["n_features"] < 0 and np.ndim(X) >= 2:
            cur_state["n_features"] = np.shape(X)[1]

        if "indexThreadQty" in cur_state["nmslib_params_index"]:
            warnings.warn("Set `indexThreadQty` via the OMP_NUM_THREADS "
                          "environment variable.")

        if os.getenv("OMP_NUM_THREADS"):
            n_threads = max(1, int(os.getenv("OMP_NUM_THREADS")))
            cur_state["nmslib_params_index"]["indexThreadQty"] = n_threads
        else:
            n_threads = 0

        if "space" in cur_state["nmslib_params_init"]:
            warnings.warn("Set `space` via the `affinity` parameter.")
        cur_state["nmslib_params_init"]["space"] = cur_state["affinity"]

        if "data_type" not in cur_state["nmslib_params_init"]:
            # nmslib.DataType.DENSE_VECTOR|OBJECT_AS_STRING|SPARSE_VECTOR
            if scipy.sparse.isspmatrix(X):
                data_type = nmslib.DataType.SPARSE_VECTOR
            elif np.ndim(X) == 2:
                data_type = nmslib.DataType.DENSE_VECTOR
            else:
                data_type = nmslib.DataType.OBJECT_AS_STRING
            cur_state["nmslib_params_init"]["data_type"] = data_type

        if "dtype" not in cur_state["nmslib_params_init"]:
            # nmslib.DistType.FLOAT|INT use FLOAT except for `leven`
            cur_state["nmslib_params_init"]["dtype"] = \
                nmslib.DistType.FLOAT if cur_state["affinity"] != "leven" else \
                nmslib.DistType.INT

        cur_state["nmslib_n_neighbors"] = min(
            n_samples-1,
            max(1, cur_state["nmslib_n_neighbors"]))

        if cur_state["nmslib_n_neighbors"] < cur_state["M"]-1:
            raise ValueError("Increase `nmslib_n_neighbors` or decrease `M`.")

        mst_dist = None
        mst_ind  = None
        nn_dist  = None
        nn_ind   = None
        d_core   = None


        if self._last_state_ is not None and \
                cur_state["X"]                   == self._last_state_["X"] and \
                cur_state["affinity"]            == self._last_state_["affinity"] and \
                cur_state["exact"]               == self._last_state_["exact"] and \
                cur_state["nmslib_n_neighbors"]  == self._last_state_["nmslib_n_neighbors"] and \
                cur_state["nmslib_params_init"]  == self._last_state_["nmslib_params_init"] and \
                cur_state["nmslib_params_index"] == self._last_state_["nmslib_params_index"] and \
                cur_state["nmslib_params_query"] == self._last_state_["nmslib_params_query"]:

            if cur_state["M"] == self._last_state_["M"]:
                mst_dist = self._mst_dist_
                mst_ind  = self._mst_ind_
                nn_dist  = self._nn_dist_
                nn_ind   = self._nn_ind_
            elif cur_state["M"] < self._last_state_["M"]:
                nn_dist  = self._nn_dist_
                nn_ind   = self._nn_ind_

        if nn_dist is None or nn_ind is None:
            index = nmslib.init(**cur_state["nmslib_params_init"])
            index.addDataPointBatch(X)
            index.createIndex(
                cur_state["nmslib_params_index"],
                print_progress=cur_state["verbose"])
            index.setQueryTimeParams(cur_state["nmslib_params_query"])
            nns = index.knnQueryBatch(
                X,
                k=cur_state["nmslib_n_neighbors"] + 1,  # will return self as well
                num_threads=n_threads)
            index = None  # no longer needed
            nn_dist, nn_ind = internal.nn_list_to_matrix(nns, cur_state["nmslib_n_neighbors"] + 1)
            nns = None  # no longer needed

        if cur_state["M"] > 1:
            d_core = internal.get_d_core(nn_dist, nn_ind, cur_state["M"])


        if mst_dist is None or mst_ind is None:
            mst_dist, mst_ind = internal.mst_from_nn(
                nn_dist,
                nn_ind,
                d_core,
                stop_disconnected=False,
                verbose=cur_state["verbose"])
            # We can have a forest here...

        self.n_samples_   = n_samples
        self._mst_dist_   = mst_dist
        self._mst_ind_    = mst_ind
        self._nn_dist_    = nn_dist
        self._nn_ind_     = nn_ind
        self._d_core_     = d_core

        return cur_state


    def _get_mst(self, X, cur_state):
        cur_state["X"] = id(X)

        if cur_state["verbose"]:
            print("[genieclust] Preprocessing data.", file=sys.stderr)

        if cur_state["exact"]:
            cur_state = self._get_mst_exact(X, cur_state)
        else:
            cur_state = self._get_mst_approx(X, cur_state)

        # this might be an "intrinsic" dimensionality:
        self.n_features_  = cur_state["n_features"]
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
        Number of clusters to detect.

        If `M` > 1 and `postprocess` is not ``"all"``, `n_clusters` = 1
        can act as a noise point/outlier detector.
        The approximate method (see parameter `exact`) can sometimes
        fail to detect the coarsest-grained partitions; in such a case,
        more clusters might be returned (with a warning).

    gini_threshold : float
        Threshold for the Genie correction in [0,1].

        The Gini index is used to quantify the inequality of the cluster
        size distribution. Low thresholds highly penalise the formation
        of small clusters. Threshold of 1.0 disables the correction
        and for `M` = 1 makes the method be equivalent to the single
        linkage algorithm.

        The algorithm tends to be *stable* with respect to small changes
        to the threshold — they do not tend to affect the output clustering.
        Usually, thresholds of 0.1, 0.3, 0.5, and 0.7 are worth giving a try.

    M : int
        Smoothing factor for the mutual reachability distance [6]_.

        `M` = 1 gives the original Genie algorithm  [1]_ (with no noise point
        detection) with respect to the chosen affinity as-is. Note that
        for `M > 1` we need additionally :math:`O(M n)` working memory
        for storing of points' nearest neighbours.

    affinity : str
        Metric used to compute the linkage.

        *   For `exact` = ``True``:

            One of:
            ``"l2"`` (synonym: ``"euclidean"``),
            ``"l1"`` (synonym: ``"manhattan"``, ``"cityblock"``),
            ``"cosinesimil"`` (synonym: ``"cosine"``), or
            ``"precomputed"``.

            In the latter case, the `X` argument to the `fit` method
            must be a distance vector or a square-form distance matrix, see `scipy.spatial.distance.pdist`.

        *   For `exact` = ``False``:

            Any dissimilarity supported by `nmslib`, see [5]_ and
            https://github.com/nmslib/nmslib/blob/master/manual/spaces.md,
            for instance:
            ``"l2"``,
            ``"l2_sparse"``,
            ``"l1"``,
            ``"l1_sparse"``,
            ``"linf"``,
            ``"linf_sparse"``,
            ``"cosinesimil"``,
            ``"cosinesimil_sparse"``,
            ``"negdotprod"``,
            ``"negdotprod_sparse"``,
            ``"angulardist"``,
            ``"angulardist_sparse"``,
            ``"leven"``,
            ``"normleven"``,
            ``"jaccard_sparse"``,
            ``"bit_jaccard"``,
            ``"bit_hamming"``.

    exact : bool
        Whether to compute the minimum spanning tree exactly or rather
        estimate it based on an approximate near-neighbour graph.

        The exact method has time complexity of :math:`O(d n^2)` [2]_
        (however, see `mlpack_enabled`) but only needs :math:`O(n)` memory.

        If `exact` is ``False``, the minimum spanning tree is approximated
        based on an approximate :math:`k`\\ -nearest neighbours graph found by
        `nmslib` [5]_. This is typically very fast but requires
        :math:`O(k n)` memory.

    compute_full_tree : bool
        Whether to determine the whole cluster hierarchy and the
        linkage matrix.

        Only available if `M` = 1 and `exact` is ``True``.
        Enables plotting of dendrograms or cutting
        of the hierarchy at an arbitrary level, see the
        `children_`, `distances_`, `counts_` attributes.

    compute_all_cuts : bool
        Whether to compute the requested `n_clusters`-partition and all
        the coarser-grained ones.

        If ``True``, then the `labels_` attribute will be a matrix, see below.

    postprocess : {"boundary", "none", "all"}
        Controls the treatment of noise points after the clusters are
        identified.

        In effect only if `M` > 1. Each leaf in the minimum spanning tree
        is treated as a *noise* point. We call it a *boundary point*
        if it is amongst its adjacent vertex's `M` - 1 nearest neighbours.
        By default, only boundary points are merged with their nearest
        *core* points.

        To force a classical `n_clusters`-partition
        of a data set (with no notion of noise), choose ``"all"``.
        Furthermore, ``"none"`` leaves all leaves, i.e., noise points
        (including the boundary ones) as-is.


    cast_float32 : bool
        Whether casting of data type to ``float32`` is to be performed.

        If `exact` is ``True``, it decreases the run-time ca. 2 times
        at a cost of greater memory use. Otherwise, note that `nmslib`
        *requires* ``float32`` data anyway when using dense or sparse
        numeric matrix inputs.

        By setting `cast_float32` to ``False`` a user assures themself
        that the inputs are of acceptable form.

    mlpack_enabled : "auto" or bool
        Whether `mlpack.emst` should be used for computing the Euclidean
        minimum spanning tree instead of the Jarník-Prim algorithm
        when `exact` is ``True``.

        Often fast for very low-dimensional spaces. As the name suggests,
        only `affinity` of ``'l2'`` is supported (and `M` = 1).
        By default, we rely on `mlpack` if it is installed and
        `n_features` <= 6.

    mlpack_leaf_size : int
        Leaf size in the kd-tree when `mlpack.emst` is used.

        According to the `mlpack` manual, leaves of size 1 give the best
        performance at the cost of greater memory use.

    nmslib_n_neighbors : int
        The number of approximate nearest neighbours used to estimate
        the minimum spanning tree when when `exact` is ``False``.

        If the number of nearest neighbours is too small, the nearest
        neighbour graph might be disconnected and the number of obtained
        clusters might be greater than the requested one.

        `nmslib_n_neighbors` must not be less than `M` - 1.

    nmslib_params_init : dict
        A dictionary of parameters to be passed to `nmslib.init`
        when `exact` is ``False``.

        See https://github.com/nmslib/nmslib/blob/master/manual/methods.md,
        https://github.com/nmslib/nmslib/blob/master/manual/spaces.md
        and https://nmslib.github.io/nmslib/
        for more details. The `space`, `data_type`, and `dtype` parameters
        will be set based on the chosen `affinity` and the input `X`.

    nmslib_params_index : dict
        A dictionary of parameters to be passed to `index.createIndex`,
        where `index` is the object constructed with `nmslib.init`.

        The `indexThreadQty` parameter will be set based on the
        ``OMP_NUM_THREADS`` environment variable.

    nmslib_params_query : dict
        A dictionary of parameters to be passed to `index.setQueryTimeParams`,
        where `index` is the object constructed with `nmslib.init`.

    verbose : bool
        Whether to print diagnostic messages and progress information
        on ``stderr``.


    Attributes
    ----------

    labels_ : ndarray
        Detected cluster labels.

        If `compute_all_cuts` is ``False`` (the default),
        this is an integer vector such that ``labels_[i]`` gives
        the cluster ID (between 0 and `n_clusters_` - 1) of the i-th object.
        If `M` > 1, noise points are labelled -1 (unless taken care of in the
        postprocessing stage).

        Otherwise, i.e., if `compute_all_cuts` is ``True``,
        all partitions of cardinality down to `n_clusters`
        are determined; ``labels_[j,i]`` denotes the cluster ID of the i-th
        point in a j-partition. We assume that both the 0- and 1- partition
        distinguishes only between noise- and non-noise points,
        however, no postprocessing is conducted on the 0-partition
        (there might be points with labels of -1 even if `postprocess`
        is ``"all"``).

        Note that the approximate method (`exact` of ``False``) might fail
        to determine the fine-grained clusters (if the approximate
        neighbour graph is disconnected) - the actual number of clusters
        detected can be larger.

    n_clusters_ : int
        The actual number of clusters detected by the algorithm.

        As we argued above, the approximate method might sometimes yield
        a more fine-grained partition than the requested one (with a warning).
        Moreover, there might be too many noise points in the dataset.

    n_samples_ : int
        The number of points in the fitted dataset.

    n_features_ : int
        The number of features in the fitted dataset.

        If the information is not available, it is be set to -1.

    children_ : None or ndarray
        If `compute_full_tree` is ``True``, this is a matrix whose
        i-th row provides the information on the clusters merged in
        the i-th iteration. See the description of ``Z[:,0]`` and ``Z[:,1]``
        in `scipy.cluster.hierarchy.linkage`. Together with `distances_` and
        `counts_`, this constitutes the linkage matrix that can be used for
        plotting the dendrogram.

    distances_ : None or ndarray
        If `compute_full_tree` is ``True``, this is a vector that gives
        the distance between two clusters merged in each iteration,
        see the description of ``Z[:,2]`` in `scipy.cluster.hierarchy.linkage`.

        As the original Genie algorithm does not guarantee that that distances
        are ordered increasingly (there are other hierarchical
        clustering linkages that violate the ultrametricity property as well),
        these are corrected by applying
        ``distances_ = genieclust.tools.cummin(distances_[::-1])[::-1]``.

    counts_ : None or ndarray
        If `compute_full_tree` is ``True``, this is a vector giving
        the number of elements in a cluster created in each iteration.
        See the description of ``Z[:,3]`` in `scipy.cluster.hierarchy.linkage`.



    Notes
    -----

    **Genie** is a robust and outlier resistant
    hierarchical clustering algorithm [1]_, originally published
    as an R package ``genie``. This new implementation is, amongst others,
    much faster and now features optional smoothing and noise
    point detection (if `M` > 1).

    Genie is based on a minimum spanning tree (MST) of the
    pairwise distance graph of a given point set.
    Just like the single linkage, it consumes the edges
    of the MST in increasing order of weights. However, it prevents
    the formation of clusters of highly imbalanced sizes; once the Gini index
    of the cluster size distribution raises above an assumed threshold,
    a forced merge of a point group of the smallest size is performed.
    Its appealing simplicity goes hand in hand with its usability;
    Genie often outperforms other clustering approaches on benchmark data.

    The Genie algorithm itself has :math:`O(n \\sqrt{n})` time
    and :math:`O(n)` memory complexity provided that a minimum spanning
    tree of the pairwise distance graph is given.
    Generally, our parallelised implementation of a Jarník (Prim/Dijkstra)-like
    method [2]_ will be called to compute an MST, which takes :math:`O(d n^2)` time.
    However, `mlpack` [3]_ provides a very fast
    alternative in the case of Euclidean spaces of (very) low dimensionality
    and `M` = 1, see [4]_ and the `mlpack_enabled` parameter.
    Moreover, in the approximate method (`exact` = ``False``) we apply
    the Kruskal algorithm on the near-neighbour graph determined
    by `nmslib` [5]_. Albeit this only gives *some* sort of a spanning *forest*,
    such a data structure turns out to be very suitable for our clustering task
    (note that the set of connected components will determine the top
    level of the identified cluster hierarchy).

    The Genie correction together with the smoothing factor `M` > 1
    gives a robustified version of the HDBSCAN\\* [6]_ algorithm that,
    contrary to its predecessor, is able to detect a *predefined* number of
    clusters. Hence, it is independent of the *DBSCAN*'s somewhat magical
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
    or left marked as "noise" observations.


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

    .. [2] Olson C.F., Parallel algorithms for hierarchical clustering,
        *Parallel Computing* 21(8), 1995, 1313-1325.
        doi:10.1016/0167-8191(95)00017-I.

    .. [3]
        Curtin R.R., Edel M., Lozhnikov M., Mentekidis Y., Ghaisas S., Zhang S.,
        mlpack 3: A fast, flexible machine learning library,
        *Journal of Open Source Software* 3(26), 726, 2018.
        doi:10.21105/joss.00726.

    .. [4]
        March W.B., Ram P., Gray A.G.,
        Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis, and Applications,
        *Proc. ACM SIGKDD'10*, 2010, 603-611.

    .. [5]
        Naidan B., Boytsov L., Malkov Y.,  Novak D.,
        *Non-metric space library (NMSLIB) manual*, version 2.0, 2019.
        https://github.com/nmslib/nmslib/blob/master/manual/latex/manual.pdf.

    .. [6]
        Campello R., Moulavi D., Zimek A., Sander J.,
        Hierarchical density estimates for data clustering, visualization,
        and outlier detection,
        *ACM Transactions on Knowledge Discovery from Data* 10(1),
        2015, 5:1–5:51. doi:10.1145/2733381.

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
            mlpack_leaf_size=1,
            nmslib_n_neighbors=64,
            nmslib_params_init=dict(method="hnsw"),
            nmslib_params_index=dict(post=2),
            nmslib_params_query=dict(),
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
            mlpack_leaf_size=mlpack_leaf_size,
            nmslib_n_neighbors=nmslib_n_neighbors,
            nmslib_params_init=nmslib_params_init,
            nmslib_params_index=nmslib_params_index,
            nmslib_params_query=nmslib_params_query,
            verbose=verbose)

        self.gini_threshold = gini_threshold
        self._new_merge = False  # experimental, likely to be removed (#51)

        self._check_params()




    def _check_params(self, cur_state=None):
        cur_state = super()._check_params(cur_state)

        cur_state["gini_threshold"] = float(self.gini_threshold)
        if not (0.0 <= cur_state["gini_threshold"] <= 1.0):
            raise ValueError("`gini_threshold` not in [0,1].")

        cur_state["new_merge"] = bool(self._new_merge)  # experimental (#51)

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


        Notes
        -----

        Refer to the `labels_` and `n_clusters_` attributes for the result.


        Acceptable `X` types depend whether we use the exact or the approximate
        method.

        *   `X` when `exact` = ``True``.

            For `affinity` of ``"precomputed"``, `X` should either
            be a distance vector of length ``n_samples*(n_samples-1)/2``
            (see `scipy.spatial.distance.pdist`) or a square distance matrix
            of shape ``(n_samples, n_samples)``
            (see `scipy.spatial.distance.squareform`).

            Otherwise, `X` should be real-valued matrix
            (dense ``numpy.ndarray``, or an object coercible to)
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

        *   `X` when `exact` = ``False``.

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
            print("[genieclust] Determining clusters with Genie++.", file=sys.stderr)

        # apply the Genie++ algorithm (the fast part):
        res = internal.genie_from_mst(
            self._mst_dist_,
            self._mst_ind_,
            n_clusters=cur_state["n_clusters"],
            gini_threshold=cur_state["gini_threshold"],
            noise_leaves=(cur_state["M"] > 1),
            compute_full_tree=cur_state["compute_full_tree"],
            compute_all_cuts=cur_state["compute_all_cuts"],
            new_merge=cur_state["new_merge"])

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

    gini_thresholds : array_like
        A list of Gini index thresholds between 0 and 1.

        The GIc algorithm optimises the information criterion in
        an agglomerative way, starting from the intersection of
        the clusterings returned by
        ``Genie(n_clusters=n_clusters+add_clusters, gini_threshold=gini_thresholds[i])``,
        for all ``i`` from ``0`` to ``len(gini_thresholds)-1``.

    M : int
        See `genieclust.Genie`.

    affinity : str
        See `genieclust.Genie`.

    exact : bool, default=True
        See `genieclust.Genie`.

    compute_full_tree : bool
        See `genieclust.Genie`.

    compute_all_cuts : bool
        See `genieclust.Genie`.

        Note that if `compute_all_cuts` is ``True``, then the i-th cut
        in the hierarchy behaves as if `add_clusters` was equal to
        ``n_clusters-i``. In other words, the returned cuts might be different
        from those obtained by multiple calls to GIc, each time with
        different `n_clusters` and constant `add_clusters` requested.

    postprocess : {"boundary", "none", "all"}
        See `genieclust.Genie`.

    cast_float32 : bool
        See `genieclust.Genie`.

    mlpack_enabled : "auto" or bool
        See `genieclust.Genie`.

    mlpack_leaf_size : int
        See `genieclust.Genie`.

    nmslib_n_neighbors : int
        See `genieclust.Genie`.

    nmslib_params_init : dict
        See `genieclust.Genie`.

    nmslib_params_index : dict
        See `genieclust.Genie`.

    nmslib_params_query : dict
        See `genieclust.Genie`.

    add_clusters : int
        Number of additional clusters to work with internally.

    n_features : float or None
        Dataset's (intrinsic) dimensionality.

        If ``None``, it will be set based on the shape of the input matrix.
        Yet, `affinity` of ``"precomputed"`` needs this to be set manually.

    verbose : bool
        See `genieclust.Genie`.


    Attributes
    ----------

    See class `genieclust.Genie`.



    See also
    --------

    genieclust.Genie



    Notes
    -----

    GIc (Genie+Information Criterion) is an Information-Theoretic
    Hierarchical Clustering Algorithm.
    It was proposed by Anna Cena in [1]_ and had been inspired
    by Mueller's (et al.) ITM [2]_ and Gagolewski's (et al.) Genie [3]_.

    GIc computes an `n_clusters`-partition
    based on a pre-computed minimum spanning tree. Clusters are merged
    so as to maximise (heuristically) the information
    criterion discussed in [2]_.

    GIc uses a bottom-up, agglomerative approach (as opposed to the ITM,
    which follows a divisive scheme). It greedily selects for merging
    a pair of clusters that maximises the information criterion [2]_.
    By default, the initial partition is determined by considering
    the intersection of the partitions found by multiple runs of
    the Genie++ method with thresholds [0.1, 0.3, 0.5, 0.7], which
    is a sensible choice for most clustering activities. Hence, contrary
    to the Genie method, we can say that GIc as virtually parameter-less.


    :Environment variables:
        OMP_NUM_THREADS
            See `genieclust.Genie`.


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
            cast_float32=True,
            mlpack_enabled="auto",
            mlpack_leaf_size=1,
            nmslib_n_neighbors=64,
            nmslib_params_init=dict(method="hnsw"),
            nmslib_params_index=dict(post=2),
            nmslib_params_query=dict(),
            add_clusters=0,
            n_features=None,
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
            mlpack_leaf_size=mlpack_leaf_size,
            nmslib_n_neighbors=nmslib_n_neighbors,
            nmslib_params_init=nmslib_params_init,
            nmslib_params_index=nmslib_params_index,
            nmslib_params_query=nmslib_params_query,
            verbose=verbose)

        self.gini_thresholds = gini_thresholds
        self.n_features      = n_features
        self.add_clusters    = add_clusters

        self._check_params()



    def _check_params(self, cur_state=None):
        cur_state = super()._check_params(cur_state)

        cur_state["add_clusters"] = int(self.add_clusters)
        if cur_state["add_clusters"] < 0:
            raise ValueError("`add_clusters` must be non-negative.")

        cur_state["gini_thresholds"] = np.array(self.gini_thresholds)
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
            raise ValueError("Please set the `n_features` attribute manually.")

        if cur_state["verbose"]:
            print("[genieclust] Determining clusters with GIc.", file=sys.stderr)

        # apply the Genie+Ic algorithm:
        res = internal.gic_from_mst(
            self._mst_dist_,
            self._mst_ind_,
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
