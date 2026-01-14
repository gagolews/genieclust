"""
The Genie Clustering Algorithm (with Extras)
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2018-2026, Marek Gagolewski <https://www.gagolewski.com>      #
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
import warnings
import deadwood
from . import internal

###############################################################################
###############################################################################
###############################################################################



class Genie(deadwood.MSTClusterer):
    """
    Genie: Fast and Robust Hierarchical Clustering with Noise Point Detection


    Parameters
    ----------

    n_clusters : int
        The number of clusters to detect.

        If *M > 0* and *postprocess* is not ``"all"``, setting
        *n_clusters = 1* makes the algorithm behave as an outlier detector.

    gini_threshold : float in [0,1]
        The threshold for the Genie correction.

        The Gini index is used to quantify the inequality of the cluster
        size distribution. Low thresholds highly penalise the formation
        of small clusters. Threshold of 1.0 disables the correction:
        in such a case, if *M = 0*, then the method is equivalent to the single
        linkage algorithm.

        Empirically, the algorithm tends to be *stable* with respect to small
        changes to the threshold, as they usually do not affect the output
        clustering. Usually, thresholds of 0.1, 0.3, 0.5, and 0.7 are worth
        giving a try.

    M : int
        The smoothing factor for the mutual reachability distance [2]_.
        *M ≤ 1* indicates the original distance as given by
        the *metric* parameter; see :any:`deadwood.MSTClusterMixin`
        for more details.

        *M = 0* gives the original Genie algorithm [1]_
        (with no outlier detection) with respect to the chosen distance.

    metric : str
        The metric used to compute the linkage; see
        :any:`deadwood.MSTClusterMixin` for more details.
        Defaults to ``"l2"``, i.e., the Euclidean distance.

    compute_full_tree : bool
        Whether to determine the entire cluster hierarchy and the linkage matrix.

        Enables plotting dendrograms or cutting the cluster hierarchy at an
        arbitrary level; see the `children_`, `distances_`, `counts_` attributes.

    compute_all_cuts : bool
        Whether to compute the requested `n_clusters`-partition and all
        the coarser-grained ones.

        If ``True``, then the `labels_` attribute will be a matrix; see below.

    quitefastmst_params : dict
        Additional parameters to be passed to ``quitefastmst.mst_euclid``
        if ``metric`` is ``"l2"``

    verbose : bool
        Whether to print diagnostic messages and progress information
        onto ``stderr``.


    Attributes
    ----------

    labels_ : ndarray
        Detected cluster labels.

        If `compute_all_cuts` is ``False`` (the default),
        it is an integer vector such that ``labels_[i]`` gives
        the cluster ID (between 0 and `n_clusters_` - 1) of the `i`-th object.
        If *M > 0*, outliers are labelled ``-1`` (unless taken care
        of at the postprocessing stage).

        Otherwise, i.e., if `compute_all_cuts` is ``True``,
        all partitions of cardinality down to `n_clusters`
        are determined; ``labels_[j,i]`` denotes the cluster ID of the i-th
        point in a j-partition.  We assume that both the 0- and 1- partition
        distinguishes only between noise- and non-noise points,
        however, no postprocessing is conducted on the 0-partition
        (there might be points with labels of -1 even if `postprocess`
        is ``"all"``).

    n_clusters_ : int
        The actual number of clusters detected by the algorithm.
        It can be different from the requested one if there are too many
        noise points in the dataset.

    n_samples_ : int
        The number of points in the dataset.

    n_features_ : int
        The number of features in the dataset.

        If the information is not available, it will be set to ``-1``.

    children_ : None or ndarray
        If `compute_full_tree` is ``True``, this is a matrix whose
        i-th row provides the information on the clusters merged in
        the i-th iteration. See the description of ``Z[:,0]`` and ``Z[:,1]``
        in ``scipy.cluster.hierarchy.linkage``. Together with `distances_` and
        `counts_`, this constitutes the linkage matrix that can be used for
        plotting the dendrogram.

    distances_ : None or ndarray
        If `compute_full_tree` is ``True``, this is a vector that gives
        the distance between two clusters merged in each iteration,
        see the description of ``Z[:,2]`` in ``scipy.cluster.hierarchy.linkage``.

        The original Genie algorithm does not guarantee that distances
        are ordered increasingly (there are other hierarchical clustering
        linkages that violate the ultrametricity property too).
        Thus, we automatically apply the following correction:

        ``distances_ = genieclust.tools.cummin(distances_[::-1])[::-1]``.

    counts_ : None or ndarray
        If `compute_full_tree` is ``True``, this is a vector giving
        the number of elements in a cluster created in each iteration.
        See the description of ``Z[:,3]`` in ``scipy.cluster.hierarchy.linkage``.



    Notes
    -----

    **Genie** is a robust and outlier-resistant hierarchical clustering
    algorithm [1]_, originally published in the R package ``genie``. This new
    implementation is, amongst others, much faster and now features optional
    outlier detection (if *M > 0*).

    Genie is based on the minimum spanning tree (MST) of the
    pairwise distance graph of a given point set (refer to
    :any:`deadwood.MSTClusterMixin` and [3]_ for more details).
    Just like the single linkage, it consumes the edges
    of the MST in increasing order of weights. However, it prevents
    the formation of clusters of highly imbalanced sizes; once the Gini index
    of the cluster size distribution raises above an assumed threshold,
    a forced merge of a point group of the smallest size is undertaken.
    Its appealing simplicity goes hand in hand with its usability; Genie often
    outperforms other clustering approaches on benchmark data.

    The Genie algorithm itself has :math:`O(n \\sqrt{n})` time
    and :math:`O(n)` memory complexity provided that a minimum spanning
    tree of the pairwise distance graph is given.
    If the Euclidean distance is selected, then
    ``quitefastmst.mst_euclid`` is used to compute the MST;
    it is quite fast in low-dimensional spaces.
    Otherwise, an implementation of the Jarník (Prim/Dijkstra)-like
    :math:`O(n^2)`-time algorithm is called.

    The Genie algorithm together with the smoothing factor *M > 0*
    gives an alternative to the HDBSCAN\\* [2]_ algorithm that is able to
    detect a predefined number of clusters (see [4]_) without depending
    on DBSCAN\\*'s ``eps`` HDBSCAN\\*'s ``min_cluster_size`` parameters.


    :Environment variables:
        OMP_NUM_THREADS
            Controls the number of threads used when computing the minimum
            spanning tree.


    References
    ----------

    .. [1]
        Gagolewski M., Bartoszuk M., Cena A.,
        Genie: A new, fast, and outlier-resistant hierarchical
        clustering algorithm, *Information Sciences* 363, 2016, 8-23.
        doi:10.1016/j.ins.2016.05.003.

    .. [2]
        Campello R.J.G.B., Moulavi D., Sander J.,
        Density-based clustering based on hierarchical density estimates,
        *Lecture Notes in Computer Science* 7819, 2013, 160-172,
        doi:10.1007/978-3-642-37456-2_14.

    .. [3]
        Gagolewski M., Cena A., Bartoszuk M., Brzozowski L.,
        Clustering with minimum spanning trees: How good can it be?,
        *Journal of Classification* 42, 2025, 90-112,
        doi:10.1007/s00357-024-09483-1.

    .. [4]
        Gagolewski M., TODO, 2025

    """

    def __init__(
            self,
            n_clusters=2,
            *,
            gini_threshold=0.3,
            M=0,
            metric="l2",
            compute_full_tree=False,
            compute_all_cuts=False,
            #preprocess="auto",
            #postprocess="midliers",
            quitefastmst_params=dict(mutreach_ties="dcore_min", mutreach_leaves="reconnect_dcore_min"),
            verbose=False
        ):
        # # # # # # # # # # # #
        super().__init__(
            n_clusters=n_clusters,
            M=M,
            metric=metric,
            #preprocess=preprocess,
            #postprocess=postprocess,
            quitefastmst_params=quitefastmst_params,
            verbose=verbose
        )

        self.compute_full_tree   = compute_full_tree
        self.compute_all_cuts    = compute_all_cuts
        self.gini_threshold      = gini_threshold

        self.children_           = None
        self.distances_          = None
        self.counts_             = None
        self._links_             = None
        self._iters_             = None


    def _check_params(self, cur_state=None):
        cur_state = super()._check_params(cur_state)

        self.gini_threshold = float(self.gini_threshold)
        if not (0.0 <= self.gini_threshold <= 1.0):
            raise ValueError("gini_threshold not in [0,1].")

        cur_state["compute_full_tree"] = bool(self.compute_full_tree)

        cur_state["compute_all_cuts"]  = bool(self.compute_all_cuts)

        return cur_state


    def fit(self, X, y=None):
        """
        Perform cluster analysis of a dataset.


        Parameters
        ----------

        X : object
            Typically a matrix with ``n_samples`` rows and ``n_features``
            columns; see :any:`deadwood.MSTClusterMixin.fit_predict` for more
            details.

        y : None
            Ignored.


        Returns
        -------

        self : genieclust.Genie
            The object that the method was called on.


        Notes
        -----

        Refer to the `labels_` and `n_clusters_` attributes for the result.

        """
        cur_state = self._check_params()  # re-check, they might have changed

        cur_state = self._get_mst(X, cur_state)

        if cur_state["n_clusters"] >= self.n_samples_:
            raise ValueError("n_clusters must be < n_samples_")

        if cur_state["verbose"]:
            print("[genieclust] Determining clusters with Genie.", file=sys.stderr)

        # apply the Genie algorithm:
        res = internal.genie_from_mst(
            self._tree_w,
            self._tree_i,
            skip_nodes=np.zeros(0, dtype=bool),  # TODO
            n_clusters=cur_state["n_clusters"],
            gini_threshold=self.gini_threshold,
            #skip_leaves=(cur_state["preprocess"] == "leaves"),  TODO
            compute_full_tree=cur_state["compute_full_tree"],
            compute_all_cuts=cur_state["compute_all_cuts"]
        )

        ########################################################

        #cur_state = self._postprocess_outputs(res, cur_state)

        self.labels_     = res["labels"]
        self._links_     = res["links"]
        self._iters_     = res["iters"]

        if res["n_clusters"] != cur_state["n_clusters"]:
            warnings.warn("The number of clusters detected (%d) is "
                          "different from the requested one (%d)." % (
                            res["n_clusters"],
                            cur_state["n_clusters"]))

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

        self.n_clusters_ = res["n_clusters"]


        if reshaped:
            self.labels_.shape = (self.labels_.shape[1], )

        ########################################################

        if cur_state["compute_full_tree"]:
            # assert cur_state["exact"] and cur_state["M"] == 0
            Z = internal.get_linkage_matrix(
                self._links_,
                self._tree_w,
                self._tree_i
            )
            self.children_    = Z["children"]
            self.distances_   = Z["distances"]
            self.counts_      = Z["counts"]

        if cur_state["verbose"]:
            print("[genieclust] Done.", file=sys.stderr)

        return self




###############################################################################
###############################################################################
###############################################################################



class GIc(deadwood.MSTClusterer):
    """
    GIc (Genie+Information Criterion) clustering algorithm


    Parameters
    ----------

    n_clusters : int
        The number of clusters to detect; see :any:`genieclust.Genie`
        for more details.

    gini_thresholds : array_like
        A list of Gini's index thresholds between 0 and 1.

        The GIc algorithm optimises the information criterion in an
        agglomerative way, starting from the intersection of the clusterings
        returned by ``Genie(n_clusters=n_clusters+add_clusters, gini_threshold=gini_thresholds[i])``,
        for all ``i`` from ``0`` to ``len(gini_thresholds)-1``.

    metric : str
        The metric used to compute the linkage; see :any:`genieclust.Genie`
        for more details.

    compute_full_tree : bool
        Whether to determine the entire cluster hierarchy and the linkage
        matrix; see :any:`genieclust.Genie` for more details.

    compute_all_cuts : bool
        Whether to compute the requested *n_clusters*-partition and all
        the coarser-grained ones; see :any:`genieclust.Genie`
        for more details.

        Note that if *compute_all_cuts* is ``True``, then the *i*-th cut
        in the hierarchy behaves as if *add_clusters* was equal to
        *n_clusters-i*. In other words, the returned cuts might be different
        from those obtained by multiple calls to GIc, each time with
        different *n_clusters* and constant *add_clusters* requested.

    add_clusters : int
        Number of additional clusters to work with internally.

    n_features : float or None
        The dataset's (intrinsic) dimensionality.

        If ``None``, it will be set based on the shape of the input matrix.
        Yet, *metric* of ``"precomputed"`` needs this to be set manually.

    quitefastmst_params : dict
        Additional parameters to be passed to ``quitefastmst.mst_euclid``
        if ``metric`` is ``"l2"``

    verbose : bool
        Whether to print diagnostic messages and progress information
        onto ``stderr``.


    Attributes
    ----------

    See :any:`genieclust.Genie`.



    See also
    --------

    genieclust.Genie

    quitefastmst.mst_euclid



    Notes
    -----

    GIc (Genie+Information Criterion) is an Information-Theoretic
    Clustering Algorithm. It was proposed by Anna Cena in [1]_. It was inspired
    by Mueller's (et al.) ITM [2]_ and Genie [3]_.

    GIc computes an *n_clusters*-partition based on a pre-computed minimum
    spanning tree (MST) of the pairwise distance graph of a given point set
    (refer to :any:`deadwood.MSTClusterMixin` and [4]_ for more details).
    Clusters are merged so as to maximise (heuristically)
    the information criterion discussed in [2]_.

    GIc uses a bottom-up, agglomerative approach (as opposed to the ITM,
    which follows a divisive scheme).  It greedily selects for merging
    a pair of clusters that maximises the information criterion.
    By default, the initial partition is determined by considering
    the intersection of the partitions found by multiple runs of
    the Genie method with thresholds [0.1, 0.3, 0.5, 0.7], which
    we observe to be a sensible choice for most clustering activities.
    Hence, contrary to the Genie method, we can say that GIc is virtually
    parameter-free. However, when run with different *n_clusters* parameter,
    it does not yield a hierarchy of nested partitions (unless some more
    laborious parameter tuning is applied).


    :Environment variables:
        OMP_NUM_THREADS
            Controls the number of threads used when computing the minimum
            spanning tree.


    References
    ----------

    .. [1]
        Cena, A., *Adaptive hierarchical clustering algorithms based on
        data aggregation methods*, PhD Thesis, Systems Research Institute,
        Polish Academy of Sciences 2018.

    .. [2]
        Mueller, A., Nowozin, S., Lampert, C.H., Information Theoretic
        Clustering using Minimum Spanning Trees, *DAGM-OAGM*, 2012.

    .. [3]
        Gagolewski, M., Bartoszuk, M., Cena, A.,
        Genie: A new, fast, and outlier-resistant hierarchical clustering
        algorithm, *Information Sciences* 363, 2016, 8-23.
        doi:10.1016/j.ins.2016.05.003.ing

    .. [4]
        Gagolewski, M., Cena, A., Bartoszuk, M., Brzozowski, L.,
        Clustering with minimum spanning trees: How good can it be?,
        *Journal of Classification* 42, 2025, 90-112,
        doi:10.1007/s00357-024-09483-1.

    """
    def __init__(
            self,
            n_clusters=2,
            *,
            gini_thresholds=[0.1, 0.3, 0.5, 0.7],
            metric="l2",
            compute_full_tree=False,
            compute_all_cuts=False,
            add_clusters=0,
            n_features=None,
            quitefastmst_params=dict(mutreach_ties="dcore_min", mutreach_leaves="reconnect_dcore_min"),
            verbose=False
        ):
        super().__init__(
            n_clusters=n_clusters,
            M=0,
            metric=metric,
            quitefastmst_params=quitefastmst_params,
            verbose=verbose
        )

        self.compute_full_tree   = compute_full_tree
        self.compute_all_cuts    = compute_all_cuts
        self.gini_thresholds     = gini_thresholds
        self.n_features          = n_features
        self.add_clusters        = add_clusters

        self.children_           = None
        self.distances_          = None
        self.counts_             = None
        self._links_             = None
        self._iters_             = None


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

        cur_state["compute_full_tree"] = bool(self.compute_full_tree)

        cur_state["compute_all_cuts"]  = bool(self.compute_all_cuts)

        return cur_state


    def fit(self, X, y=None):
        """
        Perform cluster analysis of a dataset.


        Parameters
        ----------

        X : object
            Typically a matrix with ``n_samples`` rows and ``n_features``
            columns; see :any:`deadwood.MSTClusterMixin.fit_predict` for more
            details.

        y : None
            Ignored.


        Returns
        -------

        self : genieclust.GIc
            The object that the method was called on.


        Notes
        -----

        Refer to the `labels_` and `n_clusters_` attributes for the result.

        Note that for `metric` of ``"precomputed"``, the `n_features`
        parameter must be set explicitly.

        """
        cur_state = self._check_params()  # re-check, they might have changed

        cur_state = self._get_mst(X, cur_state)

        # this is more like an inherent dimensionality for GIc
        cur_state["n_features"] = self.n_features   # users can set this manually
        if cur_state["n_features"] is not None:     # only GIc needs this
            cur_state["n_features"] = max(1.0, self.n_features_))
        else:
            cur_state["n_features"] = -1.0

        if cur_state["n_features"] < 1.0:
            # this shouldn't happen in normal use
            raise ValueError("Please set the `n_features` attribute manually.")

        if cur_state["verbose"]:
            print("[genieclust] Determining clusters with GIc.", file=sys.stderr)

        # apply the Genie+Ic algorithm:
        res = internal.gic_from_mst(
            self._tree_w,
            self._tree_i,
            n_features=cur_state["n_features"],
            n_clusters=cur_state["n_clusters"],
            add_clusters=cur_state["add_clusters"],
            gini_thresholds=cur_state["gini_thresholds"],
            compute_full_tree=cur_state["compute_full_tree"],
            compute_all_cuts=cur_state["compute_all_cuts"]
        )

        ########################################################

        self.labels_     = res["labels"]
        self._links_     = res["links"]
        self._iters_     = res["iters"]

        if res["n_clusters"] != cur_state["n_clusters"]:
            warnings.warn("The number of clusters detected (%d) is "
                          "different from the requested one (%d)." % (
                            res["n_clusters"],
                            cur_state["n_clusters"]))

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

        self.n_clusters_ = res["n_clusters"]


        if reshaped:
            self.labels_.shape = (self.labels_.shape[1], )

        ########################################################

        if cur_state["compute_full_tree"]:
            # assert cur_state["exact"] and cur_state["M"] == 0
            Z = internal.get_linkage_matrix(
                self._links_,
                self._tree_w,
                self._tree_i
            )
            self.children_    = Z["children"]
            self.distances_   = Z["distances"]
            self.counts_      = Z["counts"]

        if cur_state["verbose"]:
            print("[genieclust] Done.", file=sys.stderr)

        return self
