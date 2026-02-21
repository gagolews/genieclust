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
from . import core

###############################################################################
###############################################################################
###############################################################################



class Genie(deadwood.MSTClusterer):
    """
    Genie: Fast and Robust Hierarchical Clustering


    Parameters
    ----------

    n_clusters : int
        The number of clusters to detect.

    gini_threshold : float in [0,1]
        The threshold for the Genie correction.

        The Gini index is used to quantify the inequality of the cluster
        size distribution. Low thresholds highly penalise the formation
        of small clusters. Threshold of 1.0 disables the correction:
        in such a case, if *M = 0*, then the method is equivalent to the single
        linkage algorithm.

        Empirically, the algorithm tends to be *stable* with respect to small
        changes to the threshold, as they usually do not affect the output
        clustering.  Usually, thresholds of 0.1, 0.3, 0.5, and 0.7 are worth
        giving a try.

    M : int
        The smoothing factor for the mutual reachability distance [2]_.
        `M = 0` and `M = 1` select the original distance as given by
        the `metric` parameter; see :any:`deadwood.MSTBase`

    metric : str, default='l2'
        The metric used to compute the linkage; see
        :any:`deadwood.MSTBase` for more details.
        Defaults to the Euclidean distance.

    coarser : bool, default=False
        Whether to compute the requested `n_clusters`-partition and all
        the coarser-grained ones.

        If ``True``, then the `labels_matrix_` attribute will additionally be
        determined; see below.

    quitefastmst_params : dict
        Additional parameters to be passed to ``quitefastmst.mst_euclid``
        if ``metric`` is ``"l2"``

    verbose : bool
        Whether to print diagnostic messages and progress information
        onto ``stderr``.


    Attributes
    ----------

    labels_ : ndarray, shape (n_samples_,)
        Detected cluster labels.

        An integer vector such that ``labels_[i]`` gives
        the cluster ID (between 0 and `n_clusters_` - 1) of the `i`-th object.

    labels_matrix_ : None or ndarray, shape (n_clusters_, n_samples_)
        Available if `coarser` is True.
        `labels_matrix[i,:]` represents an `i+1`-partition.

    n_clusters_ : int
        The actual number of clusters detected by the algorithm.

    n_samples_ : int
        The number of points in the dataset.

    n_features_ : int
        The number of features in the dataset.

        If the information is not available, it will be set to ``-1``.

    children_ : None or ndarray
        A matrix whose i-th row provides the information on the clusters merged
        in the i-th iteration. See the description of ``Z[:,0]`` and ``Z[:,1]``
        in ``scipy.cluster.hierarchy.linkage``. Together with `distances_` and
        `counts_`, this constitutes the linkage matrix that can be used for
        plotting the dendrogram.

    distances_ : None or ndarray
        A vector giving the distances between two clusters merged in each
        iteration, see the description of ``Z[:,2]`` in
        ``scipy.cluster.hierarchy.linkage``.

        The original Genie algorithm does not guarantee that distances
        are ordered increasingly (there are other hierarchical clustering
        linkages that violate the ultrametricity property too).
        Thus, we automatically apply the following correction:

        ``distances_ = genieclust.tools.cummin(distances_[::-1])[::-1]``.

    counts_ : None or ndarray
        A vector giving the number of elements in a cluster created in each
        iteration. See the description of ``Z[:,3]`` in
        ``scipy.cluster.hierarchy.linkage``.



    Notes
    -----

    *Genie* is a robust hierarchical clustering algorithm [1]_.
    Its original implementation was included in the R package ``genie``.
    This is its faster and more capable variant.

    The idea behind *Genie* is beautifully simple. First, make each individual
    point the only member of its own cluster. Then, keep merging pairs
    of the closest clusters, one after another. However, to prevent
    the formation of clusters of highly imbalanced sizes, a point group of
    the *smallest* size is sometimes combined with its nearest counterpart.
    Its appealing simplicity goes hand in hand with its usability; Genie often
    outperforms other clustering approaches on benchmark data.

    Genie is based on Euclidean minimum spanning trees (MST; refer to
    :any:`deadwood.MSTBase` and [3]_ for more details).  If the Euclidean
    distance is selected, then ``quitefastmst.mst_euclid`` is used to compute
    the MST;  it is quite fast in low-dimensional spaces.
    Otherwise, an implementation of the Jarník (Prim/Dijkstra)-like
    :math:`O(n^2)`-time algorithm is called.
    The Genie algorithm itself has :math:`O(n \\sqrt{n})` time
    and :math:`O(n)` memory complexity if an MST is already provided.

    As with all distance-based methods (this includes k-means and DBSCAN as
    well), applying data preprocessing and feature engineering techniques
    (e.g., feature scaling, feature selection, dimensionality reduction)
    might lead to more meaningful results.

    `genieclust` also allows clustering with respect to mutual reachability
    distances, enabling it to act as an alternative to *HDBSCAN\** [2]_
    that can identify any number of clusters or their entire hierarchy.
    When combined with the *deadwood* package, it can act as an outlier
    detector.


    :Environment variables:
        OMP_NUM_THREADS
            Controls the number of threads used when computing the minimum
            spanning tree.


    References
    ----------

    .. [1]
        M. Gagolewski, M. Bartoszuk, A. Cena,
        Genie: A new, fast, and outlier-resistant hierarchical
        clustering algorithm, *Information Sciences* 363, 2016, 8-23.
        https://doi.org/10.1016/j.ins.2016.05.003

    .. [2]
        R.J.G.B. Campello, D. Moulavi, J. Sander,
        Density-based clustering based on hierarchical density estimates,
        *Lecture Notes in Computer Science* 7819, 2013, 160-172,
        https://doi.org/10.1007/978-3-642-37456-2_14

    .. [3]
        M. Gagolewski, A. Cena, M. Bartoszuk, Ł. Brzozowski,
        Clustering with minimum spanning trees: How good can it be?,
        *Journal of Classification* 42, 2025, 90-112,
        https://doi.org/10.1007/s00357-024-09483-1
    """

    def __init__(
            self,
            n_clusters=2,
            *,
            gini_threshold=0.3,
            M=0,
            metric="l2",
            coarser=False,
            quitefastmst_params=None,  # TODO ?dict(mutreach_ties="dcore_min", mutreach_leaves="reconnect_dcore_min"),
            verbose=False
        ):
        # # # # # # # # # # # #
        super().__init__(
            n_clusters=n_clusters,
            M=M,
            metric=metric,
            quitefastmst_params=quitefastmst_params,
            verbose=verbose
        )

        self.coarser             = coarser
        self.gini_threshold      = gini_threshold

        self.children_           = None
        self.distances_          = None
        self.counts_             = None
        self._links_             = None
        self._iters_             = None
        self.labels_matrix_      = None


    def _check_params(self, cur_state=None):
        super()._check_params()

        self.gini_threshold = float(self.gini_threshold)
        if not (0.0 <= self.gini_threshold <= 1.0):
            raise ValueError("gini_threshold not in [0,1].")

        self.coarser  = bool(self.coarser)



    def fit(self, X, y=None):
        """
        Perform cluster analysis of a dataset.


        Parameters
        ----------

        X : object
            Typically a matrix or a data frame with ``n_samples`` rows
            and ``n_features`` columns;
            see :any:`deadwood.MSTBase.fit_predict` for more details.

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
        self.labels_        = None
        self.n_clusters_    = None
        self._cut_edges_    = None
        self.labels_matrix_ = None

        self._check_params()  # re-check, they might have changed
        self._get_mst(X)  # sets n_samples_, n_features_, _tree_d, _tree_i, _d_core, etc.

        if not (1 <= self.n_clusters < self.n_samples_):
            raise ValueError("n_clusters must be between 1 and n_samples_-1")

        if self.verbose:
            print("[genieclust] Determining clusters with Genie.", file=sys.stderr)

        # NOTE: if only n_clusters has changed since the last call,
        # NOTE: then we can simply fetch the requested partition via
        # NOTE: Cmst_cluster_sizes

        # apply the Genie algorithm:
        res = core.genie_from_mst(
            self._tree_d_,
            self._tree_i_,
            n_clusters=self.n_clusters,
            gini_threshold=self.gini_threshold,
            coarser=self.coarser
        )

        ########################################################

        self._links_ = res["links"]
        Z = core.get_linkage_matrix(
            self._links_,
            self._tree_d_,
            self._tree_i_
        )
        self.children_    = Z["children"]
        self.distances_   = Z["distances"]
        self.counts_      = Z["counts"]

        self._iters_      = res["iters"]
        self.n_clusters_  = res["n_clusters"]
        self._cut_edges_  = self._links_[self._links_ >= 0][:-self.n_clusters_:-1].copy()

        if self.n_clusters_ != self.n_clusters:
            warnings.warn("The number of clusters detected (%d) is "
                          "different from the requested one (%d)." % (
                            self.n_clusters_,
                            self.n_clusters))

        if self.coarser:
            self.labels_matrix_ = res["labels"]
            self.labels_ = self.labels_matrix_[-1,:]
        else:
            self.labels_ = res["labels"]

        ########################################################

        if self.verbose:
            print("[genieclust] Done.", file=sys.stderr)

        return self



###############################################################################
###############################################################################
###############################################################################



class GIc(deadwood.MSTClusterer):
    """
    GIc (Genie+Information Criterion) clustering algorithm

    [TESTING]


    Parameters
    ----------

    n_clusters : int
        The number of clusters to detect; see :any:`genieclust.Genie`
        for more details.

    gini_thresholds : array_like
        A list of Gini's index thresholds between 0 and 1.

        The GIc algorithm optimises the information criterion agglomeratively,
        starting from the intersection of the clusterings returned by
        ``Genie(n_clusters=n_clusters+add_clusters, gini_threshold=gini_thresholds[i])``,
        for all ``i`` from ``0`` to ``len(gini_thresholds)-1``.

    metric : str, default='l2'
        The metric used to compute the linkage; see :any:`genieclust.Genie`
        for more details.

    coarser : bool, default=False
        Whether to compute the requested `n_clusters`-partition and all
        the coarser-grained ones; see :any:`genieclust.Genie`
        for more details.

        Note that if `coarser` is ``True``, then the `i`-th cut
        in the hierarchy behaves as if `add_clusters` was equal to
        `n_clusters-i`. In other words, the returned cuts might be different
        from those obtained by multiple calls to GIc, each time with
        different `n_clusters` and constant `add_clusters` requested.

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
    Clustering Algorithm.  It was proposed by Anna Cena in [1]_.
    GIc was inspired by ITM [2]_ and Genie [3]_.

    GIc computes an `n_clusters`-partition based on a pre-computed minimum
    spanning tree (MST) of the pairwise distance graph of a given point set
    (refer to :any:`deadwood.MSTBase` and [4]_ for more details).
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
    parameter-free. However, when run with different `n_clusters` parameter,
    it does not yield a hierarchy of nested partitions (unless some more
    laborious parameter tuning is applied).


    :Environment variables:
        OMP_NUM_THREADS
            Controls the number of threads used when computing the minimum
            spanning tree.


    References
    ----------

    .. [1]
        A. Cena, *Adaptive hierarchical clustering algorithms based on
        data aggregation methods*, PhD Thesis, Systems Research Institute,
        Polish Academy of Sciences 2018

    .. [2]
        A. Mueller, S. Nowozin, C.H. Lampert, Information Theoretic
        Clustering using Minimum Spanning Trees, *DAGM-OAGM*, 2012

    .. [3]
        M. Gagolewski, M. Bartoszuk, A. Cena,
        Genie: A new, fast, and outlier-resistant hierarchical clustering
        algorithm, *Information Sciences* 363, 2016, 8-23.
        https://doi.org/10.1016/j.ins.2016.05.003

    .. [4]
        M. Gagolewski, A. Cena, M. Bartoszuk, Ł. Brzozowski,
        Clustering with minimum spanning trees: How good can it be?,
        *Journal of Classification* 42, 2025, 90-112,
        https://doi.org/10.1007/s00357-024-09483-1

    """
    def __init__(
            self,
            n_clusters=2,
            *,
            gini_thresholds=[0.1, 0.3, 0.5, 0.7],
            metric="l2",
            coarser=False,
            add_clusters=0,
            n_features=None,
            quitefastmst_params=None,  # TODO ?dict(mutreach_ties="dcore_min", mutreach_leaves="reconnect_dcore_min"),
            verbose=False
        ):
        super().__init__(
            n_clusters=n_clusters,
            M=0,
            metric=metric,
            quitefastmst_params=quitefastmst_params,
            verbose=verbose
        )

        self.coarser             = coarser
        self.gini_thresholds     = gini_thresholds
        self.n_features          = n_features
        self.add_clusters        = add_clusters

        self.children_           = None
        self.distances_          = None
        self.counts_             = None
        self._links_             = None
        self._iters_             = None
        self.labels_matrix_      = None


    def _check_params(self, cur_state=None):
        super()._check_params()

        self.add_clusters = int(self.add_clusters)
        if self.add_clusters < 0:
            raise ValueError("add_clusters must be non-negative.")

        self.gini_thresholds = np.array(self.gini_thresholds)
        for g in self.gini_thresholds:
            if not (0.0 <= g <= 1.0):
                raise ValueError("All gini_thresholds must be in [0,1].")

        self.coarser  = bool(self.coarser)



    def fit(self, X, y=None):
        """
        Perform cluster analysis of a dataset.


        Parameters
        ----------

        X : object
            Typically a matrix or a data frame with ``n_samples`` rows
            and ``n_features`` columns;
            see :any:`deadwood.MSTBase.fit_predict` for more details.

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
        self.labels_     = None
        self.n_clusters_ = None
        self.labels_matrix_ = None

        self._check_params()  # re-check, they might have changed
        self._get_mst(X)  # sets n_samples_, n_features_, _tree_d, _tree_i, _d_core, etc.

        if self.n_features is not None:
            # "inherent dimensionality" as set by the user
            n_features = max(1.0, float(self.n_features))
        elif self.n_features_ is not None:
            n_features = self.n_features_
        else:
            raise ValueError("Please set the n_features attribute manually.")

        if self.verbose:
            print("[genieclust] Determining clusters with GIc.", file=sys.stderr)

        # apply the Genie+Ic algorithm:
        res = core.gic_from_mst(
            self._tree_d_,
            self._tree_i_,
            n_features=n_features,
            n_clusters=self.n_clusters,
            add_clusters=self.add_clusters,
            gini_thresholds=self.gini_thresholds,
            coarser=self.coarser
        )

        ########################################################

        self._links_ = res["links"]
        Z = core.get_linkage_matrix(
            self._links_,
            self._tree_d_,
            self._tree_i_
        )
        self.children_    = Z["children"]
        self.distances_   = Z["distances"]
        self.counts_      = Z["counts"]

        self._iters_      = res["iters"]
        self.n_clusters_  = res["n_clusters"]
        self._cut_edges_  = self._links_[self._links_ >= 0][:-self.n_clusters_:-1].copy()

        if self.n_clusters_ != self.n_clusters:
            warnings.warn("The number of clusters detected (%d) is "
                          "different from the requested one (%d)." % (
                            self.n_clusters_,
                            self.n_clusters))

        if self.coarser:
            self.labels_matrix_ = res["labels"]
            self.labels_ = self.labels_matrix_[-1,:]
        else:
            self.labels_ = res["labels"]

        ########################################################

        if self.verbose:
            print("[genieclust] Done.", file=sys.stderr)

        return self
