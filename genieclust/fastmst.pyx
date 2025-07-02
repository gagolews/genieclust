# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
k-nearest neighbours and minimum spanning trees with respect to the Euclidean
metric or the thereon-based mutual reachability distances. The module provides
access to a quite fast implementation of K-d trees.

For best speed, consider building the package from sources
using, e.g., `-O3 -march=native` compiler flags and with OpenMP support on.
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2025-2025, Marek Gagolewski <https://www.gagolewski.com>      #
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


import numpy as np
cimport numpy as np
np.import_array()
import os
import warnings

cimport libc.math
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.math cimport INFINITY


ctypedef fused T:
    int
    long
    long long
    Py_ssize_t
    float
    double

ctypedef fused floatT:
    float
    double


from . cimport c_omp
from . cimport c_fastmst


################################################################################

from . cimport c_omp

# cdef void _openmp_set_num_threads():
#     c_omp.Comp_set_num_threads(int(os.getenv("OMP_NUM_THREADS", -1)))

cpdef int omp_set_num_threads(int n_threads):
    """
    genieclust.fastmst.omp_set_num_threads(n_threads)
    """
    return c_omp.Comp_set_num_threads(n_threads)


cpdef int omp_get_max_threads():
    """
    genieclust.fastmst.omp_get_max_threads()

    The function's name is confusing: it returns the maximal number
    of threads that will be used during the next call to a parallelised
    function, not the maximal number of threads possibly available.
    """
    return c_omp.Comp_get_max_threads()


################################################################################



cpdef tuple tree_order(const floatT[:] tree_dist, const Py_ssize_t[:,:] tree_ind):
    """
    genieclust.fastmst.tree_order(tree_dist, tree_ind)


    Orders the edges of a graph (e.g., a spanning tree) wrt the weights
    increasingly, resolving ties if needed based on the points' IDs,
    i.e., the triples (dist, ind1, ind2) are sorted lexicographically.


    Parameters
    ----------

    tree_dist : c_contiguous ndarray, shape (m,)
        The m edges' weights

    tree_ind : c_contiguous ndarray, shape (m,2)
        The corresponding pairs of IDs of the incident nodes


    Returns
    -------

    pair : tuple
        A pair (tree_dist, tree_ind) after the ordering.

    """
    cdef Py_ssize_t m = tree_dist.shape[0]

    if tree_ind.shape[0] != m or tree_ind.shape[1] != 2:
        raise ValueError("incorrect shape of tree_ind")

    cdef np.ndarray[floatT] tree_dist_ret = np.asarray(tree_dist, order="C", copy=True)
    cdef np.ndarray[Py_ssize_t,ndim=2] tree_ind_ret = np.asarray(tree_ind, order="C", copy=True)

    c_fastmst.Ctree_order(m, &tree_dist_ret[0], &tree_ind_ret[0,0])

    return tree_dist_ret, tree_ind_ret




cpdef tuple knn_euclid(
    const floatT[:,:] X,
    Py_ssize_t k=1,
    const floatT[:,:] Y=None,
    str algorithm="auto",
    int max_leaf_size=0,
    bint squared=False,
    bint verbose=False
):
    """
    genieclust.fastmst.knn_euclid(X, k=1, Y=None, algorithm="auto", max_leaf_size=0, squared=False, verbose=False)

    If `Y` is None, then the function determines the first `k` amongst the nearest
    neighbours of each point in `X` with respect to the Euclidean distance.
    It is assumed that each query point is not its own neighbour.

    Otherwise, for each point in `Y`, this function determines the `k` nearest
    points thereto from `X`.

    The implemented algorithms, see the `algorithm` parameter, assume that
    `k` is rather small; say, `k <= 20`.

    Our implementation of K-d trees [1]_ has been quite optimised; amongst
    others, it has good locality of reference, features the sliding midpoint
    (midrange) rule suggested in [2]_, and a node pruning strategy inspired
    by the discussion in [3]_.  Still, it is well-known that K-d trees
    perform well only in spaces of low intrinsic dimensionality.  Thus,
    due to the so-called curse of dimensionality, for high `d`, the brute-force
    algorithm is recommended.

    The number of threads used is controlled via the `OMP_NUM_THREADS``
    environment variable or via ``genieclust.fastmst.omp_set_num_threads``
    at runtime.  For best speed, consider building the package from sources
    using, e.g., ``-O3 -march=native`` compiler flags.


    References
    ----------

    [1] J.L. Bentley, Multidimensional binary search trees used for associative
    searching, Communications of the ACM 18(9), 509–517, 1975,
    DOI:10.1145/361002.361007.

    [2] S. Maneewongvatana, D.M. Mount, It's okay to be skinny, if your friends
    are fat, 4th CGC Workshop on Computational Geometry, 1999.

    [3] N. Sample, M. Haines, M. Arnold, T. Purcell, Optimizing search
    strategies in K-d Trees, 5th WSES/IEEE Conf. on Circuits, Systems,
    Communications & Computers (CSCC'01), 2001.


    Parameters
    ----------

    X : matrix, shape `(n,d)`
        the `n` input points in :math:`\\mathbb{R}^d` (the "database")
    k : int `< n`
        number of nearest neighbours (should be rather small, say, `<= 20`)
    Y : None or an ndarray, shape `(m,d)`
        the "query points"; note that setting `Y=X`, contrary to `Y=None`,
        will include the query points themselves amongst their own neighbours
    algorithm : ``{"auto", "kd_tree", "brute"}``, default="auto"
        K-d trees can only be used for `d` between 2 and 20 only.
        ``"auto"`` selects ``"kd_tree"`` in low-dimensional spaces
    max_leaf_size : int
        maximal number of points in the K-d tree leaves;
        smaller leaves use more memory, yet are not necessarily faster;
        use ``0`` to select the default value, currently set to 32.
    squared : False
        whether to return the squared Euclidean distance
    verbose: bool
        whether to print diagnostic messages


    Returns
    -------

    pair : tuple
        A pair ``(dist, ind)`` representing the k-NN graph, where:
            dist : a c_contiguous ndarray, shape `(n,k)` or `(m,k)`;
                ``dist[i,:]`` is sorted nondecreasingly for all `i`,
                ``dist[i,j]`` gives the weight of the edge ``{i, ind[i,j]}``,
                i.e., the distance between the `i`-th point and its ``j``-th NN.
            ind : a c_contiguous ndarray of the same shape;
                ``ind[i,j]`` is the index (between `0` and `n-1`)
                of the `j`-th nearest neighbour of `i`.
    """
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]
    cdef Py_ssize_t m

    if n < 1 or d <= 1: raise ValueError("X is ill-shaped");
    if k < 1: raise ValueError("k must be >= 1")

    if algorithm == "auto":
        algorithm = "kd_tree" if 2 <= d <= 20 else "brute"

    if algorithm == "kd_tree":
        if not 2 <= d <= 20:
            raise ValueError("K-d trees can only be used for 2 <= d <= 20")

        if max_leaf_size == 0: max_leaf_size = 32  # the current default

        if max_leaf_size <= 0:
            raise ValueError("max_leaf_size must be positive")

        use_kdtree = True

    elif algorithm == "brute":
        use_kdtree = False
    else:
        raise ValueError("invalid 'algorithm'")

    cdef np.ndarray[Py_ssize_t,ndim=2] ind
    cdef np.ndarray[floatT,ndim=2]     dist

    cdef np.ndarray[floatT,ndim=2] X2
    X2 = np.asarray(X, order="C", copy=True)  # destroyable

    cdef np.ndarray[floatT,ndim=2] Y2

    if Y is None:
        if k >= n:
            raise ValueError("too many neighbours requested")

        ind  = np.empty((n, k), dtype=np.intp)
        dist = np.empty((n, k), dtype=np.float32 if floatT is float else np.float64)

        if use_kdtree:
            c_fastmst.Cknn1_euclid_kdtree(&X2[0,0], n, d, k, &dist[0,0], &ind[0,0], max_leaf_size, squared, verbose)
        else:
            c_fastmst.Cknn1_euclid_brute(&X2[0,0], n, d, k, &dist[0,0], &ind[0,0], squared, verbose)
    else:
        if k > n:
            raise ValueError("too many neighbours requested")

        Y2 = np.asarray(Y, order="C", copy=True, dtype=np.float32 if floatT is float else np.float64)

        if Y2.ndim == 1: Y2 = Y2.reshape(1, -1)

        if Y2.ndim != 2 or Y2.shape[1] != d:
            raise ValueError("Y's dimensionality does not match that of X")

        m = Y2.shape[0]

        ind  = np.empty((m, k), dtype=np.intp)
        dist = np.empty((m, k), dtype=np.float32 if floatT is float else np.float64)

        if use_kdtree:
            c_fastmst.Cknn2_euclid_kdtree(&X2[0,0], n, &Y2[0,0], m, d, k, &dist[0,0], &ind[0,0], max_leaf_size, squared, verbose)
        else:
            c_fastmst.Cknn2_euclid_brute(&X2[0,0], n, &Y2[0,0], m, d, k, &dist[0,0], &ind[0,0], squared, verbose)

    return dist, ind



cpdef tuple mst_euclid(
    const floatT[:,:] X,
    Py_ssize_t M=1,
    str algorithm="auto",
    int max_leaf_size=0,
    int first_pass_max_brute_size=0,
    bint verbose=False
):
    """
    genieclust.fastmst.mst_euclid(X, M=1, algorithm="auto", max_leaf_size=0, first_pass_max_brute_size=0, verbose=False)

    The function determines the/a(\*) minimum spanning tree (MST) of a set
    of `n` points, i.e., an acyclic undirected graph whose vertices represent
    the points, and `n-1` edges with the minimal sum of weights, given by
    the pairwise distances.  MSTs have many uses in, amongst others,
    topological data analysis (clustering, dimensionality reduction, etc.).

    For `M<=2`, we get a spanning tree that minimises the sum of Euclidean
    distances between the points. If `M==2`, the function additionally returns
    the distance to each point's nearest neighbour.

    If `M>2`, the spanning tree is the smallest wrt the degree-`M`
    mutual reachability distance [9]_ given by
    :math:`d_M(i, j)=\\max\\{ c_M(i), c_M(j), d(i, j)\\}`, where :math:`d(i,j)`
    is the Euclidean distance between the `i`-th and the `j`-th point,
    and :math:`c_M(i)` is the `i`-th `M`-core distance defined as the distance
    between the `i`-th point and its `(M-1)`-th nearest neighbour
    (not including the query points themselves).
    In clustering and density estimation, `M` plays the role of a smoothing
    factor; see [10]_ and the references therein for discussion. This parameter
    corresponds to the ``hdbscan`` Python package's ``min_samples=M-1``.


    Implementation
    --------------

    (\*) We note that if there are many pairs of equidistant points,
    there can be many minimum spanning trees. In particular, it is likely
    that there are point pairs with the same mutual reachability distances.
    To make the definition less ambiguous (albeit with no guarantees),
    internally, we rely on the adjusted distance
    :math:`d_M(i, j)=\\max\\{c_M(i), c_M(j), d(i, j)\\}+\\varepsilon d(i, j)`,
    where :math:`\\varepsilon` is a small positive constant.

    The implemented algorithms, see the `algorithm` parameter, assume that
    `M` is rather small; say, `M <= 20`.

    Our implementation of K-d trees [6]_ has been quite optimised; amongst
    others, it has good locality of reference (at the cost of making a
    copy of the input dataset), features the sliding midpoint (midrange) rule
    suggested in [7]_, and a node pruning strategy inspired by the discussion
    in [8]_.

    The "single-tree" version of the Borůvka algorithm is naively
    parallelisable: in every iteration, it seeks each point's nearest "alien",
    i.e., the nearest point thereto from another cluster.
    The "dual-tree" Borůvka version of the algorithm is, in principle, based
    on [5]_.  As far as our implementation is concerned, the dual-tree approach
    is often only faster in 2- and 3-dimensional spaces, for `M<=2`, and in
    a single-threaded setting.  For another (approximate) adaptation
    of the dual-tree algorithm to the mutual reachability distance; see [11]_.

    Nevertheless, it is well-known that K-d trees perform well only in spaces
    of low intrinsic dimensionality (a.k.a. the "curse").  For high `d`,
    the "brute-force" algorithm is recommended.  Here, we provided a
    parallelised [2]_ version of the Jarník [1]_ (a.k.a.
    Prim [3_] or Dijkstra) algorithm, where the distances are computed
    on the fly (only once for `M<=2`).

    The number of threads is controlled via the ``OMP_NUM_THREADS``
    environment variable or via ``genieclust.fastmst.omp_set_num_threads``
    at runtime. For best speed, consider building the package from sources
    using, e.g., ``-O3 -march=native`` compiler flags.



    References
    ----------

    [1] V. Jarník, O jistém problému minimálním,
    Práce Moravské Přírodovědecké Společnosti 6, 1930, 57–63.

    [2] C.F. Olson, Parallel algorithms for hierarchical clustering,
    Parallel Computing 21(8), 1995, 1313–1325.

    [3] R. Prim, Shortest connection networks and some generalizations,
    The Bell System Technical Journal 36(6), 1957, 1389–1401.

    [4] O. Borůvka, O jistém problému minimálním,
    Práce Moravské Přírodovědecké Společnosti 3, 1926, 37–58.

    [5] W.B. March, R. Parikshit, A.G. Gray, Fast Euclidean minimum spanning
    tree: Algorithm, analysis, and applications, Proc. 16th ACM SIGKDD Intl.
    Conf. Knowledge Discovery and Data Mining (KDD '10), 2010, 603–612.

    [6] J.L. Bentley, Multidimensional binary search trees used for associative
    searching, Communications of the ACM 18(9), 509–517, 1975,
    DOI:10.1145/361002.361007.

    [7] S. Maneewongvatana, D.M. Mount, It's okay to be skinny, if your friends
    are fat, The 4th CGC Workshop on Computational Geometry, 1999.

    [8] N. Sample, M. Haines, M. Arnold, T. Purcell, Optimizing search
    strategies in K-d Trees, 5th WSES/IEEE Conf. on Circuits, Systems,
    Communications & Computers (CSCC'01), 2001.

    [9] R.J.G.B. Campello, D. Moulavi, J. Sander, Density-based clustering based
    on hierarchical density estimates, Lecture Notes in Computer Science 7819,
    2013, 160–172. DOI:10.1007/978-3-642-37456-2_14.

    [10] R.J.G.B. Campello, D. Moulavi, A. Zimek. J. Sander, Hierarchical
    density estimates for data clustering, visualization, and outlier detection,
    ACM Transactions on Knowledge Discovery from Data (TKDD) 10(1),
    2015, 1–51, DOI:10.1145/2733381.

    [11] L. McInnes, J. Healy, Accelerated hierarchical density-based
    clustering, IEEE Intl. Conf. Data Mining Workshops (ICMDW), 2017, 33–42,
    DOI:10.1109/ICDMW.2017.12.


    Parameters
    ----------

    X : matrix, shape `(n,d)`
        the `n` input points in :math:`\\mathbb{R}^d`
    M : int `< n`
        the degree of the mutual reachability distance (should be rather small,
        say, `<= 20`). `M<=2` denotes the ordinary Euclidean distance
    algorithm : ``{"auto", "kd_tree_single", "kd_tree_dual", "brute"}``, default="auto"
        K-d trees can only be used for `d` between 2 and 20 only.
        ``"auto"`` selects ``"kd_tree_dual"`` for `d<=3`, `M<=2`,
        and in a single-threaded setting only. ``"kd_tree_single"`` is used
        otherwise, unless `d>20`.
    max_leaf_size : int
        maximal number of points in the K-d tree leaves;
        smaller leaves use more memory, yet are not necessarily faster;
        use ``0`` to select the default value, currently set to 32 for the
        single-tree and 8 for the dual-tree Borůvka algorithm
    first_pass_max_brute_size : int
        minimal number of points in a node to treat it as a leaf (unless it's actually a leaf) in the first iteration of the algorithm;
        use ``0`` to select the default value, currently set to 32
    verbose: bool
        whether to print diagnostic messages


    Returns
    -------

    tuple
        If `M==1`, a pair ``(mst_dist, mst_index)`` defining the `n-1` edges
        of the computed spanning tree is returned:

          1. the `(n-1)`-ary array ``mst_dist`` is such that
          ``mst_dist[i]`` gives the weight of the `i`-th edge;

          2. ``mst_index`` is a matrix with `n-1` rows and `2` columns, where
          ``{mst_index[i,0], mst_index[i,1]}`` defines the `i`-th edge of the tree.

        The tree edges are ordered w.r.t. weights nondecreasingly, and then by
        the indexes (lexicographic ordering of the ``(weight, index1, index2)``
        triples).  For each `i`, it holds ``mst_index[i,0]<mst_index[i,1]``.

        For `M>1`, we additionally get:

          3. an `n` by `M-1` matrix ``nn_dist`` giving the distances between
          each point and its `M-1` nearest neighbours,

          4. a matrix of the same shape ``nn_index`` providing the corresponding
          indexes of the neighbours.
    """

    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]

    if n < 1 or d <= 1: raise ValueError("X is ill-shaped");
    if M < 1 or M > n-1: raise ValueError("incorrect M")

    if algorithm == "auto":
        if 2 <= d <= 20:
            if c_omp.Comp_get_max_threads() == 1 and d <= 3 and M <= 2:
                algorithm = "kd_tree_dual"
            else:
                algorithm = "kd_tree_single"
        else:
            algorithm = "brute"

    if algorithm == "kd_tree_single" or algorithm == "kd_tree_dual":
        if not 2 <= d <= 20:
            raise ValueError("K-d trees can only be used for 2 <= d <= 20")

        use_kdtree = True

        if algorithm == "kd_tree_single":
            if max_leaf_size == 0:
                max_leaf_size = 32  # the current default
            if first_pass_max_brute_size == 0:
                first_pass_max_brute_size = 32  # the current default
            use_dtb = False
        elif algorithm == "kd_tree_dual":
            if max_leaf_size == 0:
                max_leaf_size = 8  # the current default
            if first_pass_max_brute_size == 0:
                first_pass_max_brute_size = 32  # the current default
            use_dtb = True

        if max_leaf_size <= 0:
            raise ValueError("max_leaf_size must be positive")
        if first_pass_max_brute_size <= 0:
            raise ValueError("first_pass_max_brute_size must be positive")

    elif algorithm == "brute":
        use_kdtree = False
    else:
        raise ValueError("invalid 'algorithm'")

    cdef np.ndarray[Py_ssize_t,ndim=2] mst_ind  = np.empty((n-1, 2),
        dtype=np.intp)
    cdef np.ndarray[floatT] mst_dist = np.empty(n-1,
        dtype=np.float32 if floatT is float else np.float64)

    cdef np.ndarray[floatT,ndim=2] X2
    X2 = np.asarray(X, order="C", copy=True)  # destroyable

    cdef np.ndarray[Py_ssize_t,ndim=2] nn_ind
    cdef np.ndarray[floatT,ndim=2]     nn_dist
    if M > 1:
        nn_ind  = np.empty((n, M-1), dtype=np.intp)
        nn_dist = np.empty((n, M-1), dtype=np.float32 if floatT is float else np.float64)

    if use_kdtree:
        c_fastmst.Cmst_euclid_kdtree(
            &X2[0,0], n, d, M,
            &mst_dist[0], &mst_ind[0,0],
            <floatT*>(0) if M==1 else &nn_dist[0,0],
            <Py_ssize_t*>(0) if M==1 else &nn_ind[0,0],
            max_leaf_size, first_pass_max_brute_size, use_dtb, verbose
        )
    else:
        c_fastmst.Cmst_euclid_brute(
            &X2[0,0], n, d, M,
            &mst_dist[0], &mst_ind[0,0],
            <floatT*>(0) if M==1 else &nn_dist[0,0],
            <Py_ssize_t*>(0) if M==1 else &nn_ind[0,0],
            verbose
        )

    if M == 1:
        return mst_dist, mst_ind
    else:
        return mst_dist, mst_ind, nn_dist, nn_ind
