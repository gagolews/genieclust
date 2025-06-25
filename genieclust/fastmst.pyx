# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
The "new" (2025) functions to compute k-nearest neighbours
and minimum spanning trees with respect to the Euclidean metric
and thereon-based mutual reachability distance.
The module provides access to a quite fast implementation of K-d trees.

For best speed, consider building the package from sources
using, e.g., `-O3 -march=native` compiler flags.
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
    int max_leaf_size=32,
    bint squared=False,
    bint verbose=False
):
    """
    genieclust.fastmst.knn_euclid(X, k=1, Y=None, algorithm="auto", max_leaf_size=32, squared=False, verbose=False)

    If `Y` is None, then the function determines the first `k` amongst the nearest
    neighbours of each point in `X` with respect to the Euclidean distance.
    It is assumed that each query point is not its own neighbour.

    Otherwise, for each point in `Y`, this function determines the `k` nearest
    points thereto from `X`.

    The implemented algorithms, see the `algorithm` parameter, assume that
    `k` is rather small; say, `k <= 20`.

    Our implementation of K-d trees [1]_ has been quite optimised; amongst
    others, it has good locality of reference, features the sliding midpoint
    (midrange) rule suggested in [2]_, and a node pruning strategy inspired by the discussion in [3]_.  However, it is well-known that K-d trees
    perform well only in spaces of low intrinsic dimensionality.  Thus,
    due to the so-called curse of dimensionality, for high `d`, the brute-force
    algorithm is recommended.

    The number of threads used is controlled via the `OMP_NUM_THREADS``
    environment variable or via ``genieclust.internal.omp_set_num_threads``
    at runtime. For best speed, consider building the package from sources
    using, e.g., ``-O3 -march=native`` compiler flags.


    References
    ----------

    [1] J.L. Bentley, Multidimensional binary search trees used for associative
    searching, Communications of the ACM 18(9), 509–517, 1975,
    DOI:10.1145/361002.361007.

    [2] S. Maneewongvatana, D.M. Mount, It's okay to be skinny, if your friends
    are fat, 4th CGC Workshop on Computational Geometry, 1999.

    [3] N. Sample, M. Haines, M. Arnold, T. Purcell, Optimizing search
    strategies in K-d Trees, 5th WSES/IEEE Conf. on Circuits, Systems, Communications & Computers (CSCC 2001), 2001.


    Parameters
    ----------

    X : c_contiguous ndarray, shape `(n,d)`
        the "database"
    k : int `< n`
        number of nearest neighbours (should be rather small, say, <= 20)
    Y : None or an ndarray, shape `(m,d)`
        the "query points"; note that setting `Y=X`, contrary to `Y=None`,
        will include the query points themselves amongst their own
        neighbours
    algorithm : ``{"auto", "kd_tree", "brute"}``, default="auto"
        K-d trees can only be used for d between 2 and 20 only.
        ``"auto"`` selects ``"kd_tree"`` in low-dimensional spaces
    max_leaf_size : int, default 32
        maximal number of points in the K-d tree leaves;
        smaller leaves use more memory, yet are not necessarily faster
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
        if 2 <= d <= 20:
            algorithm = "kd_tree"
        else:
            algorithm = "brute"

    if algorithm == "kd_tree":
        if not 2 <= d <= 20:
            raise ValueError("kd_tree can only be used for 2 <= d <= 20")
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
    bint use_kdtree=True,  # TODO: algorithm  auto brute, kd_tree_single kd_tree_dual
    int max_leaf_size=16,
    int first_pass_max_brute_size=0,  # TODO: drop?
    bint use_dtb=False,  # TODO: drop
    bint verbose=False
):
    """
    genieclust.fastmst.mst_euclid(X, M=1, use_kdtree=True, max_leaf_size=16, first_pass_max_brute_size=0, use_dtb=False, verbose=False)


    TODO: describe.....A Jarník (Prim/Dijkstra)-like algorithm for determining
    a(*) minimum spanning tree (MST) of `X` with respect to a given metric
    (distance). Distances are computed on the fly.
    Memory use: O(n*d).

    Mutual reachability distance... [8]_
    It is assumed that M is rather small, say, `M<=20`.

    or Dual-tree Boruvka....

    Our implementation of K-d trees [6]_ has been quite optimised; amongst
    others, it has good locality of reference, features the sliding midpoint
    (midrange) rule suggested in [7]_, and a node pruning strategy inspired by the discussion in [8]_.  However, it is well-known that K-d trees
    perform well only in spaces of low intrinsic dimensionality.  Thus,
    due to the so-called curse of dimensionality, for high `d`, the brute-force
    algorithm is recommended.

    The number of threads used is controlled via the ``OMP_NUM_THREADS``
    environment variable or via ``genieclust.internal.omp_set_num_threads``
    at runtime. For best speed, consider building the package from sources
    using, e.g., ``-O3 -march=native`` compiler flags.

    TODO: return M nearest neighbours


    References
    ----------

    [1] V. Jarník, O jistém problému minimálním,
    Práce Moravské Přírodovědecké Společnosti 6 (1930) 57–63.

    [2] C.F. Olson, Parallel algorithms for hierarchical clustering,
    Parallel Comput. 21 (1995) 1313–1325.

    [3] R. Prim, Shortest connection networks and some generalizations,
    Bell Syst. Tech. J. 36 (1957) 1389–1401.

    [4] O. Borůvka, O jistém problému minimálním. Práce Mor. Přírodověd. Spol.
    V Brně III 3, 1926, 37–58.

    [5] W.B. March, R. Parikshit, A.G. Gray, Fast Euclidean minimum spanning
    tree: algorithm, analysis, and applications, Proc. 16th ACM SIGKDD Intl.
    Conf. Knowledge Discovery and Data Mining (KDD '10), 2010, 603--612.

    [6] J.L. Bentley, Multidimensional binary search trees used for associative
    searching, Communications of the ACM 18(9), 509–517, 1975,
    DOI:10.1145/361002.361007.

    [7] S. Maneewongvatana, D.M. Mount, It's okay to be skinny, if your friends
    are fat, The 4th CGC Workshop on Computational Geometry, 1999.

    [8] N. Sample, M. Haines, M. Arnold, T. Purcell, Optimizing search
    strategies in K-d Trees, 5th WSES/IEEE Conf. on Circuits, Systems, Communications & Computers (CSCC 2001), 2001.

    [9] R.J.G.B. Campello, D. Moulavi, J. Sander, Density-based clustering based
    on hierarchical density estimates, Lecture Notes in Computer Science 7819
    (2013) 160–172. DOI: 10.1007/978-3-642-37456-2_14.



    Parameters TODO
    ----------

    X : c_contiguous ndarray, shape (n,d) or,
            if metric == "precomputed", (n*(n-1)/2,1) or (n,n)
        n data points in a feature space of dimensionality d
        or pairwise distances between n points
    ...
    algorithm : ``{"auto", "kd_tree", "brute"}``, default="auto"
        K-d trees can only be used for d between 2 and 20 only.
        ``"auto"`` selects ``"kd_tree"`` in low-dimensional spaces
    ...
    verbose: bool
        whether to print diagnostic messages

    Returns TODO
    -------

    tuple
        If `M==1`, a pair ``(mst_dist, mst_ind)`` defining the `n-1` edges of the MST
        is returned:
          a) the (n-1)-ary array ``mst_dist`` is such that
          ``mst_dist[i]`` gives the weight of the `i`-th edge;
          b) ``mst_ind`` is a matrix with `n-1` rows and 2 columns,
          where ``{mst[i,0], mst[i,1]}`` defines the `i`-th edge of the tree.

        For `M>1`, we additionally get:
          c) the `n` core distances.

        The results are ordered w.r.t. weights nondecreasingly,
        and then by indexes (lexicographic ordering of
        ``(weight, index1, index2)`` triples).
        For each `i`, it holds ``mst[i,0]<mst[i,1]``.
    """

    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]

    if n < 1 or d <= 1: raise ValueError("X is ill-shaped");
    if M < 1 or M > n-1: raise ValueError("incorrect M")

    cdef np.ndarray[Py_ssize_t,ndim=2] mst_ind  = np.empty((n-1, 2), dtype=np.intp)
    cdef np.ndarray[floatT] mst_dist = np.empty(n-1,
        dtype=np.float32 if floatT is float else np.float64)
    cdef np.ndarray[floatT] d_core

    cdef np.ndarray[floatT,ndim=2] X2
    X2 = np.asarray(X, order="C", copy=True)  # destroyable

    if M > 1:
        d_core = np.empty(n, dtype=np.float32 if floatT is float else np.float64)

    # _openmp_set_num_threads()
    if use_kdtree and 2 <= d <= 20:
        c_fastmst.Cmst_euclid_kdtree(
            &X2[0,0], n, d, M,
            &mst_dist[0], &mst_ind[0,0], <floatT*>(0) if M==1 else &d_core[0],
            max_leaf_size, first_pass_max_brute_size, use_dtb, verbose
        )
    else:
        c_fastmst.Cmst_euclid_brute(
            &X2[0,0], n, d, M,
            &mst_dist[0], &mst_ind[0,0], <floatT*>(0) if M==1 else &d_core[0],
            verbose
        )

    if M == 1:
        return mst_dist, mst_ind
    else:
        return mst_dist, mst_ind, d_core
