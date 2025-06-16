# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
The "new" (2025) functions to compute k nearest neighbours
and minimum spanning trees with respect to the Euclidean metric
and thereon-based mutual reachability distance.
Provides access to our fast implementation of K-d trees.
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


from . cimport c_fastmst


################################################################################

cdef void _openmp_set_num_threads():
    c_fastmst.Comp_set_num_threads(int(os.getenv("OMP_NUM_THREADS", -1)))




cpdef tuple knn_sqeuclid(
    floatT[:,::1] X,
    Py_ssize_t k,
    bint use_kdtree=True,
    int max_leaf_size=32,
    bint verbose=False
):
    """TODO: describe
    Determines the first k nearest neighbours of each point in X
    with respect to the squared Euclidean distance

    It is assumed that each query point is not its own neighbour.

    The implemented algorithms assume that k is rather small, say, k <= 20.


    Parameters
    ----------

    X : c_contiguous ndarray, shape (n,d)
    k : int < n
        number of nearest neighbours
    use_kdtree : True
        whether a KD-tree should be used (for d <= 20 only);
        good for small k, small d, but large n.
    max_leaf_size : int
        number of points in leaves of the KD-tree; 0 for the default value
    verbose: bool
        whether to print diagnostic messages


    Returns
    -------

    pair : tuple
        A pair (dist, ind) representing the k-NN graph, where:
            dist : a c_contiguous ndarray, shape (n,k)
                dist[i,:] is sorted nondecreasingly for all i,
                dist[i,j] gives the weight of the edge {i, ind[i,j]},
                i.e., the distance between the i-th point and its j-th NN.
            ind : a c_contiguous ndarray, shape (n,k)
                edge definition, interpreted as {i, ind[i,j]};
                ind[i,j] is the index of the j-th nearest neighbour of i.
    """
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]

    if k >= n:
        raise ValueError("too many nearest neighbours requested")

    cdef Py_ssize_t i
    cdef np.ndarray[Py_ssize_t,ndim=2] ind  = np.empty((n, k), dtype=np.intp)
    cdef np.ndarray[floatT,ndim=2]  dist = np.empty((n, k),
        dtype=np.float32 if floatT is float else np.float64)

    cdef np.ndarray[floatT,ndim=2] X2
    X2 = np.asarray(X, order="C", copy=True)  # destroyable

    _openmp_set_num_threads()
    if use_kdtree and 2 <= d <= 20:
        #    c_fastmst.Cknn_sqeuclid_picotree(&X2[0,0], n, d, k, &dist[0,0], &ind[0,0], max_leaf_size, verbose)  # ours is faster
        c_fastmst.Cknn_sqeuclid_kdtree(&X2[0,0], n, d, k, &dist[0,0], &ind[0,0], max_leaf_size, verbose)
    else:
        c_fastmst.Cknn_sqeuclid_brute(&X2[0,0], n, d, k, &dist[0,0], &ind[0,0], verbose)

    return dist, ind



cpdef tuple tree_order(floatT[::1] tree_dist, Py_ssize_t[:,::1] tree_ind):
    """
    Orders the edges of a graph (e.g., a spanning tree) wrt the weights
    increasingly, resolving ties if needed based on the points' IDs.


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



cpdef tuple mst_euclid(
    floatT[:,::1] X,
    Py_ssize_t M,
    bint use_kdtree=True,
    int max_leaf_size=4,
    int first_pass_max_brute_size=16,
    bint verbose=False
):
    """
    TODO: describe.....A Jarník (Prim/Dijkstra)-like algorithm for determining
    a(*) minimum spanning tree (MST) of X with respect to a given metric
    (distance). Distances are computed on the fly.
    Memory use: O(n*d).

    It is assumed that M is rather small, say, M<=20.

    or Dual-tree Boruvka

    References
    ----------

    [1] V. Jarník, O jistém problému minimálním,
    Práce Moravské Přírodovědecké Společnosti 6 (1930) 57–63.

    [2] C.F. Olson, Parallel algorithms for hierarchical clustering,
    Parallel Comput. 21 (1995) 1313–1325.

    [3] R. Prim, Shortest connection networks and some generalizations,
    Bell Syst. Tech. J. 36 (1957) 1389–1401.

    [4] R.J.G.B. Campello, D. Moulavi, J. Sander, Density-based clustering based
    on hierarchical density estimates, Lecture Notes in Computer Science 7819
    (2013) 160–172. DOI: 10.1007/978-3-642-37456-2_14.


    TODO: .... papers on kd-trees and dual-tree Boruvka,
    + Boruvka algo


    Parameters TODO
    ----------

    X : c_contiguous ndarray, shape (n,d) or,
            if metric == "precomputed", (n*(n-1)/2,1) or (n,n)
        n data points in a feature space of dimensionality d
        or pairwise distances between n points
    ...
    verbose: bool
        whether to print diagnostic messages

    Returns TODO
    -------

    triple : tuple
        A triple (mst_dist, mst_ind, d_core) defining the n-1 edges of the MST:
          a) the (n-1)-ary array mst_dist is such that
          mst_dist[i] gives the weight of the i-th edge;
          b) mst_ind is a matrix with n-1 rows and 2 columns,
          where {mst[i,0], mst[i,1]} defines the i-th edge of the tree,
        and giving
          c) the n core distances (or None if M==1).

        The results are ordered w.r.t. nondecreasing weights.
        For each i, it holds mst[i,0]<mst[i,1].
    """

    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]

    if M < 1 or M > n-1: raise ValueError("incorrect M")

    cdef np.ndarray[Py_ssize_t,ndim=2] mst_ind  = np.empty((n-1, 2), dtype=np.intp)
    cdef np.ndarray[floatT] mst_dist = np.empty(n-1,
        dtype=np.float32 if floatT is float else np.float64)
    cdef np.ndarray[floatT] d_core

    cdef np.ndarray[floatT,ndim=2] X2
    X2 = np.asarray(X, order="C", copy=True)  # destroyable

    if M > 1:
        d_core = np.empty(n, dtype=np.float32 if floatT is float else np.float64)

    _openmp_set_num_threads()
    if use_kdtree and 2 <= d <= 20:
        c_fastmst.Cmst_euclid_kdtree(
            &X2[0,0], n, d, M,
            &mst_dist[0], &mst_ind[0,0], <floatT*>(0) if M==1 else &d_core[0],
            max_leaf_size, first_pass_max_brute_size, verbose
        )
    else:
        c_fastmst.Cmst_euclid_brute(
            &X2[0,0], n, d, M,
            &mst_dist[0], &mst_ind[0,0], <floatT*>(0) if M==1 else &d_core[0],
            verbose
        )

    return mst_dist, mst_ind, None if M==1 else d_core
