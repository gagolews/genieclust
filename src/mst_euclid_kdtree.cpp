/*  This file is part of the 'quitefastmst' package.
 *
 *  Copyleft (C) 2025-2025, Marek Gagolewski <https://www.gagolewski.com>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Affero General Public License
 *  Version 3, 19 November 2007, published by the Free Software Foundation.
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU Affero General Public License Version 3 for more details.
 *  You should have received a copy of the License along with this program.
 *  If this is not the case, refer to <https://www.gnu.org/licenses/>.
 */


#include "c_fastmst.h"
#include "c_common.h"
#include <cmath>
#include "c_kdtree_boruvka.h"


/**
 * helper function called by Cmst_euclid_kdtree below
 */
template <class FLOAT, Py_ssize_t D>
void _mst_euclid_kdtree(
    FLOAT* X, Py_ssize_t n, Py_ssize_t M,
    FLOAT* mst_dist, Py_ssize_t* mst_ind,
    FLOAT* nn_dist, Py_ssize_t* nn_ind,
    Py_ssize_t max_leaf_size,
    Py_ssize_t first_pass_max_brute_size,
    bool use_dtb,
    FLOAT mutreach_adj,
    bool /*verbose*/
) {
    using DISTANCE=quitefastkdtree::kdtree_distance_sqeuclid<FLOAT, D>;

    GENIECLUST_PROFILER_USE

    GENIECLUST_PROFILER_START
    quitefastkdtree::kdtree_boruvka<FLOAT, D, DISTANCE> tree(X, n, M,
        max_leaf_size, first_pass_max_brute_size, use_dtb, mutreach_adj);
    GENIECLUST_PROFILER_STOP("tree init")

    GENIECLUST_PROFILER_START
    quitefastkdtree::mst<FLOAT, D, DISTANCE>(
        tree, mst_dist, mst_ind, nn_dist, nn_ind
    );
    GENIECLUST_PROFILER_STOP("mst call")

    GENIECLUST_PROFILER_START
    for (Py_ssize_t i=0; i<n-1; ++i)
        mst_dist[i] = sqrt(mst_dist[i]);

    if (M>1) {
        for (Py_ssize_t i=0; i<n*(M-1); ++i)
            nn_dist[i] = sqrt(nn_dist[i]);
    }

    Ctree_order(n-1, mst_dist, mst_ind);
    GENIECLUST_PROFILER_STOP("Cmst_euclid_kdtree finalise")

}


/*! The function determines the/a(\*) minimum spanning tree (MST) of a set
 *  of `n` points, i.e., an acyclic undirected graph whose vertices represent
 *  the points, and `n-1` edges with the minimal sum of weights, given by
 *  the pairwise distances.  MSTs have many uses in, amongst others,
 *  topological data analysis (clustering, dimensionality reduction, etc.).
 *
 *  For `M<=2`, we get a spanning tree that minimises the sum of Euclidean
 *  distances between the points. If `M==2`, the function additionally returns
 *  the distance to each point's nearest neighbour.
 *
 *  If `M>2`, the spanning tree is the smallest wrt the degree-`M`
 *  mutual reachability distance [9]_ given by
 *  :math:`d_M(i, j)=\\max\\{ c_M(i), c_M(j), d(i, j)\\}`, where :math:`d(i,j)`
 *  is the Euclidean distance between the `i`-th and the `j`-th point,
 *  and :math:`c_M(i)` is the `i`-th `M`-core distance defined as the distance
 *  between the `i`-th point and its `(M-1)`-th nearest neighbour
 *  (not including the query points themselves).
 *  In clustering and density estimation, `M` plays the role of a smoothing
 *  factor; see [10]_ and the references therein for discussion. This parameter
 *  corresponds to the ``hdbscan`` Python package's ``min_samples=M-1``.
 *
 *  (\*) We note that if there are many pairs of equidistant points,
 *  there can be many minimum spanning trees. In particular, it is likely
 *  that there are point pairs with the same mutual reachability distances.
 *  See ``mutreach_adj`` for an adjustment to address this (partially).
 *
 *  The implemented algorithm assumes that `M` is rather small; say, `M <= 20`.
 *
 *  Our implementation of K-d trees [6]_ has been quite optimised; amongst
 *  others, it has good locality of reference (at the cost of making a
 *  copy of the input dataset), features the sliding midpoint (midrange) rule
 *  suggested in [7]_, and a node pruning strategy inspired by the discussion
 *  in [8]_.
 *
 *  The "single-tree" version of the Borůvka algorithm is naively
 *  parallelisable: in every iteration, it seeks each point's nearest "alien",
 *  i.e., the nearest point thereto from another cluster.
 *  The "dual-tree" Borůvka version of the algorithm is, in principle, based
 *  on [5]_. As far as our implementation is concerned, the dual-tree approach
 *  is only faster in 2- and 3-dimensional spaces, for `M<=2`, and in
 *  a single-threaded setting.  For another (approximate) adaptation
 *  of the dual-tree algorithm to the mutual reachability distance, see [11]_.
 *
 *  Nevertheless, it is well-known that K-d trees perform well only in spaces
 *  of low intrinsic dimensionality (a.k.a. the "curse").
 *
 *
 *  References:
 *  ----------
 *
 *  [4] O. Borůvka, O jistém problému minimálním. Práce Mor. Přírodověd. Spol.
 *  V Brně III 3, 1926, 37–58.
 *
 *  [5] W.B. March, R. Parikshit, A.G. Gray, Fast Euclidean minimum spanning
 *  tree: algorithm, analysis, and applications, Proc. 16th ACM SIGKDD Intl.
 *  Conf. Knowledge Discovery and Data Mining (KDD '10), 2010, 603–612.
 *
 *  [6] J.L. Bentley, Multidimensional binary search trees used for associative
 *  searching, Communications of the ACM 18(9), 509–517, 1975,
 *  DOI:10.1145/361002.361007.
 *
 *  [7] S. Maneewongvatana, D.M. Mount, It's okay to be skinny, if your friends
 *  are fat, The 4th CGC Workshop on Computational Geometry, 1999.
 *
 *  [8] N. Sample, M. Haines, M. Arnold, T. Purcell, Optimizing search
 *  strategies in K-d Trees, 5th WSES/IEEE Conf. on Circuits, Systems,
 *  Communications & Computers (CSCC'01), 2001.
 *
 *  [9] R.J.G.B. Campello, D. Moulavi, J. Sander, Density-based clustering based
 *  on hierarchical density estimates, Lecture Notes in Computer Science 7819,
 *  2021, 160–172. DOI: 10.1007/978-3-642-37456-2_14.
 *
 *  [10] R.J.G.B. Campello, D. Moulavi, A. Zimek. J. Sander, Hierarchical
 *  density estimates for data clustering, visualization, and outlier detection,
 *  ACM Transactions on Knowledge Discovery from Data (TKDD) 10(1),
 *  2015, 1–51, DOI:10.1145/2733381.
 *
 *  [11] L. McInnes, J. Healy, Accelerated hierarchical density-based
 *  clustering, IEEE Intl. Conf. Data Mining Workshops (ICMDW), 2017, 33–42,
 *  DOI:10.1109/ICDMW.2017.12.
 *
 *
 * @param X [destroyable] a C-contiguous data matrix, shape n*d
 * @param n number of rows
 * @param d number of columns, 2<=d<=20
 * @param M the level of the "core" distance if M > 1
 * @param mst_dist [out] a vector of length n-1, gives weights of the
 *        resulting MST edges in nondecreasing order
 * @param mst_ind [out] a vector of length 2*(n-1), representing
 *        a c_contiguous array of shape (n-1,2), defining the edges
 *        corresponding to mst_d, with mst_i[j,0] < mst_i[j,1] for all j
 * @param nn_dist [out] NULL for M==1 or the n*(M-1) distances to the n points'
 *        (M-1) nearest neighbours
 * @param nn_ind [out] NULL for M==1 or the n*(M-1) indexes of the n points'
 *        (M-1) nearest neighbours
 * @param max_leaf_size maximal number of points in the K-d tree's leaves
 * @param first_pass_max_brute_size minimal number of points in a node to treat
 *        it as a leaf (unless it's actually a leaf) in the first iteration
 *        of the algorithm
 * @param use_dtb whether a dual or a single-tree Borůvka algorithm
 *        should be used
 * @param mutreach_adj (M>2 only) adjustment for mutual reachability distance
 *        ambiguity (for M>2):
 *        values in `(-1,0)` prefer connecting to farther NNs,
 *        values in `(0, 1)` fall for closer NNs,
 *        values in `(-2,-1)` prefer connecting to points with smaller core distances,
 *        values in `(1, 2)` favour larger core distances
 * @param verbose should we output diagnostic/progress messages?
 */
template <class FLOAT>
void Cmst_euclid_kdtree(
    FLOAT* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t M,
    FLOAT* mst_dist, Py_ssize_t* mst_ind,
    FLOAT* nn_dist, Py_ssize_t* nn_ind,
    Py_ssize_t max_leaf_size,
    Py_ssize_t first_pass_max_brute_size,
    bool use_dtb,
    FLOAT mutreach_adj,
    bool verbose
) {
    GENIECLUST_PROFILER_USE

    if (n <= 0)   throw std::domain_error("n <= 0");
    if (d <= 0)   throw std::domain_error("d <= 0");
    if (M <= 0)   throw std::domain_error("M <= 0");
    if (M-1 >= n) throw std::domain_error("M >= n-1");
    if (std::abs(mutreach_adj)>=2) throw std::domain_error("|mutreach_adj|>=2");
    GENIECLUST_ASSERT(mst_dist);
    GENIECLUST_ASSERT(mst_ind);

    if (max_leaf_size <= 0) throw std::domain_error("max_leaf_size <= 0");


    //if (first_pass_max_brute_size <= 0)
    // does no harm - will have no effect

    if (verbose) GENIECLUST_PRINT("[genieclust] Computing the MST... ");

    #define IF_d_CALL_MST_EUCLID_KDTREE(D_) \
        if (d == D_) \
            _mst_euclid_kdtree<FLOAT,  D_>(\
                X, n, M, mst_dist, mst_ind, \
                nn_dist, nn_ind, max_leaf_size, first_pass_max_brute_size, \
                use_dtb, mutreach_adj, verbose \
            )

    /* LMAO; templates... */
    GENIECLUST_PROFILER_START
    /**/ IF_d_CALL_MST_EUCLID_KDTREE(2);
    else IF_d_CALL_MST_EUCLID_KDTREE(3);
    else IF_d_CALL_MST_EUCLID_KDTREE(4);
    else IF_d_CALL_MST_EUCLID_KDTREE(5);
    else IF_d_CALL_MST_EUCLID_KDTREE(6);
    else IF_d_CALL_MST_EUCLID_KDTREE(7);
    else IF_d_CALL_MST_EUCLID_KDTREE(8);
    else IF_d_CALL_MST_EUCLID_KDTREE(9);
    else IF_d_CALL_MST_EUCLID_KDTREE(10);
    else IF_d_CALL_MST_EUCLID_KDTREE(11);
    else IF_d_CALL_MST_EUCLID_KDTREE(12);
    else IF_d_CALL_MST_EUCLID_KDTREE(13);
    else IF_d_CALL_MST_EUCLID_KDTREE(14);
    else IF_d_CALL_MST_EUCLID_KDTREE(15);
    else IF_d_CALL_MST_EUCLID_KDTREE(16);
    else IF_d_CALL_MST_EUCLID_KDTREE(17);
    else IF_d_CALL_MST_EUCLID_KDTREE(18);
    else IF_d_CALL_MST_EUCLID_KDTREE(19);
    else IF_d_CALL_MST_EUCLID_KDTREE(20);
    else {
        // TODO: does it work for d==1?
        // although then a trivial, faster algorithm exists...
        throw std::runtime_error("d should be between 2 and 20");
    }
    GENIECLUST_PROFILER_STOP("Cmst_euclid_kdtree");

    if (verbose) GENIECLUST_PRINT("done.\n");
}


// instantiate:
template void Cmst_euclid_kdtree<float>(
    float* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t M,
    float* mst_dist, Py_ssize_t* mst_ind,
    float* nn_dist, Py_ssize_t* nn_ind,
    Py_ssize_t max_leaf_size,
    Py_ssize_t first_pass_max_brute_size,
    bool use_dtb,
    float mutreach_adj,
    bool verbose
);

template void Cmst_euclid_kdtree<double>(
    double* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t M,
    double* mst_dist, Py_ssize_t* mst_ind,
    double* nn_dist, Py_ssize_t* nn_ind,
    Py_ssize_t max_leaf_size,
    Py_ssize_t first_pass_max_brute_size,
    bool use_dtb,
    double mutreach_adj,
    bool verbose
);
