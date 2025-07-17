/*  Minimum spanning tree and k-nearest neighbour algorithms
 *  (quite fast in low-dimensional spaces, currently Euclidean distance only)
 *
 *
 *  [1] V. Jarník, O jistém problému minimálním,
 *  Práce Moravské Přírodovědecké Společnosti 6, 1930, 57–63.
 *
 *  [2] C.F. Olson, Parallel algorithms for hierarchical clustering,
 *  Parallel Computing 21(8), 1995, 1313–1325.
 *
 *  [3] R. Prim, Shortest connection networks and some generalizations,
 *  The Bell System Technical Journal 36(6), 1957, 1389–1401.
 *
 *  [4] O. Borůvka, O jistém problému minimálním,
 *  Práce Moravské Přírodovědecké Společnosti 3, 1926, 37–58.
 *
 *  [5] W.B. March, R. Parikshit, A.G. Gray, Fast Euclidean minimum spanning
 *  tree: Algorithm, analysis, and applications, Proc. 16th ACM SIGKDD Intl.
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
 *  2013, 160–172. DOI:10.1007/978-3-642-37456-2_14.
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


#ifndef __c_fastmst_h
#define __c_fastmst_h

#include "c_common.h"


template <class FLOAT>
void Cknn1_euclid_brute(
    const FLOAT* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t k,
    FLOAT* nn_dist, Py_ssize_t* nn_ind, bool squared=false, bool verbose=false
);


template <class FLOAT>
void Cknn2_euclid_brute(
    const FLOAT* X, Py_ssize_t n, const FLOAT* Y, Py_ssize_t m,
    Py_ssize_t d, Py_ssize_t k,
    FLOAT* nn_dist, Py_ssize_t* nn_ind, bool squared=false, bool verbose=false
);


template <class FLOAT>
void Ctree_order(Py_ssize_t m, FLOAT* tree_dist, Py_ssize_t* tree_ind);


template <class FLOAT>
void Cmst_euclid_brute(
    FLOAT* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t M,
    FLOAT* mst_dist, Py_ssize_t* mst_ind,
    FLOAT* nn_dist, Py_ssize_t* nn_ind,
    FLOAT mutreach_adj=-1.00000011920928955078125,
    bool verbose=false
);


template <class FLOAT>
void Cknn2_euclid_kdtree(
    FLOAT* X, const Py_ssize_t n,
    const FLOAT* Y, const Py_ssize_t m,
    const Py_ssize_t d, const Py_ssize_t k,
    FLOAT* nn_dist, Py_ssize_t* nn_ind,
    Py_ssize_t max_leaf_size=32, bool squared=false, bool verbose=false
);


template <class FLOAT>
void Cknn1_euclid_kdtree(
    FLOAT* X, const Py_ssize_t n,
    const Py_ssize_t d, const Py_ssize_t k,
    FLOAT* nn_dist, Py_ssize_t* nn_ind,
    Py_ssize_t max_leaf_size=32, bool squared=false, bool verbose=false
);


template <class FLOAT>
void Cmst_euclid_kdtree(
    FLOAT* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t M,
    FLOAT* mst_dist, Py_ssize_t* mst_ind,
    FLOAT* nn_dist=nullptr, Py_ssize_t* nn_ind=nullptr,
    Py_ssize_t max_leaf_size=32,
    Py_ssize_t first_pass_max_brute_size=32,
    bool use_dtb=false,
    FLOAT mutreach_adj=-1.00000011920928955078125,
    bool verbose=false
);


#endif
