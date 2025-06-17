# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3



"""
Provides access to the "new" KNN- and MST-related functions.
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


cdef extern from "../src/c_fastmst.h":

    void Comp_set_num_threads(Py_ssize_t n_threads)

    void Ctree_order[T](Py_ssize_t n, T* tree_dist, Py_ssize_t* tree_ind)

    void Cknn_sqeuclid_kdtree[T](
        T* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t k,
        T* nn_dist, Py_ssize_t* nn_ind,
        Py_ssize_t max_leaf_size,
        bint verbose
    ) except +

    void Cknn_sqeuclid_brute[T](
        T* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t k,
        T* nn_dist, Py_ssize_t* nn_ind,
        bint verbose
    ) except +

    void Cmst_euclid_kdtree[T](
        T* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t M,
        T* mst_dist, Py_ssize_t* mst_ind, T* d_core,
        Py_ssize_t max_leaf_size, Py_ssize_t first_pass_max_brute_size,
        bint use_dtb, bint verbose
    ) except +

    void Cmst_euclid_brute[T](
        T* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t M,
        T* mst_dist, Py_ssize_t* mst_ind, T* d_core,
        bint verbose
    ) except +


    # void Cknn_sqeuclid_picotree[T](
    #      T* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t k,
    #     T* nn_dist, Py_ssize_t* nn_ind, Py_ssize_t max_leaf_size, bint verbose) except +
