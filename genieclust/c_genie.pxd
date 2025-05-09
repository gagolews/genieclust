# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3



"""
Provides access to the CGenie and CGIc classes.
"""


# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2020-2025, Marek Gagolewski <https://www.gagolewski.com>      #
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


cdef extern from "../src/c_genie.h":

    cdef cppclass CGenie[T]:
        CGenie() except +
        CGenie(T* mst_d, Py_ssize_t* mst_i, Py_ssize_t n, bint skip_leaves, bint experimental_forced_merge) except +
        void compute(Py_ssize_t n_clusters, double gini_threshold) except +
        Py_ssize_t get_max_n_clusters()
        Py_ssize_t get_links(Py_ssize_t* res)
        Py_ssize_t get_labels(Py_ssize_t n_clusters, Py_ssize_t* res)
        void get_labels_matrix(Py_ssize_t n_clusters, Py_ssize_t* res)
        void get_is_noise(int* res)


    cdef cppclass CGIc[T]:
        CGIc() except +
        CGIc(T* mst_d, Py_ssize_t* mst_i, Py_ssize_t n, bint skip_leaves) except +
        void compute(Py_ssize_t n_clusters, Py_ssize_t add_clusters,
            double n_features, double* gini_thresholds, Py_ssize_t n_thresholds)  except +
        Py_ssize_t get_max_n_clusters()
        Py_ssize_t get_links(Py_ssize_t* res)
        Py_ssize_t get_labels(Py_ssize_t n_clusters, Py_ssize_t* res)
        void get_labels_matrix(Py_ssize_t n_clusters, Py_ssize_t* res)
