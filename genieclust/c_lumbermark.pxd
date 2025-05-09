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


cdef extern from "../src/c_lumbermark.h":

    cdef cppclass CLumbermark[T]:
        CLumbermark() except +
        CLumbermark(T* mst_d, Py_ssize_t* mst_i, Py_ssize_t n) except +
        Py_ssize_t compute(
            Py_ssize_t n_clusters, Py_ssize_t min_cluster_size,
            T min_cluster_factor, bint skip_leaves
        ) except +
        void get_labels(Py_ssize_t* res)
        void get_links(Py_ssize_t* res)
        void get_is_noise(int* res)
