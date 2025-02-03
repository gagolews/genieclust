# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3



"""
Provides access to the internal cluster validity measures.
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


cdef extern from "../src/c_cvi.h":
    double c_calinski_harabasz_index(const double* X, const Py_ssize_t* y,
        size_t n, size_t d, Py_ssize_t K)

    double c_dunnowa_index(const double* X, const Py_ssize_t* y,
        size_t n, size_t d, Py_ssize_t K, size_t M,
        const char* owa_numerator, const char* owa_denominator)

    double c_generalised_dunn_index(const double* X, const Py_ssize_t* y,
        size_t n, size_t d, Py_ssize_t K, size_t lowercase_d, size_t uppercase_d)

    double c_negated_ball_hall_index(const double* X, const Py_ssize_t* y,
        size_t n, size_t d, Py_ssize_t K)

    double c_negated_davies_bouldin_index(const double* X, const Py_ssize_t* y,
        size_t n, size_t d, Py_ssize_t K)

    double c_negated_wcss_index(const double* X, const Py_ssize_t* y,
        size_t n, size_t d, Py_ssize_t K)

    double c_silhouette_index(const double* X, const Py_ssize_t* y,
        size_t n, size_t d, Py_ssize_t K)

    double c_silhouette_w_index(const double* X, const Py_ssize_t* y,
        size_t n, size_t d, Py_ssize_t K)

    double c_wcnn_index(const double* X, const Py_ssize_t* y,
        size_t n, size_t d, Py_ssize_t K, size_t M)
