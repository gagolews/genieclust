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
#   Copyleft (C) 2020-2022, Marek Gagolewski <https://www.gagolewski.com>      #
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
    double c_calinski_harabasz_index(const double* X, const ssize_t* y,
        size_t n, size_t d, ssize_t K)

    #double c_dunnowa_index

    #double c_generalised_dunn_index

    double c_negated_ball_hall_index(const double* X, const ssize_t* y,
        size_t n, size_t d, ssize_t K)

    double c_negated_davies_bouldin_index(const double* X, const ssize_t* y,
        size_t n, size_t d, ssize_t K)

    double c_negated_wcss_index(const double* X, const ssize_t* y,
        size_t n, size_t d, ssize_t K)

    double c_silhouette_index(const double* X, const ssize_t* y,
        size_t n, size_t d, ssize_t K)

    double c_silhouette_w_index(const double* X, const ssize_t* y,
        size_t n, size_t d, ssize_t K)

    #double c_wcnn_index
