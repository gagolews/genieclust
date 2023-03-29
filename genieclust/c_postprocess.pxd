# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3



"""
Noisy k-partition post-processing
"""


# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2020-2023, Marek Gagolewski <https://www.gagolewski.com>      #
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


cdef extern from "../src/c_postprocess.h":
    void Cmerge_boundary_points(const Py_ssize_t* ind, Py_ssize_t num_edges,
        const Py_ssize_t* nn, Py_ssize_t num_neighbours, Py_ssize_t M,
        Py_ssize_t* c, Py_ssize_t n)
    void Cmerge_noise_points(const Py_ssize_t* ind, Py_ssize_t num_edges,
        Py_ssize_t* c, Py_ssize_t n)
