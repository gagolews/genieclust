# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3



"""
Minimum spanning tree and k-nearest neighbour algorithms (the "old" (<= 2025),
slow implementation, but works with some non-Euclidean distances).
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


cdef extern from "../src/c_oldmst.h":

    cdef cppclass CDistance[T]:
        pass

    cdef cppclass CDistanceMutualReachability[T]: # inherits from CDistance
        CDistanceMutualReachability()
        CDistanceMutualReachability(const T* d_core, Py_ssize_t n, CDistance[T]* d_pairwise)

    cdef cppclass CDistanceEuclidean[T]: # inherits from CDistance
        CDistanceEuclidean()
        CDistanceEuclidean(T* X, Py_ssize_t n, Py_ssize_t d)

    cdef cppclass CDistanceEuclideanSquared[T]: # inherits from CDistance
        CDistanceEuclideanSquared()
        CDistanceEuclideanSquared(T* X, Py_ssize_t n, Py_ssize_t d)

    cdef cppclass CDistanceManhattan[T]: # inherits from CDistance
        CDistanceManhattan()
        CDistanceManhattan(T* X, Py_ssize_t n, Py_ssize_t d)

    cdef cppclass CDistanceCosine[T]: # inherits from CDistance
        CDistanceCosine()
        CDistanceCosine(T* X, Py_ssize_t n, Py_ssize_t d)

    cdef cppclass CDistancePrecomputedMatrix[T]: # inherits from CDistance
        CDistancePrecomputedMatrix()
        CDistancePrecomputedMatrix(T* d, Py_ssize_t n)

    cdef cppclass CDistancePrecomputedVector[T]: # inherits from CDistance
        CDistancePrecomputedVector()
        CDistancePrecomputedVector(T* d, Py_ssize_t n)



    void Cknn_from_complete[T](
        CDistance[T]* D, Py_ssize_t n, Py_ssize_t k,
        T* dist, Py_ssize_t* ind, bint verbose) except +

    void Cmst_from_complete[T](
        CDistance[T]* D, Py_ssize_t n,
        T* mst_dist, Py_ssize_t* mst_ind, bint verbose) except +

    Py_ssize_t Cmst_from_nn[T](
        T* dist, Py_ssize_t* ind, const T* d_core, Py_ssize_t n, Py_ssize_t k,
        T* mst_dist, Py_ssize_t* mst_ind, int* maybe_inexact, bint verbose) except +
