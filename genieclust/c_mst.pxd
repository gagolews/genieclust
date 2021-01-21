# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3



"""
Provides access to MST-related functions.
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2020-2021, Marek Gagolewski <https://www.gagolewski.com>      #
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


cdef extern from "../src/c_mst.h":

    cdef cppclass CDistance[T]:
        pass

    cdef cppclass CDistanceMutualReachability[T]: # inherits from CDistance
        CDistanceMutualReachability()
        CDistanceMutualReachability(const T* d_core, ssize_t n, CDistance[T]* d_pairwise)

    cdef cppclass CDistanceEuclidean[T]: # inherits from CDistance
        CDistanceEuclidean()
        CDistanceEuclidean(T* X, ssize_t n, ssize_t d)

    cdef cppclass CDistanceEuclideanSquared[T]: # inherits from CDistance
        CDistanceEuclideanSquared()
        CDistanceEuclideanSquared(T* X, ssize_t n, ssize_t d)

    cdef cppclass CDistanceManhattan[T]: # inherits from CDistance
        CDistanceManhattan()
        CDistanceManhattan(T* X, ssize_t n, ssize_t d)

    cdef cppclass CDistanceCosine[T]: # inherits from CDistance
        CDistanceCosine()
        CDistanceCosine(T* X, ssize_t n, ssize_t d)

    cdef cppclass CDistancePrecomputedMatrix[T]: # inherits from CDistance
        CDistancePrecomputedMatrix()
        CDistancePrecomputedMatrix(T* d, ssize_t n)

    cdef cppclass CDistancePrecomputedVector[T]: # inherits from CDistance
        CDistancePrecomputedVector()
        CDistancePrecomputedVector(T* d, ssize_t n)



    # cdef cppclass CMstTriple[T]:
    #     CMstTriple(ssize_t i1, ssize_t i2, T d, bint order=False)

    # ssize_t Cmst_from_nn_list[T](
    #     CMstTriple[T]* nns, ssize_t c,
    #     ssize_t n, T* mst_dist, ssize_t* mst_ind, bint verbose) except +


    ssize_t Cmst_from_nn[T](
        T* dist, ssize_t* ind, const T* d_core, ssize_t n, ssize_t k,
        T* mst_dist, ssize_t* mst_ind, bint* maybe_inexact, bint verbose) except +

    void Cknn_from_complete[T](
        CDistance[T]* D, ssize_t n, ssize_t k,
        T* dist, ssize_t* ind, bint verbose) except +

    void Cmst_from_complete[T](
        CDistance[T]* D, ssize_t n,
        T* mst_dist, ssize_t* mst_ind, bint verbose) except +


    void Comp_set_num_threads(ssize_t n_threads)
