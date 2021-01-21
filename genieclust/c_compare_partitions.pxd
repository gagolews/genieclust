# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3



"""
Provides access to the Ccontingency_table(), Capply_pivoting()
and Ccompare_partitions_*() functions.
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


cdef extern from "../src/c_compare_partitions.h":
    cdef struct CComparePartitionsPairsResult:
        double ar
        double r
        double fm
        double afm

    cdef struct CComparePartitionsInfoResult:
        double mi
        double nmi
        double ami


    void Cminmax[T](const T* x, ssize_t n, T* xmin, T* xmax)

    void Ccontingency_table(ssize_t* C, ssize_t xc, ssize_t yc,
        ssize_t xmin, ssize_t ymin,
        ssize_t* x, ssize_t* y, ssize_t n)

    void Capply_pivoting(ssize_t* C, ssize_t xc, ssize_t yc)

    CComparePartitionsPairsResult Ccompare_partitions_pairs(const ssize_t* C,
        ssize_t xc, ssize_t yc)
    CComparePartitionsInfoResult  Ccompare_partitions_info(const ssize_t* C,
        ssize_t xc, ssize_t yc)
    double Ccompare_partitions_nacc(const ssize_t* C, ssize_t xc, ssize_t yc)
    double Ccompare_partitions_psi(const ssize_t* C, ssize_t xc, ssize_t yc)
