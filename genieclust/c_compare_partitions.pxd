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

    cdef struct CCompareSetMatchingResult:
        double psi
        double spsi


    void Cminmax[T](const T* x, Py_ssize_t n, T* xmin, T* xmax)

    void Ccontingency_table(Py_ssize_t* Cout, Py_ssize_t xc, Py_ssize_t yc,
        Py_ssize_t xmin, Py_ssize_t ymin,
        Py_ssize_t* x, Py_ssize_t* y, Py_ssize_t n)

    void Cnormalizing_permutation[T](const T* C, Py_ssize_t xc, Py_ssize_t yc,
        Py_ssize_t* Iout)

    void Capply_pivoting(const Py_ssize_t* C, Py_ssize_t xc, Py_ssize_t yc, Py_ssize_t* Cout)

    CComparePartitionsPairsResult Ccompare_partitions_pairs(const Py_ssize_t* C,
        Py_ssize_t xc, Py_ssize_t yc)

    CComparePartitionsInfoResult Ccompare_partitions_info(const Py_ssize_t* C,
        Py_ssize_t xc, Py_ssize_t yc)

    double Ccompare_partitions_nacc(const Py_ssize_t* C, Py_ssize_t xc, Py_ssize_t yc)

    double Ccompare_partitions_aaa(const Py_ssize_t* C, Py_ssize_t xc, Py_ssize_t yc)

    CCompareSetMatchingResult Ccompare_partitions_psi(const Py_ssize_t* C,
        Py_ssize_t xc, Py_ssize_t yc)
