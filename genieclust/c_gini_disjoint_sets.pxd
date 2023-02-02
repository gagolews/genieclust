"""
cppclass CGiniDisjointSets
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


from libcpp.vector cimport vector

cdef extern from "../src/c_gini_disjoint_sets.h":
    cdef cppclass CGiniDisjointSets:
        CGiniDisjointSets() except +
        CGiniDisjointSets(Py_ssize_t) except +
        Py_ssize_t get_k()
        Py_ssize_t get_n()
        Py_ssize_t find(Py_ssize_t)
        Py_ssize_t merge(Py_ssize_t, Py_ssize_t)
        double get_gini()
        Py_ssize_t get_smallest_count()
        Py_ssize_t get_count(Py_ssize_t)
        void get_counts(Py_ssize_t*)
