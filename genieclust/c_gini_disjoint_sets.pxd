"""
cppclass CGiniDisjointSets
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


from libcpp.vector cimport vector

cdef extern from "../src/c_gini_disjoint_sets.h":
    cdef cppclass CGiniDisjointSets:
        CGiniDisjointSets() except +
        CGiniDisjointSets(ssize_t) except +
        ssize_t get_k()
        ssize_t get_n()
        ssize_t find(ssize_t)
        ssize_t merge(ssize_t, ssize_t)
        double get_gini()
        ssize_t get_smallest_count()
        ssize_t get_count(ssize_t)
        void get_counts(ssize_t*)
