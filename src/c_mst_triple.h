/*
 *  Copyleft (C) 2018-2025, Marek Gagolewski <https://www.gagolewski.com>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Affero General Public License
 *  Version 3, 19 November 2007, published by the Free Software Foundation.
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU Affero General Public License Version 3 for more details.
 *  You should have received a copy of the License along with this program.
 *  If this is not the case, refer to <https://www.gnu.org/licenses/>.
 */


#ifndef __c_mst_triple_h
#define __c_mst_triple_h


/*! Represents an edge in a weighted graph.
 *  Features a comparer used to sort the edges w.r.t. increasing weights;
 *  more precisely, lexicographically w.r.t. (d, i1, d2).
 */
template <class T>
struct CMstTriple
{
    Py_ssize_t i1;  //!< first  vertex defining an edge
    Py_ssize_t i2;  //!< second vertex defining an edge
    T d;            //!< edge weight

    CMstTriple() {}

    CMstTriple(Py_ssize_t i1, Py_ssize_t i2, T d, bool order=true)
    {
        GENIECLUST_ASSERT(i1 != i2);
        GENIECLUST_ASSERT(i1 >= 0);
        GENIECLUST_ASSERT(i2 >= 0);
        this->d = d;
        if (!order || (i1 < i2)) {
            this->i1 = i1;
            this->i2 = i2;
        }
        else {
            this->i1 = i2;
            this->i2 = i1;
        }
    }

    bool operator<(const CMstTriple<T>& other) const
    {
        if (d == other.d) {
            if (i1 == other.i1)
                return i2 < other.i2;
            else
                return i1 < other.i1;
        }
        else
            return d < other.d;
    }
};

#endif
