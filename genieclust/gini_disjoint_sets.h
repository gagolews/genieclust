/*  class GiniDisjointSets
 *
 *  Copyright (C) 2018-2019 Marek.Gagolewski.com
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without 
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *  contributors may be used to endorse or promote products derived from this
 *  software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
 *  OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */




#ifndef __gini_disjoint_sets_h
#define __gini_disjoint_sets_h

#include "disjoint_sets.h"


/*! "Augmented" Disjoint Sets (Union-Find) Data Structure
 *
 *  A class to represent partitions of the set {0,1,...,n-1} for any n.
 *
 *  The class allows to compute the normalized Gini index of the distribution 
 *  of subset sizes, i.e.,
 *  \[
 *   G(x_1,\dots,x_k) = \frac{
 *   \sum_{i=1}^{n-1} \sum_{j=i+1}^n |x_i-x_j|
 *   }{
 *   (n-1) \sum_{i=1}^n x_i
 *   }.
 *  \]
 * 
 *  The merge() operation, which also recomputes the Gini index, 
 *  has O(sqrt n) time complexity.
 *
 *  For a use case, see: Gagolewski M., Bartoszuk M., Cena A.,
 *  Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
 *  Information Sciences 363, 2016, pp. 8-23. doi:10.1016/j.ins.2016.05.003
 */
class GiniDisjointSets : public DisjointSets{

protected:
    std::vector<ulonglong> cnt;  //!< cnt[find(x)] is the size of the relevant subset

    std::vector<ulonglong> tab;  /*!< tab[i] gives the number of subsets of size i
                                  * (it's a pretty sparse array 
                                  * - at most sqrt(n) elements are non-zero)
                                  */

    std::vector<ulonglong> tab_next; //!< an array-based...
    std::vector<ulonglong> tab_prev; //!< ...doubly-linked list...
    ulonglong  tab_head; //!< ...for quickly accessing and iterating over...
    ulonglong  tab_tail; //!< ...this->tab data

    double gini;   //!< the Gini index of the current subset sizes


public:
    /*! Starts with a "weak" partition {  {0}, {1}, ..., {n-1}  },
     *  i.e., n singletons.
     * 
     *  @param n number of elements, n>=0.
     */
    GiniDisjointSets(ulonglong n) :
        DisjointSets(n),
        cnt(n, 1),   // each cluster is of size 1
        tab(n+1, 0),
        tab_next(n+1),
        tab_prev(n+1)
    {
        this->tab[1] = n;   // there are n clusters of size 1
        this->tab_head = 1; // the smallest cluster is of size 1
        this->tab_tail = 1; // the largest cluster is of size 1
        this->gini = 0.0;   // perfectly balanced cluster size distribution
    }


    /*! A nullary constructor allows Cython to allocate
     *  the instances on the stack.  Do not use otherwise.
    */
    GiniDisjointSets() : GiniDisjointSets(0) { }


    /*! Returns the Gini index of the distribution of subsets' sizes.
     *
     *  Run time: O(1), as the Gini index is updated during a call
     *  to merge().
     */
    double get_gini() const { return this->gini; }


    /*! Returns the size of the smallest subset. 
     *
     *  Run time: O(1).
     */
    ulonglong get_smallest_count() const { return this->tab_head; }


    /*! Returns the size of the subset containing x.
     *
     * Run time: the cost of find(x)
     */
    ulonglong get_count(ulonglong x) {
        x = this->find(x);
        return this->cnt[x];
    }


    /*! Merges the sets containing x and y in {0,...,n-1}.
     *
     *  Let px be the parent id of x, and py be the parent id of y.
     *  If px < py, then the new parent id of py will be set to py.
     *  Otherwise, px will have py as its parent.
     *
     *  If x and y are already members of the same subset, 
     *  an exception is thrown.
     *
     *  @return the id of the parent of x or y, whichever is smaller.
     * 
     *  @param x a value in {0,...,n-1}
     *  @param y a value in {0,...,n-1}
     *
     *  Update time: pessimistically O(sqrt(n)).
     */
    virtual ulonglong merge(ulonglong x, ulonglong y) { // well, union is a reserved C++ keyword :)

        // the ordinary DisjointSet's merge:
        x = this->find(x); // includes a range check for x
        y = this->find(y); // includes a range check for y
        if (x == y) throw std::invalid_argument("find(x) == find(y)");
        if (y < x) std::swap(x, y);

        this->par[y] = x; // update the parent of y
        this->k -= 1;     // decreaset the subset count

        // update the counts
        ulonglong size1 = this->cnt[x];
        ulonglong size2 = this->cnt[y];
        ulonglong size12 = size1+size2;
        this->cnt[x] += this->cnt[y]; // cluster x has more elements now
        this->cnt[y] = 0;             // cluster y, well, cleaning up

        this->tab[size1]  -= 1; // one cluster of size1 is no more
        this->tab[size2]  -= 1; // one cluster of size2 is an ex-cluster
        this->tab[size12] += 1; // long live cluster of size1+2


        // update the doubly-linked list for accessing non-zero elems in this->tab
        // first, the element corresponding to size12
        if (this->tab_tail < size12) { // new tail
            this->tab_prev[size12] = this->tab_tail;
            this->tab_next[this->tab_tail] = size12;
            this->tab_tail = size12;
        }
        else if (this->tab[size12] == 1) { // new elem in the 'middle'
            ulonglong w = this->tab_tail;
            while (w > size12) {
                w = this->tab_prev[w];
            }
            ulonglong v = this->tab_next[w];
            this->tab_next[w] = size12;
            this->tab_prev[v] = size12;
            this->tab_next[size12] = v;
            this->tab_prev[size12] = w;
        }
        // else element already existed


        // get rid of size1 and size2, if necessary
        if (size2 < size1) std::swap(size1, size2);
        if (this->tab[size1] == 0) {
            if (this->tab_head == size1) {
                this->tab_head = this->tab_next[this->tab_head];
            }
            else { // remove in the 'middle'
                this->tab_next[this->tab_prev[size1]] = this->tab_next[size1];
                this->tab_prev[this->tab_next[size1]] = this->tab_prev[size1];
            }
        }

        if (this->tab[size2] == 0 && size1 != size2) { // i.e., size2>size1
            if (this->tab_head == size2) {
                this->tab_head = this->tab_next[this->tab_head];
            }
            else { // remove in the 'middle'
                this->tab_next[this->tab_prev[size2]] = this->tab_next[size2];
                this->tab_prev[this->tab_next[size2]] = this->tab_prev[size2];
            }
        }

        if (this->tab[this->tab_head] <= 0)
            throw std::out_of_range("ASSERT FAIL: this->tab[this->tab_head] > 0");
        if (this->tab[this->tab_tail] <= 0)
            throw std::out_of_range("ASSERT FAIL: this->tab[this->tab_tail] > 0");


        // re-compute the normalized Gini index
        // based on a formula given in @TODO:derive the formula nicely@
        this->gini = 0.0;
        if (this->tab_head != this->tab_tail) { // otherwise all clusters are of identical sizes
            ulonglong v = this->tab_head, w;
            ulonglong i = 0;
            while (v != this->tab_tail) {
                w = v;                 // previous v
                v = this->tab_next[v]; // next v
                i += this->tab[w];     // cumulative counts
                this->gini += ((double)v-w)*i*((double)this->k-i);
            }
            this->gini /= (double)(this->n*(this->k-1.0)); // this is the normalized Gini index
            if (this->gini > 1.0) this->gini = 1.0; // account for round-off errors
            if (this->gini < 0.0) this->gini = 0.0;
        }

        // all done
        return x;
    }


    /*! Generates an array of subsets' sizes.
     *  The resulting vector is ordered nondecreasingly.
     * 
     *  Run time: O(k), where k is the current number of subsets.
     */
    std::vector<ulonglong> get_counts() {
        ulonglong i = 0;
        std::vector<ulonglong> out(this->k);
        ulonglong v = this->tab_head;
        while (true) {
            // add this->tab[v] times v
            for (ulonglong j=0; j<this->tab[v]; ++j) {
                if (i >= k) throw std::out_of_range("ASSERT1 FAIL in get_counts()");
                out[i++] = v;
            }
            if (v == this->tab_tail) {
                if (i < k) throw std::out_of_range("ASSERT2 FAIL in get_counts()");
                return out;
            }
            v = this->tab_next[v];
        }
    }

};

#endif
