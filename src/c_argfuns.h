/*  Some sort/search/vector indexing-related functions
 *  missing in the Standard Library, including the ones to:
 *  a. find the (stable) ordering permutation of a vector
 *  b. find the k-th smallest value in a vector
 *
 *  Copyright (C) 2018-2020 Marek Gagolewski (https://www.gagolewski.com)
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


#ifndef __c_argsort_h
#define __c_argsort_h

#include <stdexcept>
#include <algorithm>



/*! Comparer for argsort().
 *
 *  Assures the resulting permutation is stable.
 */
template<class T>
struct __argsort_comparer {
    const T* x;
    __argsort_comparer(const T* x) { this->x = x; }
    bool operator()(ssize_t i, ssize_t j) const {
        return this->x[i] <  this->x[j] ||
              (this->x[i] == this->x[j] && i < j);
    }
};


/*! Finds an(*) ordering permutation w.r.t. <.
 *
 *  Both ret and x should be of the same length n;
 *  ret will be overwritten.
 *
 *  (*) or THE stable one, if stable=true, which is the default.
 *
 *  We call permutation o stable, whenever i<j and x[i]==x[j]
 *  implies that o[i]<o[j].
 *
 *  @param ret return array
 *  @param x array to order
 *  @param n size of ret and x
 *  @param stable use a stable sorting algorithm? (slower)
 */
template<class T>
void Cargsort(ssize_t* ret, const T* x, ssize_t n, bool stable=true)
{
    if (n <= 0) throw std::domain_error("n <= 0");

    for (ssize_t i=0; i<n; ++i)
        ret[i] = i;

    if (stable)
        std::stable_sort(ret, ret+n, __argsort_comparer<T>(x));
    else
        std::sort(ret, ret+n, __argsort_comparer<T>(x));
}




/*! Returns the index of the (k-1)-th smallest value in an array x.
 *
 *  argkmin(x, 0) == argmin(x), or, more generally,
 *  argkmin(x, k) == np.argsort(x)[k].
 *
 *  Run time: O(nk), where n == len(x). Working mem: O(k).
 *  Does not modify x.
 *
 *  In practice, very fast for small k and randomly ordered
 *  or almost sorted (increasingly) data.
 *
 *
 *  If buf is not NULL, it must be of length at least k+1.
 *
 *  @param x data
 *  @param n length of x
 *  @param k value in {0,...,n-1}, preferably small
 *  @param buf optional working buffer of size >= k+1, will be overwritten
 */
template<class T>
ssize_t Cargkmin(const T* x, ssize_t n, ssize_t k, ssize_t* buf=NULL)
{
    ssize_t* idx;

    if (n <= 0)   throw std::domain_error("n <= 0");
    if (k >= n)   throw std::domain_error("k >= n");

    k += 1;
    if (!buf) idx = new ssize_t[k];
    else      idx = buf;

    for (ssize_t i=0; i<k; ++i) {
        ssize_t j = i;
        idx[i] = i;
        while (j > 0 && x[i] < x[idx[j-1]]) {
            idx[j] = idx[j-1];
            j -= 1;
        }
        idx[j] = i;
    }

    for (ssize_t i=k; i<n; ++i) {
        if (x[idx[k-1]] <= x[i])
            continue;
        ssize_t j = k-1;
        idx[k-1] = i;
        while (j > 0 && x[i] < x[idx[j-1]]) {
            idx[j] = idx[j-1];
            j -= 1;
        }
        idx[j] = i;
    }


    ssize_t ret = idx[k-1];

    if (!buf) delete [] idx;

    return ret;
}


#endif
