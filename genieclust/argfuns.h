/*
Some sort/search/vector indexing-related functions
missing in the C++ standard library, including the ones to:

a. find the (stable) ordering permutation of a vector
b. find the k-th smallest value in a vector


Copyright (C) 2018 Marek.Gagolewski.com
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef __argsort_h
#define __argsort_h

#include <stdexcept>
#include <algorithm>


template<class T>
struct __argsort_comparer {
    const T* data;
    __argsort_comparer(const T* data) { this->data = data; }
    bool operator()(size_t i, size_t j) const {
        return this->data[i] <  this->data[j] ||
              (this->data[i] == this->data[j] && i < j);
    }
};


/*
 *  Find an(*) ordering permutation of data.
 *
 *  both ret and data should be of the same length n;
 *  ret will be overwritten
 *
 *  (*) or THE stable one, if stable=true, which is the default
*/
template<class T>
void argsort(size_t* ret, const T* data, size_t n, bool stable=true) {
    if (n <= 0) throw std::domain_error("n <= 0");

    for (size_t i=0; i<n; ++i)
        ret[i] = i;

    if (stable)
        std::stable_sort(ret, ret+n, __argsort_comparer<T>(data));
    else
        std::sort(ret, ret+n, __argsort_comparer<T>(data));
}


#include <iostream>

/*
    Returns the index of the (k-1)-th smallest value in an array x,
    where argkmin(x, 0) == argmin(x), or, more generally,
    argkmin(x, k) == np.argsort(x)[k].

    Run time: O(nk), where n == len(x). Working mem: O(k).
    Does not modify x.

    In practice, very fast for small k and randomly ordered
    or almost sorted (increasingly) data.


    If buf is not NULL, it must be of length at least k+1.
*/
template<class T>
size_t argkmin(const T* x, size_t n, size_t k, size_t* buf=NULL) {
    size_t* idx;

    if (n <= 0)   throw std::domain_error("n <= 0");
    if (k >= n)   throw std::domain_error("k >= n");

    k += 1;
    if (!buf) idx = new size_t[k];
    else      idx = buf;

    for (size_t i=0; i<k; ++i) {
        size_t j = i;
        idx[i] = i;
        while (j > 0 && x[i] < x[idx[j-1]]) {
            idx[j] = idx[j-1];
            j -= 1;
        }
        idx[j] = i;
    }

    for (size_t i=k; i<n; ++i) {
        if (x[idx[k-1]] <= x[i])
            continue;
        size_t j = k-1;
        idx[k-1] = i;
        while (j > 0 && x[i] < x[idx[j-1]]) {
            idx[j] = idx[j-1];
            j -= 1;
        }
        idx[j] = i;
    }


    size_t ret = idx[k-1];

    if (!buf) delete [] idx;

    return ret;
}


#endif
