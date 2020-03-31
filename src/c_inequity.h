/*  Economic Inequity (Inequality) Measures.
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


#ifndef __c_inequity_h
#define __c_inequity_h

#include "c_common.h"
#include <algorithm>


/*! The normalised Gini index
 *
 * The normalised  Gini index is given by:
 * $$
 *     G(x_1,\dots,x_n) = \frac{
 *     \sum_{i=1}^{n-1} \sum_{j=i+1}^n |x_i-x_j|
 *     }{
 *     (n-1) \sum_{i=1}^n x_i
 *     }.
 * $$
 *
 * Time complexity: $O(n)$ for sorted data; it holds:
 * $$
 *     G(x_1,\dots,x_n) = \frac{
 *     \sum_{i=1}^{n} (n-2i+1) x_{\sigma(n-i+1)}
 *     }{
 *     (n-1) \sum_{i=1}^n x_i
 *     },
 * $$
 * where $\sigma$ is an ordering permutation of $(x_1,\dots,x_n)$.
 *
 *
 * @param x non-decreasingly sorted c_contiguous input vector
 * @param n length of x
 *
 * @return the value of the inequity index, a number in [0,1].
 */
template<class T>
double Cgini_sorted(const T* x, ssize_t n)
{
    double s = 0.0, t = 0.0;
    for (ssize_t i=1; i<=n; ++i) {
        t += x[n-i];
        s += (n-2.0*i+1.0)*x[n-i];
    }
    return s/(n-1.0)/t;
}



/*! The normalised Bonferroni index
 *
 * The normalised Bonferroni index is given by:
 * $$
 *     B(x_1,\dots,x_n) = \frac{
 *     \sum_{i=1}^{n}  \left( n-\sum_{j=1}^i \frac{n}{n-j+1} \right)
 *          x_{\sigma(n-i+1)}
 *     }{
 *     (n-1) \sum_{i=1}^n x_i
 *     },
 * $$
 * where $\sigma$ is an ordering permutation of $(x_1,\dots,x_n)$.
 *
 * Time complexity: $O(n)$ for sorted data.
 *
 *
 * @param x non-decreasingly sorted c_contiguous input vector
 * @param n length of x
 *
 * @return the value of the inequity index, a number in [0,1].
 */
template<class T>
double Cbonferroni_sorted(const T* x, ssize_t n)
{
    double s = 0.0, t = 0.0, c = 0.0;
    for (ssize_t i=1; i<=n; ++i) {
        c += n/(n-i+1.0);
        t += x[n-i];
        s += (n-c)*x[n-i];
    }
    return s/(n-1.0)/t;
}

// #cpdef np.float64_t coefvar(np.ndarray[T] x, bint is_sorted=False):
//     #"""
//     #Coefficient of variation
//
//         #$$
//         #C(x_1,\dots,x_n) = \sqrt{\frac{
//         #\sum_{i=1}^{n-1} \sum_{j=i+1}^n (x_i-x_j)^2
//         #}{
//         #(n-1) \sum_{i=1}^n x_i^2
//         #}}.
//         #$$
//
//     #Is this an inequity measures BTW?
//     #"""
//
//     ## sorting is not necessary
//     #cdef unsigned int n = len(x)
//     #cdef np.float64_t s = 0.0, t = square(x[0])
//     #cdef unsigned int i, j
//
//     #for i in range(n-1):
//         #t += square(x[i+1])
//         #for j in range(i+1, n):
//             #s += square(x[i]-x[j])
//
//     #return cmath.sqrt(s/(n-1.0)/t)
//
//
// # cpdef np.float64_t vergottini(np.ndarray[T] x, bint is_sorted=False):
// # "de Vergottini index
// #    x <- sort(x, decreasing=TRUE)
// #    n <- length(x)
// #    vmax <- sum(1/(2:n))
// #    (sum(sapply(1:length(x), function(i) mean(x[1:i])))/sum(x)-1)/vmax
// # }


#endif
