/*  Inequity (Inequality) Measures
 *
 *  Copyleft (C) 2018-2020, Marek Gagolewski <https://www.gagolewski.com>
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


#ifndef __c_inequity_h
#define __c_inequity_h

#include "c_common.h"
#include <algorithm>


/*! The Normalised Gini Index
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
 * References
 * ----------
 *
 * Gini C., Variabilita e Mutabilita, Tipografia di Paolo Cuppini,
 * Bologna, 1912.
 *
 *
 * @param x non-decreasingly sorted c_contiguous input vector >= 0
 * @param n length of x
 *
 * @return the value of the inequity index, a number in [0,1].
 */
template<class T>
double Cgini_sorted(const T* x, ssize_t n)
{
    double s = 0.0, t = 0.0;
    GENIECLUST_ASSERT(x[0] >= 0);
    GENIECLUST_ASSERT(x[n-1] > 0);
    for (ssize_t i=1; i<=n; ++i) {
        t += x[n-i];
        s += (n-2.0*i+1.0)*x[n-i];
    }
    s = s/(n-1.0)/t;
    if (s > 1.0) return 1.0;
    else if (s < 0.0) return 0.0;
    else return s;
}



/*! The Normalised Bonferroni Index
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
 * References
 * ----------
 *
 * Bonferroni C., Elementi di Statistica Generale, Libreria Seber,
 * Firenze, 1930.
 *
 * @param x non-decreasingly sorted c_contiguous input vector >= 0
 * @param n length of x
 *
 * @return the value of the inequity index, a number in [0,1].
 */
template<class T>
double Cbonferroni_sorted(const T* x, ssize_t n)
{
    double s = 0.0, t = 0.0, c = 0.0;
    GENIECLUST_ASSERT(x[0] >= 0);
    GENIECLUST_ASSERT(x[n-1] > 0);
    for (ssize_t i=1; i<=n; ++i) {
        c += n/(n-i+1.0);
        t += x[n-i];
        s += (n-c)*x[n-i];
    }
    s = s/(n-1.0)/t;
    if (s > 1.0) return 1.0;
    else if (s < 0.0) return 0.0;
    else return s;
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
