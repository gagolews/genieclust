/*  Inequality Measures
 *
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


#ifndef __c_inequality_h
#define __c_inequality_h

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
 * Gini, C., Variabilita e Mutabilita, Tipografia di Paolo Cuppini,
 * Bologna, 1912.
 *
 *
 * @param x non-decreasingly sorted c_contiguous input vector >= 0
 * @param n length of x
 *
 * @return the value of the inequality index, a number in [0,1].
 */
template<class T>
double Cgini_sorted(const T* x, Py_ssize_t n)
{
    double s = 0.0, t = 0.0;
    GENIECLUST_ASSERT(x[0] >= 0);
    GENIECLUST_ASSERT(x[n-1] > 0);
    for (Py_ssize_t i=1; i<=n; ++i) {
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
 * Bonferroni, C., Elementi di Statistica Generale, Libreria Seber,
 * Firenze, 1930.
 *
 * @param x non-decreasingly sorted c_contiguous input vector >= 0
 * @param n length of x
 *
 * @return the value of the inequality index, a number in [0,1].
 */
template<class T>
double Cbonferroni_sorted(const T* x, Py_ssize_t n)
{
    double s = 0.0, t = 0.0, c = 0.0;
    GENIECLUST_ASSERT(x[0] >= 0);
    GENIECLUST_ASSERT(x[n-1] > 0);
    for (Py_ssize_t i=1; i<=n; ++i) {
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
//     #Is this an inequality measures BTW?
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


/*! The Normalised De Vergottini Index
 *
 * The normalised De Vergottini index is given by:
 * $$
 *     V(x_1,\dots,x_n) = \frac{1}{\sum_{i=2}^n \frac{1}{i}} \left(
 *    \frac{ \sum_{i=1}^n \left( \sum_{j=i}^{n} \frac{1}{j}\right)
 *       x_{\sigma(n-i+1)} }{\sum_{i=1}^{n} x_i} - 1
 * \right)
 * $$
 * where $\sigma$ is an ordering permutation of $(x_1,\dots,x_n)$.
 *
 * Time complexity: $O(n)$ for sorted data.
 *
 *
 *
 *
 * @param x non-decreasingly sorted c_contiguous input vector >= 0
 * @param n length of x
 *
 * @return the value of the inequality index, a number in [0,1].
 */
template<class T>
double Cdevergottini_sorted(const T* x, Py_ssize_t n)
{
    double s = 0.0, t = 0.0, c = 0.0, f=0.0, d=0.0;
    GENIECLUST_ASSERT(x[0] >= 0);
    GENIECLUST_ASSERT(x[n-1] > 0);

    for (Py_ssize_t i=2; i<=n; ++i)
        c += 1.0/(double)i;

    for (Py_ssize_t i=1; i<=n; ++i) {
        t += x[i-1];
        f += 1.0/(double)(n-i+1);
        d += f*x[i-1];  // the i-th smallest
    }

    s = (d/t-1.0)/c;
    if (s > 1.0) return 1.0;
    else if (s < 0.0) return 0.0;
    else return s;
}


#endif
