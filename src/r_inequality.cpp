/*  Inequality Measures
 *
 *  Copyleft (C) 2018-2024, Marek Gagolewski <https://www.gagolewski.com>
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

#include "c_inequality.h"
#include <Rcpp.h>
#include <algorithm>



//' @title Inequality Measures
//'
//' @description
//' \code{gini_index()} gives the normalised Gini index,
//' \code{bonferroni_index()} implements the Bonferroni index, and
//' \code{devergottini_index()} implements the De Vergottini index.
//'
//' @details
//' These indices can be used to quantify the "inequality" of a numeric sample.
//' They can be conceived as normalised measures of data dispersion.
//' For constant vectors (perfect equity), the indices yield values of 0.
//' Vectors with all elements but one equal to 0 (perfect inequality),
//' are assigned scores of 1.
//' They follow the Pigou-Dalton principle (are Schur-convex):
//' setting \eqn{x_i = x_i - h} and \eqn{x_j = x_j + h} with \eqn{h > 0}
//' and \eqn{x_i - h \geq  x_j + h} (taking from the "rich" and
//' giving to the "poor") decreases the inequality
//'
//' These indices have applications in economics, amongst others.
//' The Genie clustering algorithm uses the Gini index as a measure
//' of the inequality of cluster sizes.
//'
//'
//' The normalised Gini index is given by:
//' \deqn{
//'     G(x_1,\dots,x_n) = \frac{
//'     \sum_{i=1}^{n} (n-2i+1) x_{\sigma(n-i+1)}
//'     }{
//'     (n-1) \sum_{i=1}^n x_i
//'     },
//' }
//'
//' The normalised Bonferroni index is given by:
//' \deqn{
//'     B(x_1,\dots,x_n) = \frac{
//'     \sum_{i=1}^{n}  (n-\sum_{j=1}^i \frac{n}{n-j+1})
//'          x_{\sigma(n-i+1)}
//'     }{
//'     (n-1) \sum_{i=1}^n x_i
//'     }.
//' }
//'
//' The normalised De Vergottini index is given by:
//' \deqn{
//'     V(x_1,\dots,x_n) =
//'     \frac{1}{\sum_{i=2}^n \frac{1}{i}} \left(
//'        \frac{ \sum_{i=1}^n \left( \sum_{j=i}^{n} \frac{1}{j}\right)
//'        x_{\sigma(n-i+1)} }{\sum_{i=1}^{n} x_i} - 1
//'     \right).
//' }
//'
//' Here, \eqn{\sigma} is an ordering permutation of \eqn{(x_1,\dots,x_n)}.
//'
//' Time complexity: \eqn{O(n)} for sorted (increasingly) data.
//' Otherwise, the vector will be sorted.
//'
//'
//' @references
//' Bonferroni C., \emph{Elementi di Statistica Generale}, Libreria Seber,
//' Firenze, 1930.
//'
//' Gagolewski M., Bartoszuk M., Cena A., Genie: A new, fast, and
//' outlier-resistant hierarchical clustering algorithm,
//' \emph{Information Sciences} 363, 2016, pp. 8-23.
//' \doi{10.1016/j.ins.2016.05.003}
//'
//' Gini C., \emph{Variabilita e Mutabilita},
//' Tipografia di Paolo Cuppini, Bologna, 1912.
//'
//'
//' @param x numeric vector of non-negative values
//'
//' @return The value of the inequality index, a number in \eqn{[0, 1]}.
//'
//' @examples
//' gini_index(c(2, 2, 2, 2, 2))   # no inequality
//' gini_index(c(0, 0, 10, 0, 0))  # one has it all
//' gini_index(c(7, 0, 3, 0, 0))   # give to the poor, take away from the rich
//' gini_index(c(6, 0, 3, 1, 0))   # (a.k.a. Pigou-Dalton principle)
//' bonferroni_index(c(2, 2, 2, 2, 2))
//' bonferroni_index(c(0, 0, 10, 0, 0))
//' bonferroni_index(c(7, 0, 3, 0, 0))
//' bonferroni_index(c(6, 0, 3, 1, 0))
//' devergottini_index(c(2, 2, 2, 2, 2))
//' devergottini_index(c(0, 0, 10, 0, 0))
//' devergottini_index(c(7, 0, 3, 0, 0))
//' devergottini_index(c(6, 0, 3, 1, 0))
//'
//' @name inequality
//' @rdname inequality
//' @export
// [[Rcpp::export]]
double gini_index(Rcpp::NumericVector x)
{
    Py_ssize_t n = x.size();

    // check if sorted; if not, sort.
    for (Py_ssize_t i=1; i<n; ++i) {
        if (x[i-1] > x[i]) {
            x = Rcpp::clone(x);
            std::sort(x.begin(), x.end());
            break;
        }
    }

    return Cgini_sorted(REAL(SEXP(x)), n);
}


//' @rdname inequality
//' @export
// [[Rcpp::export]]
double bonferroni_index(Rcpp::NumericVector x)
{
    Py_ssize_t n = x.size();

    // check if sorted; if not, sort.
    for (Py_ssize_t i=1; i<n; ++i) {
        if (x[i-1] > x[i]) {
            x = Rcpp::clone(x);
            std::sort(x.begin(), x.end());
            break;
        }
    }

    return Cbonferroni_sorted(REAL(SEXP(x)), n);
}



//' @rdname inequality
//' @export
// [[Rcpp::export]]
double devergottini_index(Rcpp::NumericVector x)
{
    Py_ssize_t n = x.size();

    // check if sorted; if not, sort.
    for (Py_ssize_t i=1; i<n; ++i) {
        if (x[i-1] > x[i]) {
            x = Rcpp::clone(x);
            std::sort(x.begin(), x.end());
            break;
        }
    }

    return Cdevergottini_sorted(REAL(SEXP(x)), n);
}
