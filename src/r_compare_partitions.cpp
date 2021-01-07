/*  Partition Similarity Scores
 *
 *  Copyleft (C) 2018-2021, Marek Gagolewski <https://www.gagolewski.com>
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

#include "c_compare_partitions.h"
#include <Rcpp.h>
#include <algorithm>
#include <vector>
using namespace Rcpp;



/** Extract or compute the contingency matrix based on given arguments
 *
 *  @param x vector or contingency table (matrix)
 *  @param y R_NilValue or vector of size x.size() if x is not a matrix
 *  @param xc [out]
 *  @param yc [out]
 *
 *  @return flat, contiguous c_style vector representing the contingency table
 *   with xc rows and yc columns
 */
std::vector<int> get_contingency_matrix(RObject x, RObject y,
                                          ssize_t* xc, ssize_t* yc)
{
    if (Rf_isMatrix(x)) {
        if (!Rf_isNull(y))
            stop("if x is a contingency matrix, y must be NULL");
        if (!(Rf_isInteger(x) | Rf_isReal(x)))
            stop("x must be of type numeric");

        IntegerMatrix X(x);
        *xc = X.nrow();
        *yc = X.ncol();
        std::vector<int> C((*xc)*(*yc));
        ssize_t k=0;
        for (ssize_t i=0; i<*xc; ++i)
            for (ssize_t j=0; j<*yc; ++j)
                C[k++] = X(i, j); // Fortran -> C-style
        return C;
    }
    else {
        if (Rf_isNull(y))
            stop("if x is not a contingency matrix, y must not be NULL");
        if (!(Rf_isInteger(x) | Rf_isReal(x) | Rf_isLogical(x) | Rf_isFactor(x)))
            stop("x must be of type numeric");
        if (!(Rf_isInteger(x) | Rf_isReal(x) | Rf_isLogical(x) | Rf_isFactor(x)))
            stop("y must be of type numeric");

        IntegerVector rx(x);
        IntegerVector ry(y);

        ssize_t n = rx.size();
        if (ry.size() != n)
            stop("x and y must be of equal lengths");

        for (ssize_t i=0; i<n; ++i) {
            if (rx[i] == NA_INTEGER || ry[i] == NA_INTEGER)
                stop("missing values not allowed");
        }

        int xmin, xmax;
        Cminmax(INTEGER(SEXP(rx)), n, &xmin, &xmax);
        *xc = (xmax-xmin+1);

        int ymin, ymax;
        Cminmax(INTEGER(SEXP(ry)), n, &ymin, &ymax);
        *yc = (ymax-ymin+1);

        std::vector<int> C((*xc)*(*yc));
        Ccontingency_table(C.data(), *xc, *yc,
            xmin, ymin, INTEGER(SEXP(rx)), INTEGER(SEXP(ry)), n);
        return C;
    }
}



//' @title Pairwise Partition Similarity Scores
//'
//' @description
//' Let \code{x} and \code{y} represent two partitions of a set of \code{n}
//' elements into \eqn{K} and \eqn{L}, respectively,
//' nonempty and pairwise disjoint subsets,
//' e.g., two clusterings of a dataset with \code{n} observations
//' represented as label vectors.
//' These functions quantify the similarity between \code{x}
//' and \code{y}. They can be used as external cluster
//' validity measures, i.e., in the presence of reference (ground-truth)
//' partitions.
//'
//' @details
//' Every index except \code{mi_score()} (which computes the mutual
//' information score) outputs 1 given two identical partitions.
//' Note that partitions are always defined up to a bijection of the set of
//' possible labels, e.g., (1, 1, 2, 1) and (4, 4, 2, 4)
//' represent the same 2-partition.
//'
//' \code{rand_score()} gives the Rand score (the "probability" of agreement
//' between the two partitions) and
//' \code{adjusted_rand_score()} is its version corrected for chance,
//' see (Hubert, Arabie, 1985),
//' its expected value is 0.0 given two independent partitions.
//' Due to the adjustment, the resulting index might also be negative
//' for some inputs.
//'
//' Similarly, \code{fm_score()} gives the Fowlkes-Mallows (FM) score
//' and \code{adjusted_fm_score()} is its adjusted-for-chance version,
//' see (Hubert, Arabie, 1985).
//'
//' Note that both the (unadjusted) Rand and FM scores are bounded from below
//' by \eqn{1/(K+1)} if \eqn{K=L}, hence their adjusted versions are preferred.
//'
//' \code{mi_score()}, \code{adjusted_mi_score()} and
//' \code{normalized_mi_score()} are information-theoretic
//' scores, based on mutual information,
//' see the definition of \eqn{AMI_{sum}} and \eqn{NMI_{sum}}
//' in (Vinh et al., 2010).
//'
//' \code{normalized_accuracy()} is defined as
//' \eqn{(Accuracy(C_\sigma)-1/L)/(1-1/L)}, where \eqn{C_\sigma} is a version
//' of the confusion matrix for given \code{x} and \code{y},
//' \eqn{K \leq L}, with columns permuted based on the solution to the
//' Maximal Linear Sum Assignment Problem.
//' \eqn{Accuracy(C_\sigma)} is sometimes referred to as Purity,
//' e.g., in (Rendon et al. 2011).
//'
//' \code{pair_sets_index()} gives the  Pair Sets Index (PSI)
//' adjusted for chance (Rezaei, Franti, 2016), \eqn{K \leq L}.
//' Pairing is based on the solution to the Linear Sum Assignment Problem
//' of a transformed version of the confusion matrix.
//'
//' @references
//' Hubert L., Arabie P., Comparing Partitions,
//' Journal of Classification 2(1), 1985, 193-218, esp. Eqs. (2) and (4).
//'
//' Rendon E., Abundez I., Arizmendi A., Quiroz E.M.,
//' Internal versus external cluster validation indexes,
//' International Journal of Computers and Communications 5(1), 2011, 27-34.
//'
//' Rezaei M., Franti P., Set matching measures for external cluster validity,
//' IEEE Transactions on Knowledge and Data Mining 28(8), 2016, 2173-2186.
//'
//' Vinh N.X., Epps J., Bailey J.,
//' Information theoretic measures for clusterings comparison:
//' Variants, properties, normalization and correction for chance,
//' Journal of Machine Learning Research 11, 2010, 2837-2854.
//'
//'
//' @param x an integer vector of length n (or an object coercible to)
//' representing a K-partition of an n-set,
//' or a confusion matrix with K rows and L columns (see \code{table(x, y)})
//'
//' @param y an integer vector of length n (or an object coercible to)
//' representing an L-partition of the same set),
//' or NULL (if x is an K*L confusion matrix)
//'
//' @return A single real value giving the similarity score.
//'
//' @examples
//' y_true <- iris[[5]]
//' y_pred <- kmeans(as.matrix(iris[1:4]), 3)$cluster
//' adjusted_rand_score(y_true, y_pred)
//' rand_score(table(y_true, y_pred)) # the same
//' adjusted_fm_score(y_true, y_pred)
//' fm_score(y_true, y_pred)
//' mi_score(y_true, y_pred)
//' normalized_mi_score(y_true, y_pred)
//' adjusted_mi_score(y_true, y_pred)
//' normalized_accuracy(y_true, y_pred)
//' pair_sets_index(y_true, y_pred)
//'
//' @rdname comparing_partitions
//' @export
//[[Rcpp::export]]
double adjusted_rand_score(RObject x, RObject y=R_NilValue)
{
    ssize_t xc, yc;
    std::vector<int> C(
        get_contingency_matrix(x, y, &xc, &yc)
    );

    return Ccompare_partitions_pairs(C.data(), xc, yc).ar;
}


//' @rdname comparing_partitions
//' @export
//[[Rcpp::export]]
double rand_score(RObject x, RObject y=R_NilValue)
{
    ssize_t xc, yc;
    std::vector<int> C(
        get_contingency_matrix(x, y, &xc, &yc)
    );

    return Ccompare_partitions_pairs(C.data(), xc, yc).r;
}


//' @rdname comparing_partitions
//' @export
//[[Rcpp::export]]
double adjusted_fm_score(RObject x, RObject y=R_NilValue)
{
    ssize_t xc, yc;
    std::vector<int> C(
        get_contingency_matrix(x, y, &xc, &yc)
    );

    return Ccompare_partitions_pairs(C.data(), xc, yc).afm;
}


//' @rdname comparing_partitions
//' @export
//[[Rcpp::export]]
double fm_score(RObject x, RObject y=R_NilValue)
{
    ssize_t xc, yc;
    std::vector<int> C(
        get_contingency_matrix(x, y, &xc, &yc)
    );

    return Ccompare_partitions_pairs(C.data(), xc, yc).fm;
}


//' @rdname comparing_partitions
//' @export
//[[Rcpp::export]]
double mi_score(RObject x, RObject y=R_NilValue)
{
    ssize_t xc, yc;
    std::vector<int> C(
        get_contingency_matrix(x, y, &xc, &yc)
    );

    return Ccompare_partitions_info(C.data(), xc, yc).mi;
}



//' @rdname comparing_partitions
//' @export
//[[Rcpp::export]]
double normalized_mi_score(RObject x, RObject y=R_NilValue)
{
    ssize_t xc, yc;
    std::vector<int> C(
        get_contingency_matrix(x, y, &xc, &yc)
    );

    return Ccompare_partitions_info(C.data(), xc, yc).nmi;
}



//' @rdname comparing_partitions
//' @export
//[[Rcpp::export]]
double adjusted_mi_score(RObject x, RObject y=R_NilValue)
{
    ssize_t xc, yc;
    std::vector<int> C(
        get_contingency_matrix(x, y, &xc, &yc)
    );

    return Ccompare_partitions_info(C.data(), xc, yc).ami;
}



//' @rdname comparing_partitions
//' @export
//[[Rcpp::export]]
double normalized_accuracy(RObject x, RObject y=R_NilValue)
{
    ssize_t xc, yc;
    std::vector<int> C(
        get_contingency_matrix(x, y, &xc, &yc)
    );

    return Ccompare_partitions_nacc(C.data(), xc, yc);
}


//' @rdname comparing_partitions
//' @export
//[[Rcpp::export]]
double pair_sets_index(RObject x, RObject y=R_NilValue)
{
    ssize_t xc, yc;
    std::vector<int> C(
        get_contingency_matrix(x, y, &xc, &yc)
    );

    return Ccompare_partitions_psi(C.data(), xc, yc);
}


