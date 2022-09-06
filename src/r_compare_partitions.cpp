/*  Partition Similarity Scores
 *
 *  Copyleft (C) 2018-2022, Marek Gagolewski <https://www.gagolewski.com>
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



//' @title External Cluster Validity Measures and Pairwise Partition Similarity Scores
//'
//' @description
//' The functions described in this section quantify the similarity between
//' two label vectors \code{x} and \code{y} which represent two partitions
//' of a set of \eqn{n} elements into, respectively, \eqn{K} and \eqn{L}
//' nonempty and pairwise disjoint subsets.
//'
//' For instance, \code{x} and \code{y} can be two clusterings
//' of a dataset with \eqn{n} observations specified by two vectors
//' of labels. These functions can be used as external cluster
//' validity measures, where we assume that \code{x} is
//' the reference (ground-truth) partition (compare Gagolewski, 2022).
//'
//' @details
//' Each index except \code{adjusted_asymmetric_accuracy()}
//' can act as a pairwise partition similarity score: it is symmetric,
//' i.e., \code{index(x, y) == index(y, x)}.
//'
//' Each index except \code{mi_score()} (which computes the mutual
//' information score) outputs 1 given two identical partitions.
//' Note that partitions are always defined up to a bijection of the set of
//' possible labels, e.g., (1, 1, 2, 1) and (4, 4, 2, 4)
//' represent the same 2-partition.
//'
//'
//' \code{adjusted_asymmetric_accuracy()} (Gagolewski, 2022)
//' only accepts \eqn{K = L}. It is an external cluster validity measure
//' which assumes that the label vector \code{x} (or rows in the confusion
//' matrix) represents the reference (ground truth) partition.
//' It is a corrected-for-chance summary of the proportion of correctly
//' classified points in each cluster (with cluster matching based on the
//' solution to the maximal linear sum assignment problem;
//' see \code{\link{normalized_confusion_matrix}}), given by:
//' \eqn{(\max_\sigma \sum_{i=1}^K (c_{i, \sigma(i)}/(c_{i, 1}+...+c_{i, K})) - 1)/(K - 1)},
//' where \eqn{C} is the confusion matrix.
//'
//' \code{normalized_accuracy()} is defined as
//' \eqn{(Accuracy(C_\sigma)-1/L)/(1-1/L)}, where \eqn{C_\sigma} is a version
//' of the confusion matrix for given \code{x} and \code{y},
//' \eqn{K \leq L}, with columns permuted based on the solution to the
//' maximal linear sum assignment problem.
//' The \eqn{Accuracy(C_\sigma)} part is sometimes referred to as
//' set-matching classification rate or pivoted accuracy.
//'
//' \code{pair_sets_index()} gives the Pair Sets Index (PSI)
//' adjusted for chance (Rezaei, Franti, 2016), \eqn{K \leq L}.
//' Pairing is based on the solution to the linear sum assignment problem
//' of a transformed version of the confusion matrix.
//' Its simplified version assumes E=1 in the definition of the index,
//' i.e., uses Eq. (20) instead of (18).
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
//'
//' \code{normalized_confusion_matrix()} computes the confusion matrix
//' and permutes its rows and columns so that the sum of the elements
//' of the main diagonal is the largest possible (by solving
//' the maximal assignment problem).
//' The function only accepts \eqn{K \leq L}.
//' The sole reordering of the columns of a confusion matrix can be determined
//' by calling \code{normalizing_permutation()}.
//'
//' Also note that the built-in
//' \code{\link{table}()} determines the standard confusion matrix.
//'
//'
//' @references
//' Gagolewski M., \emph{A Framework for Benchmarking Clustering Algorithms},
//' 2022, \url{https://clustering-benchmarks.gagolewski.com}.
//'
//' Gagolewski M., Adjusted asymmetric accuracy: A well-behaving external
//' cluster validity measure, 2022, submitted for publication.
//'
//' Hubert L., Arabie P., Comparing partitions,
//' \emph{Journal of Classification} 2(1), 1985, 193-218, esp. Eqs. (2) and (4).
//'
//' Meila M., Heckerman D., An experimental comparison of model-based clustering
//' methods, \emph{Machine Learning} 42, 2001, pp. 9-29,
//' \doi{10.1023/A:1007648401407}.
//'
//' Rezaei M., Franti P., Set matching measures for external cluster validity,
//' \emph{IEEE Transactions on Knowledge and Data Mining} 28(8), 2016,
//' 2173-2186.
//'
//' Steinley D., Properties of the Hubert-Arabie adjusted Rand index,
//' \emph{Psychological Methods} 9(3), 2004, pp. 386-396,
//' \doi{10.1037/1082-989X.9.3.386}.
//'
//' Vinh N.X., Epps J., Bailey J.,
//' Information theoretic measures for clusterings comparison:
//' Variants, properties, normalization and correction for chance,
//' \emph{Journal of Machine Learning Research} 11, 2010, 2837-2854.
//'
//'
//'
//' @param x an integer vector of length n (or an object coercible to)
//' representing a K-partition of an n-set (e.g., a reference partition),
//' or a confusion matrix with K rows and L columns
//' (see \code{\link{table}(x, y)})
//'
//' @param y an integer vector of length n (or an object coercible to)
//' representing an L-partition of the same set (e.g., the output of a
//' clustering algorithm we wish to compare with \code{x}),
//' or NULL (if x is an K*L confusion matrix)
//'
//' @param simplified whether to assume E=1 in the definition of the pair sets index index,
//'     i.e., use Eq. (20) instead of (18); see (Rezaei, Franti, 2016).
//'
//'
//' @return Each cluster validity measure is a single numeric value.
//'
//' \code{normalized_confusion_matrix()} returns an integer matrix.
//'
//'
//' @examples
//' y_true <- iris[[5]]
//' y_pred <- kmeans(as.matrix(iris[1:4]), 3)$cluster
//' adjusted_asymmetric_accuracy(y_true, y_pred)
//' normalized_accuracy(y_true, y_pred)
//' pair_sets_index(y_true, y_pred)
//' pair_sets_index(y_true, y_pred, simplified=TRUE)
//' adjusted_rand_score(y_true, y_pred)
//' rand_score(table(y_true, y_pred)) # the same
//' adjusted_fm_score(y_true, y_pred)
//' fm_score(y_true, y_pred)
//' mi_score(y_true, y_pred)
//' normalized_mi_score(y_true, y_pred)
//' adjusted_mi_score(y_true, y_pred)
//' normalized_confusion_matrix(y_true, y_pred)
//' normalizing_permutation(y_true, y_pred)
//'
//' @rdname compare_partitions
//' @name compare_partitions
//' @export
//[[Rcpp::export]]
double adjusted_asymmetric_accuracy(RObject x, RObject y=R_NilValue)
{
    ssize_t xc, yc;
    std::vector<int> C(
        get_contingency_matrix(x, y, &xc, &yc)
    );

    return Ccompare_partitions_aaa(C.data(), xc, yc);
}


//' @rdname compare_partitions
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


//' @rdname compare_partitions
//' @export
//[[Rcpp::export]]
double pair_sets_index(RObject x, RObject y=R_NilValue, bool simplified=false)
{
    ssize_t xc, yc;
    std::vector<int> C(
        get_contingency_matrix(x, y, &xc, &yc)
    );

    if (simplified)
        return Ccompare_partitions_psi(C.data(), xc, yc).spsi;
    else
        return Ccompare_partitions_psi(C.data(), xc, yc).psi;
}


//' @rdname compare_partitions
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


//' @rdname compare_partitions
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


//' @rdname compare_partitions
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


//' @rdname compare_partitions
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


//' @rdname compare_partitions
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



//' @rdname compare_partitions
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



//' @rdname compare_partitions
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



//' @rdname compare_partitions
//' @export
//[[Rcpp::export]]
IntegerMatrix normalized_confusion_matrix(RObject x, RObject y=R_NilValue)
{
    ssize_t xc, yc;
    std::vector<int> C(
        get_contingency_matrix(x, y, &xc, &yc)
    );

    std::vector<int> C_out_Corder(xc*yc);
    Capply_pivoting(C.data(), xc, yc, C_out_Corder.data());

    IntegerMatrix Cout(xc, yc);
    for (ssize_t i=0; i<xc; ++i)  // make Fortran order
            for (ssize_t j=0; j<yc; ++j)
                Cout(i, j) = C_out_Corder[j+i*yc];
    return Cout;
}



//' @rdname compare_partitions
//' @export
//[[Rcpp::export]]
IntegerVector normalizing_permutation(RObject x, RObject y=R_NilValue)
{
    ssize_t xc, yc;
    std::vector<int> C(
        get_contingency_matrix(x, y, &xc, &yc)
    );

    IntegerVector Iout(yc);

    Cnormalizing_permutation(C.data(), xc, yc, INTEGER(SEXP(Iout)));

    for (ssize_t j=0; j<yc; ++j)
        Iout[j]++; // 0-based -> 1-based

    return Iout;
}
