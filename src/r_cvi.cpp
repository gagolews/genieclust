/*  Rcpp exports - Internal cluster validity indices
 *
 *  Code originally contributed in <https://github.com/gagolews/optim_cvi>,
 *  see https://doi.org/10.1016/j.ins.2021.10.004.
 *
 *  Copyleft (C) 2020-2022, Marek Gagolewski <https://www.gagolewski.com>
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


#ifndef R_CVI_INTERNAL_H
#define R_CVI_INTERNAL_H

#include <string>
#include <cstring>
#include <Rcpp.h>
#include <vector>

#include "c_common.h"
#include "c_matrix.h"
#include "cvi.h"
#include "cvi_calinski_harabasz.h"
#include "cvi_davies_bouldin.h"
#include "cvi_silhouette.h"
// #include "cvi_dunn.h"
// #include "cvi_gamma.h"
#include "cvi_wcss.h"
#include "cvi_wcnn.h"
#include "cvi_dunnowa.h"
#include "cvi_generalized_dunn.h"
#include "cvi_generalized_dunn_lowercase_d1.h"
#include "cvi_generalized_dunn_lowercase_d2.h"
#include "cvi_generalized_dunn_lowercase_d3.h"
#include "cvi_generalized_dunn_lowercase_d4.h"
#include "cvi_generalized_dunn_lowercase_d5.h"
#include "cvi_generalized_dunn_lowercase_d6.h"
#include "cvi_generalized_dunn_uppercase_d1.h"
#include "cvi_generalized_dunn_uppercase_d2.h"
#include "cvi_generalized_dunn_uppercase_d3.h"
using namespace Rcpp;



/** Converts a 1-based label vector to a 0-based vector of small integers.
 *
 * @param x numeric vector with integer elements in [1, 256].
 *
 * @return vector
 */
std::vector<uint8_t> translateLabels_fromR(const Rcpp::NumericVector& x)
{
    size_t n = x.size();
    std::vector<uint8_t> ret(n);
    for (size_t i=0; i<n; ++i) {
        int xi = (int)x[i];
        GENIECLUST_ASSERT(xi >= 1 && xi <= 256)
        ret[i] = (uint8_t)(xi-1); // 1-based -> 0-based
    }
    return ret;
}


/** Converts a 0-based label vector to a 1-based one.
 *
 * @param x
 *
 * @return R's numeric vector
 */
Rcpp::NumericVector translateLabels_toR(const std::vector<uint8_t>& x)
{
    size_t n = x.size();
    Rcpp::NumericVector ret(n);
    for (size_t i=0; i<n; ++i)
        ret[i] = (x[i]+1); // 1-based -> 0-based
    return ret;
}


/** Convert Rcpp's numeric matrix object (column-major) to our
 * internal type (row-major).
 *
 * @param X
 *
 * @return matrix (internal type)
 */
matrix<FLOAT_T> translateMatrix_fromR(const Rcpp::NumericMatrix& X)
{
//     size_t n = X.nrow();
//     size_t d = X.ncol();
//     matrix<FLOAT_T> Y(n, d);
//     for (size_t i=0; i<n; i++) {
//         for (size_t j=0; j<d; j++) {
//                 Y(i, j) = X(i, j);
//         }
//     }
//     return Y;
    return matrix<FLOAT_T>(REAL(SEXP(X)), X.nrow(), X.ncol(), false);
}




//' @title Internal Cluster Validity Measures
//'
//' @description
//' Implements a number of cluster validity indices critically
//' reviewed in (Gagolewski, Bartoszuk, Cena, 2021). See Section 2
//' therein for the respective definitions.
//'
//' The greater the index value, the more \emph{valid} (whatever that means)
//' the assessed partition. For consistency, the Ball-Hall and
//' Davies-Bouldin indexes take negative values.
//'
//'
//' @param X numeric matrix with \code{n} rows and \code{d} columns,
//'     representing \code{n} points in a \code{d}-dimensional space
//'
//' @param y vector of \code{n} integer labels with elements in \eqn{[1, K]},
//'     representing a partition whose \emph{quality} is to be
//'     assessed; \code{y[i]} is the cluster ID of the \code{i}-th point,
//'     \code{X[i, ]}
//'
//' @param K number of clusters, equal to \code{max(y)}
//'
//' @param M number of nearest neighbours
//'
//' @param lowercase_delta an integer between 1 and 6, denoting
//'     \eqn{d_1}, ..., \eqn{d_6} in the definition
//'     of the generalised Dunn index (numerator)
//'
//' @param uppercase_delta an integer between 1 and 3, denoting
//'     \eqn{D_1}, ..., \eqn{D_3} in the definition
//'     of the generalised Dunn index (denominator)
//'
//' @param owa_numerator,owa_denominator single string defining
//'     the OWA operator to use in the definition of the DuNN index;
//'     one of: \code{"Mean"}, \code{"Min"}, \code{"Max"}, \code{"Const"},
//'     \code{"SMin:M"}, \code{"SMax:M"}, where \code{M} is an integer
//'     defining the number of nearest neighbours.
//'
//' @return A single numeric value (the more, the \emph{better}).
//'
//' @references
//' G.H. Ball, D.J. Hall,
//' ISODATA: A novel method of data analysis and pattern classification,
//' Technical report No. AD699616, Stanford Research Institute, 1965.
//'
//' J. Bezdek, N. Pal, Some new indexes of cluster validity,
//' IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics) 28
//' 1998, pp. 301--315, \doi{10.1109/3477.678624}.
//'
//' T. Calinski, J. Harabasz. A dendrite method for cluster analysis,
//' Communications in Statistics, 3(1), 1974, pp. 1-27,
//' \doi{10.1080/03610927408827101}.
//'
//' D.L. Davies, D.W. Bouldin,
//' A Cluster Separation Measure,
//' IEEE Transactions on Pattern Analysis and Machine Intelligence. PAMI-1 (2),
//' 1979, pp. 224-227, \doi{10.1109/TPAMI.1979.4766909}.
//'
//' J.C. Dunn, A Fuzzy Relative of the ISODATA Process and Its Use in Detecting
//' Compact Well-Separated Clusters, Journal of Cybernetics 3(3), 1973,
//' pp. 32-57, \doi{10.1080/01969727308546046}.
//'
//' M. Gagolewski, M. Bartoszuk, A. Cena,
//' Are cluster validity measures (in)valid?, Information Sciences 581,
//' 620–636, 2021, \doi{10.1016/j.ins.2021.10.004};
//' preprint: \url{https://raw.githubusercontent.com/gagolews/bibliography/master/preprints/2021cvi.pdf}.
//'
//' P.J. Rousseeuw, Silhouettes: A Graphical Aid to the Interpretation and
//' Validation of Cluster Analysis, Computational and Applied Mathematics 20,
//' 1987, pp. 53-65, \doi{10.1016/0377-0427(87)90125-7}.
//'
//'
//'
//' @examples
//' X <- as.matrix(iris[,1:4])
//' X[,] <- jitter(X)  # otherwise we get a non-unique solution
//' y <- as.integer(iris[[5]])
//' calinski_harabasz_index(X, y, max(y))
//' calinski_harabasz_index(X, sample(1:3, nrow(X), replace=TRUE), max(y))
//'
//' @name cluster_validity_measures
//' @rdname cluster_validity_measures
//' @export
// [[Rcpp::export]]
double calinski_harabasz_index(NumericMatrix X, NumericVector y, int K)
{
    CalinskiHarabaszIndex ind(
        matrix<FLOAT_T>(REAL(SEXP(X)), X.nrow(), X.ncol(), false),
        (uint8_t)K
    );
    ind.set_labels(translateLabels_fromR(y));
    return (double)ind.compute();
}


//' @rdname cluster_validity_measures
//' @export
// [[Rcpp::export]]
double dunnowa_index(NumericMatrix X, NumericVector y, int K, int M=10,
                Rcpp::String owa_numerator="Min",
                Rcpp::String owa_denominator="Max")
{
    GENIECLUST_ASSERT(M>0);    // M = min(n-1, M) in the constructor

    int _owa_numerator = DuNNOWA_get_OWA(std::string(owa_numerator));
    int _owa_denominator = DuNNOWA_get_OWA(std::string(owa_denominator));

    if (_owa_numerator < 0 || _owa_denominator < 0) {
        Rf_error("invalid OWA operator specifier");
    }

    DuNNOWAIndex ind(
        matrix<FLOAT_T>(REAL(SEXP(X)), X.nrow(), X.ncol(), false),
        (uint8_t)K, false, M, _owa_numerator, _owa_denominator
    );

    ind.set_labels(translateLabels_fromR(y));
    return (double)ind.compute();
}



//' @rdname cluster_validity_measures
//' @export
// [[Rcpp::export]]
double generalised_dunn_index(NumericMatrix X, NumericVector y, int K, int lowercase_delta, int uppercase_delta)
{
    LowercaseDeltaFactory* lowercase_deltaFactory;
    UppercaseDeltaFactory* uppercase_deltaFactory;

    if (lowercase_delta == 1) {
        lowercase_deltaFactory = new LowercaseDelta1Factory();
    }
    else if (lowercase_delta == 2) {
        lowercase_deltaFactory = new LowercaseDelta2Factory();
    }
    else if (lowercase_delta == 3) {
        lowercase_deltaFactory = new LowercaseDelta3Factory();
    }
    else if (lowercase_delta == 4) {
        lowercase_deltaFactory = new LowercaseDelta4Factory();
    }
    else if (lowercase_delta == 5) {
        lowercase_deltaFactory = new LowercaseDelta5Factory();
    }
    else if (lowercase_delta == 6) {
        lowercase_deltaFactory = new LowercaseDelta6Factory();
    }
    else {
        Rf_error("invalid lowercase_delta");
    }

    if (uppercase_delta == 1) {
        uppercase_deltaFactory = new UppercaseDelta1Factory();
    }
    else if (uppercase_delta == 2) {
        uppercase_deltaFactory = new UppercaseDelta2Factory();
    }
    else if (uppercase_delta == 3) {
        uppercase_deltaFactory = new UppercaseDelta3Factory();
    }
    else {
        Rf_error("invalid uppercase_delta");
    }

    bool areCentroidsNeeded = (
        lowercase_deltaFactory->IsCentroidNeeded() ||
        uppercase_deltaFactory->IsCentroidNeeded()
    );

    if (areCentroidsNeeded) {
        GeneralizedDunnIndexCentroidBased ind(
        matrix<FLOAT_T>(REAL(SEXP(X)), X.nrow(), X.ncol(), false),
        K,
        lowercase_deltaFactory,
        uppercase_deltaFactory
        );

        delete lowercase_deltaFactory;
        delete uppercase_deltaFactory;

        ind.set_labels(translateLabels_fromR(y));
        return (double)ind.compute();
    } else {
        GeneralizedDunnIndex ind(
        matrix<FLOAT_T>(REAL(SEXP(X)), X.nrow(), X.ncol(), false),
        (uint8_t)K,
        lowercase_deltaFactory,
        uppercase_deltaFactory
        );

        delete lowercase_deltaFactory;
        delete uppercase_deltaFactory;

        ind.set_labels(translateLabels_fromR(y));
        return (double)ind.compute();
    }
}


//' @rdname cluster_validity_measures
//' @export
// [[Rcpp::export]]
double negated_ball_hall_index(NumericMatrix X, NumericVector y, int K)
{
    WCSSIndex ind(
        matrix<FLOAT_T>(REAL(SEXP(X)), X.nrow(), X.ncol(), false),
        (uint8_t)K, false, true/*weighted*/
    );
    ind.set_labels(translateLabels_fromR(y));
    return (double)ind.compute();
}


//' @rdname cluster_validity_measures
//' @export
// [[Rcpp::export]]
double negated_davies_bouldin_index(NumericMatrix X, NumericVector y, int K)
{
    DaviesBouldinIndex ind(
        matrix<FLOAT_T>(REAL(SEXP(X)), X.nrow(), X.ncol(), false),
        (uint8_t)K
    );
    ind.set_labels(translateLabels_fromR(y));
    return (double)ind.compute();
}


//' @rdname cluster_validity_measures
//' @export
// [[Rcpp::export]]
double silhouette_index(NumericMatrix X, NumericVector y, int K)
{
    SilhouetteIndex ind(
        matrix<FLOAT_T>(REAL(SEXP(X)), X.nrow(), X.ncol(), false),
        (uint8_t)K, false, false
    );
    ind.set_labels(translateLabels_fromR(y));
    return (double)ind.compute();
}


//' @rdname cluster_validity_measures
//' @export
// [[Rcpp::export]]
double silhouette_w_index(NumericMatrix X, NumericVector y, int K)
{
    SilhouetteIndex ind(
        matrix<FLOAT_T>(REAL(SEXP(X)), X.nrow(), X.ncol(), false),
        (uint8_t)K, false, true
    );
    ind.set_labels(translateLabels_fromR(y));
    return (double)ind.compute();
}


//' @rdname cluster_validity_measures
//' @export
// [[Rcpp::export]]
double wcnn_index(NumericMatrix X, NumericVector y, int K, int M=10)
{
    GENIECLUST_ASSERT(M>0);  // M = min(n-1, M) in the constructor

    WCNNIndex ind(
        matrix<FLOAT_T>(REAL(SEXP(X)), X.nrow(), X.ncol(), false),
        (uint8_t)K, false, M
    );

    ind.set_labels(translateLabels_fromR(y));
    return (double)ind.compute();
}


//' @rdname cluster_validity_measures
//' @export
// [[Rcpp::export]]
double wcss_index(NumericMatrix X, NumericVector y, int K)
{
    WCSSIndex ind(
        matrix<FLOAT_T>(REAL(SEXP(X)), X.nrow(), X.ncol(), false),
        (uint8_t)K, false, false/*not weighted*/
    );
    ind.set_labels(translateLabels_fromR(y));
    return (double)ind.compute();
}



#endif
