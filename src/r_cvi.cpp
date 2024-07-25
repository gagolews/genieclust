/*  Rcpp exports - Internal cluster validity indices
 *
 *  Code originally contributed in <https://github.com/gagolews/optim_cvi>,
 *  see https://doi.org/10.1016/j.ins.2021.10.004.
 *
 *  Copyleft (C) 2020-2024, Marek Gagolewski <https://www.gagolewski.com>
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
 * @param x numeric vector with integer elements
 * @param K [out] the number of clusters
 * @return vector
 */
std::vector<Py_ssize_t> translateLabels_fromR(const Rcpp::NumericVector& x, Py_ssize_t& K)
{
    size_t n = x.size();
    std::vector<Py_ssize_t> ret(n);
    K = 0;
    for (size_t i=0; i<n; ++i) {
        int xi = (int)x[i];
        if (xi < 1) Rf_error("All elements in a label vector must be >= 1.");
        ret[i] = (Py_ssize_t)(xi-1); // 1-based -> 0-based

        if (K < xi) K = xi;  // determine the max(x)
    }
    return ret;
}



//' @title Internal Cluster Validity Measures
//'
//' @description
//' Implementation of a number of so-called cluster validity indices critically
//' reviewed in (Gagolewski, Bartoszuk, Cena, 2021). See Section 2
//' therein and (Gagolewski, 2022) for the respective definitions.
//'
//' The greater the index value, the more \emph{valid} (whatever that means)
//' the assessed partition. For consistency, the Ball-Hall and
//' Davies-Bouldin indexes as well as the within-cluster sum of squares (WCSS)
//' take negative values.
//'
//'
//' @param X numeric matrix with \code{n} rows and \code{d} columns,
//'     representing \code{n} points in a \code{d}-dimensional space
//'
//' @param y vector of \code{n} integer labels,
//'     representing a partition whose \emph{quality} is to be
//'     assessed; \code{y[i]} is the cluster ID of the \code{i}-th point,
//'     \code{X[i, ]}; \code{1 <= y[i] <= K}, where \code{K} is the number
//'     or clusters
//'
//' @param M number of nearest neighbours
//'
//' @param lowercase_d an integer between 1 and 5, denoting
//'     \eqn{d_1}, ..., \eqn{d_5} in the definition
//'     of the generalised Dunn (Bezdek-Pal) index (numerator:
//'     min, max, and mean pairwise intracluster distance,
//'     distance between cluster centroids,
//'     weighted point-centroid distance, respectively)
//'
//' @param uppercase_d an integer between 1 and 3, denoting
//'     \eqn{D_1}, ..., \eqn{D_3} in the definition
//'     of the generalised Dunn (Bezdek-Pal) index (denominator:
//'       max and min pairwise intracluster distance, average point-centroid
//'       distance, respectively)
//'
//' @param owa_numerator,owa_denominator single string specifying
//'     the OWA operators to use in the definition of the DuNN index;
//'     one of: \code{"Mean"}, \code{"Min"}, \code{"Max"}, \code{"Const"},
//'     \code{"SMin:D"}, \code{"SMax:D"}, where \code{D} is an integer
//'     defining the degree of smoothness
//'
//'
//' @return
//' A single numeric value (the more, the \emph{better}).
//'
//' @references
//' Ball G.H., Hall D.J.,
//' \emph{ISODATA: A novel method of data analysis and pattern classification},
//' Technical report No. AD699616, Stanford Research Institute, 1965.
//'
//' Bezdek J., Pal N., Some new indexes of cluster validity,
//' \emph{IEEE Transactions on Systems, Man, and Cybernetics, Part B} 28,
//' 1998, 301-315, \doi{10.1109/3477.678624}.
//'
//' Calinski T., Harabasz J., A dendrite method for cluster analysis,
//' \emph{Communications in Statistics} 3(1), 1974, 1-27,
//' \doi{10.1080/03610927408827101}.
//'
//' Davies D.L., Bouldin D.W.,
//' A Cluster Separation Measure,
//' \emph{IEEE Transactions on Pattern Analysis and Machine Intelligence}
//' PAMI-1 (2), 1979, 224-227, \doi{10.1109/TPAMI.1979.4766909}.
//'
//' Dunn J.C., A Fuzzy Relative of the ISODATA Process and Its Use in Detecting
//' Compact Well-Separated Clusters, \emph{Journal of Cybernetics} 3(3), 1973,
//' 32-57, \doi{10.1080/01969727308546046}.
//'
//' Gagolewski M., Bartoszuk M., Cena A.,
//' Are cluster validity measures (in)valid?, \emph{Information Sciences} 581,
//' 620-636, 2021, \doi{10.1016/j.ins.2021.10.004};
//' preprint: \url{https://raw.githubusercontent.com/gagolews/bibliography/master/preprints/2021cvi.pdf}.
//'
//' Gagolewski M., A Framework for Benchmarking Clustering Algorithms,
//' \emph{SoftwareX} 20, 2022, 101270,
//' \doi{10.1016/j.softx.2022.101270},
//' \url{https://clustering-benchmarks.gagolewski.com}.
//'
//' Rousseeuw P.J., Silhouettes: A Graphical Aid to the Interpretation and
//' Validation of Cluster Analysis, \emph{Computational and Applied Mathematics}
//' 20, 1987, 53-65, \doi{10.1016/0377-0427(87)90125-7}.
//'
//'
//'
//' @examples
//' X <- as.matrix(iris[,1:4])
//' X[,] <- jitter(X)  # otherwise we get a non-unique solution
//' y <- as.integer(iris[[5]])
//' calinski_harabasz_index(X, y)  # good
//' calinski_harabasz_index(X, sample(1:3, nrow(X), replace=TRUE))  # bad
//'
//' @name cluster_validity
//' @rdname cluster_validity
//' @export
// [[Rcpp::export]]
double calinski_harabasz_index(NumericMatrix X, NumericVector y)
{
    Py_ssize_t K;
    std::vector<Py_ssize_t> _y = translateLabels_fromR(y, /*out*/K);
    CMatrix<FLOAT_T> _X(REAL(SEXP(X)), X.nrow(), X.ncol(), false);
    if (_X.nrow() < 1 || _X.nrow() != _y.size())
        Rf_error("Incompatible X and y");

    CalinskiHarabaszIndex ind(_X, (Py_ssize_t)K);
    ind.set_labels(_y);

    return (double)ind.compute();
}


//' @rdname cluster_validity
//' @export
// [[Rcpp::export]]
double dunnowa_index(NumericMatrix X, NumericVector y, int M=25,
                Rcpp::String owa_numerator="SMin:5",
                Rcpp::String owa_denominator="Const")
{
    Py_ssize_t K;
    std::vector<Py_ssize_t> _y = translateLabels_fromR(y, /*out*/K);
    CMatrix<FLOAT_T> _X(REAL(SEXP(X)), X.nrow(), X.ncol(), false);
    if (_X.nrow() < 1 || _X.nrow() != _y.size())
        Rf_error("Incompatible X and y");

    if (M <= 0)    // M = min(n-1, M) in the constructor
        Rf_error("M must be positive.");

    int _owa_numerator = DuNNOWA_get_OWA(std::string(owa_numerator));
    int _owa_denominator = DuNNOWA_get_OWA(std::string(owa_denominator));

    if (_owa_numerator == OWA_ERROR || _owa_denominator == OWA_ERROR) {
        Rf_error("invalid OWA operator specifier");
    }

    DuNNOWAIndex ind(_X, (Py_ssize_t)K, false, M, _owa_numerator, _owa_denominator);
    ind.set_labels(_y);

    return (double)ind.compute();
}



//' @rdname cluster_validity
//' @export
// [[Rcpp::export]]
double generalised_dunn_index(NumericMatrix X, NumericVector y,
    int lowercase_d, int uppercase_d)
{
    Py_ssize_t K;
    std::vector<Py_ssize_t> _y = translateLabels_fromR(y, /*out*/K);
    CMatrix<FLOAT_T> _X(REAL(SEXP(X)), X.nrow(), X.ncol(), false);
    if (_X.nrow() < 1 || _X.nrow() != _y.size())
        Rf_error("Incompatible X and y");

    LowercaseDeltaFactory* lowercase_deltaFactory;
    UppercaseDeltaFactory* uppercase_deltaFactory;

    if (lowercase_d == 1) {
        lowercase_deltaFactory = new LowercaseDelta1Factory();
    }
    else if (lowercase_d == 2) {
        lowercase_deltaFactory = new LowercaseDelta2Factory();
    }
    else if (lowercase_d == 3) {
        lowercase_deltaFactory = new LowercaseDelta3Factory();
    }
    else if (lowercase_d == 4) {
        lowercase_deltaFactory = new LowercaseDelta4Factory();
    }
    else if (lowercase_d == 5) {
        lowercase_deltaFactory = new LowercaseDelta5Factory();
    }
    else if (lowercase_d == 6) {
        lowercase_deltaFactory = new LowercaseDelta6Factory();
    }
    else {
        Rf_error("invalid lowercase_d");
    }

    if (uppercase_d == 1) {
        uppercase_deltaFactory = new UppercaseDelta1Factory();
    }
    else if (uppercase_d == 2) {
        uppercase_deltaFactory = new UppercaseDelta2Factory();
    }
    else if (uppercase_d == 3) {
        uppercase_deltaFactory = new UppercaseDelta3Factory();
    }
    else {
        Rf_error("invalid uppercase_d");
    }

    bool areCentroidsNeeded = (
        lowercase_deltaFactory->IsCentroidNeeded() ||
        uppercase_deltaFactory->IsCentroidNeeded()
    );

    if (areCentroidsNeeded) {
        GeneralizedDunnIndexCentroidBased ind(_X, (Py_ssize_t)K,
            lowercase_deltaFactory, uppercase_deltaFactory);

        delete lowercase_deltaFactory;
        delete uppercase_deltaFactory;

        ind.set_labels(_y);

        return (double)ind.compute();
    }
    else {
        GeneralizedDunnIndex ind(_X, (Py_ssize_t)K,
            lowercase_deltaFactory, uppercase_deltaFactory);

        delete lowercase_deltaFactory;
        delete uppercase_deltaFactory;

        ind.set_labels(_y);

        return (double)ind.compute();
    }
}


//' @rdname cluster_validity
//' @export
// [[Rcpp::export]]
double negated_ball_hall_index(NumericMatrix X, NumericVector y)
{
    Py_ssize_t K;
    std::vector<Py_ssize_t> _y = translateLabels_fromR(y, /*out*/K);
    CMatrix<FLOAT_T> _X(REAL(SEXP(X)), X.nrow(), X.ncol(), false);
    if (_X.nrow() < 1 || _X.nrow() != _y.size())
        Rf_error("Incompatible X and y");

    WCSSIndex ind(_X, (Py_ssize_t)K, false, true/*weighted*/);
    ind.set_labels(_y);

    return (double)ind.compute();
}


//' @rdname cluster_validity
//' @export
// [[Rcpp::export]]
double negated_davies_bouldin_index(NumericMatrix X, NumericVector y)
{
    Py_ssize_t K;
    std::vector<Py_ssize_t> _y = translateLabels_fromR(y, /*out*/K);
    CMatrix<FLOAT_T> _X(REAL(SEXP(X)), X.nrow(), X.ncol(), false);
    if (_X.nrow() < 1 || _X.nrow() != _y.size())
        Rf_error("Incompatible X and y");

    DaviesBouldinIndex ind(_X, (Py_ssize_t)K);
    ind.set_labels(_y);

    return (double)ind.compute();
}


//' @rdname cluster_validity
//' @export
// [[Rcpp::export]]
double negated_wcss_index(NumericMatrix X, NumericVector y)
{
    Py_ssize_t K;
    std::vector<Py_ssize_t> _y = translateLabels_fromR(y, /*out*/K);
    CMatrix<FLOAT_T> _X(REAL(SEXP(X)), X.nrow(), X.ncol(), false);
    if (_X.nrow() < 1 || _X.nrow() != _y.size())
        Rf_error("Incompatible X and y");

    WCSSIndex ind(_X, (Py_ssize_t)K, false, false/*not weighted*/);
    ind.set_labels(_y);

    return (double)ind.compute();
}


//' @rdname cluster_validity
//' @export
// [[Rcpp::export]]
double silhouette_index(NumericMatrix X, NumericVector y)
{
    Py_ssize_t K;
    std::vector<Py_ssize_t> _y = translateLabels_fromR(y, /*out*/K);
    CMatrix<FLOAT_T> _X(REAL(SEXP(X)), X.nrow(), X.ncol(), false);
    if (_X.nrow() < 1 || _X.nrow() != _y.size())
        Rf_error("Incompatible X and y");

    SilhouetteIndex ind(_X, (Py_ssize_t)K, false, false);
    ind.set_labels(_y);

    return (double)ind.compute();
}


//' @rdname cluster_validity
//' @export
// [[Rcpp::export]]
double silhouette_w_index(NumericMatrix X, NumericVector y)
{
    Py_ssize_t K;
    std::vector<Py_ssize_t> _y = translateLabels_fromR(y, /*out*/K);
    CMatrix<FLOAT_T> _X(REAL(SEXP(X)), X.nrow(), X.ncol(), false);
    if (_X.nrow() < 1 || _X.nrow() != _y.size())
        Rf_error("Incompatible X and y");

    SilhouetteIndex ind(_X, (Py_ssize_t)K, false, true);
    ind.set_labels(_y);

    return (double)ind.compute();
}


//' @rdname cluster_validity
//' @export
// [[Rcpp::export]]
double wcnn_index(NumericMatrix X, NumericVector y, int M=25)
{
    Py_ssize_t K;
    std::vector<Py_ssize_t> _y = translateLabels_fromR(y, /*out*/K);
    CMatrix<FLOAT_T> _X(REAL(SEXP(X)), X.nrow(), X.ncol(), false);
    if (_X.nrow() < 1 || _X.nrow() != _y.size())
        Rf_error("Incompatible X and y");

    if (M <= 0)    // M = min(n-1, M) in the constructor
        Rf_error("M must be positive.");

    WCNNIndex ind(_X, (Py_ssize_t)K, false, M);
    ind.set_labels(_y);

    return (double)ind.compute();
}
