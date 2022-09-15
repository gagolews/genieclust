/*  Provides easy access to the internal cluster validity indices
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


#ifndef __c_cvi_h
#define __c_cvi_h

#include <string>
#include <cstring>
#include <vector>
#include <exception>

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



double c_calinski_harabasz_index(const double* X, const Py_ssize_t* y,
                                 size_t n, size_t d, Py_ssize_t K)
{
    CalinskiHarabaszIndex ind(
        CMatrix<FLOAT_T>(X, n, d, /*_c_order=*/true), (Py_ssize_t)K
    );
    ind.set_labels(std::vector<Py_ssize_t>(y, y+n));

    return (double)ind.compute();
}


double c_negated_ball_hall_index(const double* X, const Py_ssize_t* y,
                               size_t n, size_t d, Py_ssize_t K)
{
    WCSSIndex ind(
        CMatrix<FLOAT_T>(X, n, d, /*_c_order=*/true), (Py_ssize_t)K,
        false, true/*weighted*/
    );
    ind.set_labels(std::vector<Py_ssize_t>(y, y+n));

    return (double)ind.compute();
}


double c_negated_davies_bouldin_index(const double* X, const Py_ssize_t* y,
                                    size_t n, size_t d, Py_ssize_t K)
{
    DaviesBouldinIndex ind(
        CMatrix<FLOAT_T>(X, n, d, /*_c_order=*/true), (Py_ssize_t)K
    );
    ind.set_labels(std::vector<Py_ssize_t>(y, y+n));

    return (double)ind.compute();
}


double c_negated_wcss_index(const double* X, const Py_ssize_t* y,
                          size_t n, size_t d, Py_ssize_t K)
{
    WCSSIndex ind(
        CMatrix<FLOAT_T>(X, n, d, /*_c_order=*/true), (Py_ssize_t)K,
        false, false/*not weighted*/
    );
    ind.set_labels(std::vector<Py_ssize_t>(y, y+n));

    return (double)ind.compute();
}


double c_silhouette_index(const double* X, const Py_ssize_t* y,
                        size_t n, size_t d, Py_ssize_t K)
{
    SilhouetteIndex ind(
        CMatrix<FLOAT_T>(X, n, d, /*_c_order=*/true), (Py_ssize_t)K, false, false
    );
    ind.set_labels(std::vector<Py_ssize_t>(y, y+n));

    return (double)ind.compute();
}


double c_silhouette_w_index(const double* X, const Py_ssize_t* y,
                          size_t n, size_t d, Py_ssize_t K)
{
    SilhouetteIndex ind(
        CMatrix<FLOAT_T>(X, n, d, /*_c_order=*/true), (Py_ssize_t)K, false, true
    );
    ind.set_labels(std::vector<Py_ssize_t>(y, y+n));

    return (double)ind.compute();
}


double c_wcnn_index(const double* X, const Py_ssize_t* y,
                    size_t n, size_t d, Py_ssize_t K, size_t M)
{
    if (M <= 0)
        throw std::invalid_argument("M must be positive.");

    WCNNIndex ind(
        CMatrix<FLOAT_T>(X, n, d, /*_c_order=*/true), (Py_ssize_t)K, false, M
    );
    ind.set_labels(std::vector<Py_ssize_t>(y, y+n));

    return (double)ind.compute();
}


double c_dunnowa_index(const double* X, const Py_ssize_t* y,
    size_t n, size_t d, Py_ssize_t K, size_t M,
    const char* owa_numerator, const char* owa_denominator)
{
    int _owa_numerator = DuNNOWA_get_OWA(std::string(owa_numerator));
    int _owa_denominator = DuNNOWA_get_OWA(std::string(owa_denominator));

    if (_owa_numerator == OWA_ERROR || _owa_denominator == OWA_ERROR)
        throw std::invalid_argument("Invalid OWA operator specifier.");

    if (M <= 0)
        throw std::invalid_argument("M must be positive.");

    DuNNOWAIndex ind(
        CMatrix<FLOAT_T>(X, n, d, /*_c_order=*/true), (Py_ssize_t)K,
        false, M, _owa_numerator, _owa_denominator
    );
    ind.set_labels(std::vector<Py_ssize_t>(y, y+n));

    return (double)ind.compute();
}


double c_generalised_dunn_index(const double* X, const Py_ssize_t* y,
    size_t n, size_t d, Py_ssize_t K, size_t lowercase_d, size_t uppercase_d)
{
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
        throw std::invalid_argument("Invalid lowercase_d.");
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
        throw std::invalid_argument("Invalid uppercase_d.");
    }

    bool areCentroidsNeeded = (
        lowercase_deltaFactory->IsCentroidNeeded() ||
        uppercase_deltaFactory->IsCentroidNeeded()
    );

    if (areCentroidsNeeded) {
        GeneralizedDunnIndexCentroidBased ind(
            CMatrix<FLOAT_T>(X, n, d, /*_c_order=*/true), (Py_ssize_t)K,
            lowercase_deltaFactory, uppercase_deltaFactory);

        delete lowercase_deltaFactory;
        delete uppercase_deltaFactory;

        ind.set_labels(std::vector<Py_ssize_t>(y, y+n));

        return (double)ind.compute();
    }
    else {
        GeneralizedDunnIndex ind(
            CMatrix<FLOAT_T>(X, n, d, /*_c_order=*/true), (Py_ssize_t)K,
            lowercase_deltaFactory, uppercase_deltaFactory);

        delete lowercase_deltaFactory;
        delete uppercase_deltaFactory;

        ind.set_labels(std::vector<Py_ssize_t>(y, y+n));

        return (double)ind.compute();
    }
}



#endif
