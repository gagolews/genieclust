/*  This file is part of the 'quitefastmst' package.
 *
 *  Copyleft (C) 2025-2025, Marek Gagolewski <https://www.gagolewski.com>
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


#include "c_fastmst.h"
#include "c_common.h"
#include <vector>
#include <cmath>


#define MST_OMP_CHUNK_SIZE 1024



/*! Determine the k nearest neighbours of each point
 *  wrt the Euclidean distance
 *
 *  Exactly n*(n-1)/2 distance computations are performed.
 *
 *  It is assumed that each query point is not its own neighbour.
 *
 *  Worst-case time complexity: O(n*(n-1)/2*d*k).
 *  So, use for small k, say, k<=20.
 *
 *
 *  @param X the n input points in R^d; a c_contiguous array, shape (n,d)
 *  @param n number of points
 *  @param d number of features
 *  @param k number of nearest neighbours requested
 *  @param nn_dist [out]  a c_contiguous array, shape (n,k),
 *         dist[i,j] gives the weight of the (undirected) edge {i, ind[i,j]}
 *  @param nn_ind [out]   a c_contiguous array, shape (n,k),
 *         (undirected) edge definition, interpreted as {i, ind[i,j]}
 *  @param squared return the squared Euclidean distance?
 *  @param verbose output diagnostic/progress messages?
 */
template <class FLOAT>
void Cknn1_euclid_brute(
    const FLOAT* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t k,
    FLOAT* nn_dist, Py_ssize_t* nn_ind, bool squared, bool verbose
) {
    if (n <= 0)   throw std::domain_error("n <= 0");
    if (d <= 0)   throw std::domain_error("d <= 0");
    if (k <= 0)   throw std::domain_error("k <= 0");
    if (k >= n)   throw std::domain_error("k >= n");

    if (verbose) GENIECLUST_PRINT("[genieclust] Determining the nearest neighbours... ");

    for (Py_ssize_t i=0; i<n*k; ++i) nn_dist[i] = INFINITY;
    for (Py_ssize_t i=0; i<n*k; ++i) nn_ind[i] = -1;

    std::vector<FLOAT> dij(n);
    for (Py_ssize_t i=0; i<n-1; ++i) {
        const FLOAT* x_cur = X+i*d;

        #if OPENMP_IS_ENABLED
        #pragma omp parallel for schedule(static,MST_OMP_CHUNK_SIZE)  /* chunks get smaller and smaller... */
        #endif
        for (Py_ssize_t j=i+1; j<n; ++j) {
            FLOAT dd = 0.0;
            for (Py_ssize_t u=0; u<d; ++u)
                dd += square(x_cur[u]-X[j*d+u]);
            dij[j] = dd;

            if (dd < nn_dist[j*k+k-1]) {
                // i might be amongst k-NNs of j;
                // insert into an ordered sequence
                Py_ssize_t l = k-1;
                while (l > 0 && dd < nn_dist[j*k+l-1]) {
                    nn_dist[j*k+l] = nn_dist[j*k+l-1];
                    nn_ind[j*k+l]  = nn_ind[j*k+l-1];
                    l -= 1;
                }
                nn_dist[j*k+l] = dd;
                nn_ind[j*k+l]  = i;
            }
        }

        // This part can't be (naively) parallelised
        for (Py_ssize_t j=i+1; j<n; ++j) {
            if (dij[j] < nn_dist[i*k+k-1]) {
                // j might be amongst k-NNs of i
                Py_ssize_t l = k-1;
                while (l > 0 && dij[j] < nn_dist[i*k+l-1]) {
                    nn_dist[i*k+l] = nn_dist[i*k+l-1];
                    nn_ind[i*k+l]  = nn_ind[i*k+l-1];
                    l -= 1;
                }
                nn_dist[i*k+l] = dij[j];
                nn_ind[i*k+l]  = j;
            }
        }

        // if (verbose) GENIECLUST_PRINT("\b\b\b\b%3d%%", (n-1+n-i-1)*(i+1)*100/n/(n-1));

        if (i % MST_OMP_CHUNK_SIZE == MST_OMP_CHUNK_SIZE-1) {
            #if GENIECLUST_R
            Rcpp::checkUserInterrupt();  // throws an exception, not a longjmp
            #elif GENIECLUST_PYTHON
            if (PyErr_CheckSignals() != 0) throw std::runtime_error("signal caught");
            #endif
        }
    }

    if (!squared) {
        for (Py_ssize_t i=0; i<k*n; ++i)
            nn_dist[i] = sqrt(nn_dist[i]);
    }

    if (verbose) GENIECLUST_PRINT("done.\n");
}


/*! Determine the k nearest neighbours of each point
 *  wrt the Euclidean distance
 *
 *  Use for small k, say, k<=20.
 *
 *
 *  @param X the n input points in R^d; a c_contiguous array, shape (n,d)
 *  @param n number of points
 *  @param Y the m query points in R^d; a c_contiguous array, shape (m,d)
 *  @param m number of points
 *  @param d number of features
 *  @param k number of nearest neighbours requested
 *  @param nn_dist [out]  a c_contiguous array, shape (m,k),
 *         dist[i,j] gives the weight of the (undirected) edge {i, ind[i,j]}
 *  @param nn_ind [out]   a c_contiguous array, shape (m,k),
 *         (undirected) edge definition, interpreted as {i, ind[i,j]}
 *  @param squared return the squared Euclidean distance?
 *  @param verbose output diagnostic/progress messages?
 */
template <class FLOAT>
void Cknn2_euclid_brute(
    const FLOAT* X, Py_ssize_t n, const FLOAT* Y, Py_ssize_t m,
    Py_ssize_t d, Py_ssize_t k,
    FLOAT* nn_dist, Py_ssize_t* nn_ind, bool squared, bool verbose
) {
    if (n <= 0)   throw std::domain_error("n <= 0");
    if (m <= 0)   throw std::domain_error("m <= 0");
    if (d <= 0)   throw std::domain_error("d <= 0");
    if (k <= 0)   throw std::domain_error("k <= 0");
    if (k >  n)   throw std::domain_error("k > n");

    if (verbose) GENIECLUST_PRINT("[genieclust] Determining the nearest neighbours... ");

    for (Py_ssize_t i=0; i<m*k; ++i) nn_dist[i] = INFINITY;
    for (Py_ssize_t i=0; i<m*k; ++i) nn_ind[i] = -1;

    #if OPENMP_IS_ENABLED
    #pragma omp parallel for schedule(static)
    #endif
    for (Py_ssize_t i=0; i<m; ++i) {
        const FLOAT* y_cur = Y+i*d;

        const FLOAT* x_cur = X;
        for (Py_ssize_t j=0; j<n; ++j) {
            FLOAT dd = 0.0;
            for (Py_ssize_t u=0; u<d; ++u)
                dd += square(y_cur[u]-x_cur[u]);
            x_cur += d;

            if (dd < nn_dist[i*k+k-1]) {
                Py_ssize_t l = k-1;
                while (l > 0 && dd < nn_dist[i*k+l-1]) {
                    nn_dist[i*k+l] = nn_dist[i*k+l-1];
                    nn_ind[i*k+l]  = nn_ind[i*k+l-1];
                    l -= 1;
                }
                nn_dist[i*k+l] = dd;
                nn_ind[i*k+l]  = j;
            }
        }
    }

    if (!squared) {
        for (Py_ssize_t i=0; i<m*k; ++i)
            nn_dist[i] = sqrt(nn_dist[i]);
    }

    if (verbose) GENIECLUST_PRINT("done.\n");
}


// instantiate:
template void Cknn1_euclid_brute<float>(
    const float* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t k,
    float* nn_dist, Py_ssize_t* nn_ind, bool squared, bool verbose
);

template void Cknn2_euclid_brute<float>(
    const float* X, Py_ssize_t n, const float* Y, Py_ssize_t m,
    Py_ssize_t d, Py_ssize_t k,
    float* nn_dist, Py_ssize_t* nn_ind, bool squared, bool verbose
);

template void Cknn1_euclid_brute<double>(
    const double* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t k,
    double* nn_dist, Py_ssize_t* nn_ind, bool squared, bool verbose
);

template void Cknn2_euclid_brute<double>(
    const double* X, Py_ssize_t n, const double* Y, Py_ssize_t m,
    Py_ssize_t d, Py_ssize_t k,
    double* nn_dist, Py_ssize_t* nn_ind, bool squared, bool verbose
);
