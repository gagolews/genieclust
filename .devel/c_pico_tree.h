/*
Testing Jonathan Broere's PicoTree: a C++ header only library for fast nearest
neighbor and range searches using K-d trees.
<https://github.com/Jaybro/pico_tree>

The code below worked with Version 1.0.0 (c5f719837df9707ee12d94cb0108aa0c34bfe96f) thereof.
We'd rather stick with our own implementation of K-d trees.
*/


#ifndef __c_picotree_h
#define __c_picotree_h


#include "pico_tree/array_traits.hpp"
#include "pico_tree/kd_tree.hpp"
#include "pico_tree/vector_traits.hpp"


template <class FLOAT, Py_ssize_t D>
void Cknn_sqeuclid_picotree(const FLOAT* X, const Py_ssize_t n, const Py_ssize_t k,
    FLOAT* nn_dist, Py_ssize_t* nn_ind, Py_ssize_t max_leaf_size, bool /*verbose*/)
{
    std::vector<std::array<FLOAT, D>> points(n);    // float32 - faster
    for (Py_ssize_t i=0; i<n; ++i) {
        for (Py_ssize_t u=0; u<D; ++u) {
            points[i][u] = (FLOAT)X[i*D+u];
        }
    }

    pico_tree::kd_tree tree(std::ref(points), (pico_tree::max_leaf_size_t)max_leaf_size);

    #if OPENMP_IS_ENABLED
    #pragma omp parallel for schedule(static)
    #endif
    for (Py_ssize_t i=0; i<n; ++i) {
        std::vector< pico_tree::neighbor<int, FLOAT> > knn;
        tree.search_knn(points[i], k+1, knn);
        GENIECLUST_ASSERT(knn[0].index == i);

        const FLOAT* x_cur = X+i*D;
        for (Py_ssize_t j=0; j<k; ++j) {
            nn_ind[i*k+j]  = knn[j+1].index;
            nn_dist[i*k+j] = (FLOAT)knn[j+1].distance;
            // // recompute the distance using FLOAT's precision
            // const FLOAT* x_other = X+nn_ind[i*k+j]*D;
            // FLOAT _d = 0.0;
            // for (Py_ssize_t u=0; u<D; ++u) {
            //     FLOAT _df = x_cur[u]-x_other[u];
            //     _d += _df*_df;
            // }
            // nn_dist[i*k+j] = _d;
        }

        // #if GENIECLUST_R
        // Rcpp::checkUserInterrupt();
        // #elif GENIECLUST_PYTHON
        // if (PyErr_CheckSignals() != 0) throw std::runtime_error("signal caught");
        // #endif
    }
}


/*! Get the k nearest neighbours of each point w.r.t. the Euclidean distance
 *
 *
 * @param X a C-contiguous data matrix
 * @param n number of rows
 * @param d number of columns
 * @param k number of nearest neighbours to look for
 * @param nn_dist [out] vector(matrix) of length n*k, distances to NNs
 * @param nn_ind [out] vector(matrix) of length n*k, indexes of NNs
 * @param verbose output diagnostic/progress messages?
 */
template <class T>
void Cknn_sqeuclid_picotree(const T* X, Py_ssize_t n, Py_ssize_t d, Py_ssize_t k,
    T* nn_dist, Py_ssize_t* nn_ind, Py_ssize_t max_leaf_size=12, bool verbose=false)
{
    if (n <= 0)   throw std::domain_error("n <= 0");
    if (k <= 0)   throw std::domain_error("k <= 0");
    if (k >= n)   throw std::domain_error("k >= n");

    if (max_leaf_size < 0) throw std::domain_error("max_leaf_size < 0");
    else if (max_leaf_size == 0) max_leaf_size = 12;  // default

    /* OMFG. Templates. */
    /**/ if (d ==  2)  Cknn_sqeuclid_picotree<T,  2>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d ==  3)  Cknn_sqeuclid_picotree<T,  3>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d ==  4)  Cknn_sqeuclid_picotree<T,  4>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d ==  5)  Cknn_sqeuclid_picotree<T,  5>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d ==  6)  Cknn_sqeuclid_picotree<T,  6>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d ==  7)  Cknn_sqeuclid_picotree<T,  7>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d ==  8)  Cknn_sqeuclid_picotree<T,  8>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d ==  9)  Cknn_sqeuclid_picotree<T,  9>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 10)  Cknn_sqeuclid_picotree<T, 10>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 11)  Cknn_sqeuclid_picotree<T, 11>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 12)  Cknn_sqeuclid_picotree<T, 12>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 13)  Cknn_sqeuclid_picotree<T, 13>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 14)  Cknn_sqeuclid_picotree<T, 14>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 15)  Cknn_sqeuclid_picotree<T, 15>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 16)  Cknn_sqeuclid_picotree<T, 16>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 17)  Cknn_sqeuclid_picotree<T, 17>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 18)  Cknn_sqeuclid_picotree<T, 18>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 19)  Cknn_sqeuclid_picotree<T, 19>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else if (d == 20)  Cknn_sqeuclid_picotree<T, 20>(X, n, k, nn_dist, nn_ind, max_leaf_size, verbose);
    else {
        throw std::runtime_error("d should be between 2 and 20");
    }
}



// [[Rcpp::export]]
Rcpp::RObject knn_pico_tree(Rcpp::NumericMatrix X, int k=1, int max_leaf_size=12)
{
    using FLOAT = double;  // float is not faster..

    Py_ssize_t n = (Py_ssize_t)X.nrow();
    Py_ssize_t d = (Py_ssize_t)X.ncol();
    if (n < 1 || d < 1) return R_NilValue;
    if (k < 1) return R_NilValue;

    std::vector<FLOAT> XC(n*d);
    Py_ssize_t j = 0;
    for (Py_ssize_t i=0; i<n; ++i)
        for (Py_ssize_t u=0; u<d; ++u)
            XC[j++] = (FLOAT)X(i, u);  // row-major

    std::vector<FLOAT>  nn_dist(n*k);
    std::vector<Py_ssize_t> nn_ind(n*k);

    if (d >= 2 && d <= 20) {
        Cknn_sqeuclid_picotree<FLOAT>(XC.data(), n, d, k, nn_dist.data(), nn_ind.data(), max_leaf_size);
    }
    else {
        throw std::runtime_error("d not between 2 and 20");
    }

    Rcpp::IntegerMatrix out_ind(n, k);
    Rcpp::NumericMatrix out_dist(n, k);
    Py_ssize_t u = 0;
    for (Py_ssize_t i=0; i<n; ++i) {
        for (Py_ssize_t j=0; j<k; ++j) {
            out_ind(i, j)  = nn_ind[u]+1.0;  // R-based indexing
            out_dist(i, j) = sqrt(nn_dist[u]);
            u++;
        }
    }

    return Rcpp::List::create(
        Rcpp::Named("nn.index")=out_ind,
        Rcpp::Named("nn.dist")=out_dist
    );
}


#endif
