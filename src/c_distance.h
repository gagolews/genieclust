/*  Various distances (Euclidean, mutual reachability distance, ...)
 *
 *  Copyright (C) 2018-2020 Marek Gagolewski (https://www.gagolewski.com)
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *  contributors may be used to endorse or promote products derived from this
 *  software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 *  OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef __c_distance_h
#define __c_distance_h

#include "c_common.h"
#include <vector>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif



template<class T>
inline T square(T x) { return x*x; }



/*! Abstract base class for all distances */
template<class T>
struct CDistance {
    virtual ~CDistance() {}

    /*!
     * @param i point index, 0<=i<n
     * @param M indices
     * @param k length of M
     * @return distances from the i-th point to M[0], .., M[k-1],
     *         with ret[M[j]]=d(i, M[j]);
     *         the user does not own ret;
     *         the function is not thread-safe
     */
    virtual const T* operator()(ssize_t i, const ssize_t* M, ssize_t k) = 0;
};



/*! A class to "compute" the distances from the i-th point
 *  to all n points based on a pre-computed n*n symmetric,
 *  complete pairwise distance c_contiguous matrix.
 */
template<class T>
struct CDistanceCompletePrecomputed : public CDistance<T> {
    const T* dist;
    ssize_t n;

    /*!
     * @param dist n*n c_contiguous array, dist[i,j] is the distance between
     *    the i-th and the j-th point, the matrix is symmetric
     * @param n number of points
     */
    CDistanceCompletePrecomputed(const T* dist, ssize_t n) {
        this->n = n;
        this->dist = dist;
    }

    CDistanceCompletePrecomputed()
        : CDistanceCompletePrecomputed(NULL, 0) { }

    virtual const T* operator()(ssize_t i, const ssize_t* /*M*/, ssize_t /*k*/) {
        return &this->dist[i*n]; // the i-th row of dist
    }
};





/*! A class to compute the Euclidean distances from the i-th point
 *  to all given k points.
 */
template<class T>
struct CDistanceEuclidean : public CDistance<T>  {
    const T* X;
    ssize_t n;
    ssize_t d;
    bool squared;
    std::vector<T> buf;
    // std::vector<T> sqnorm;

    /*!
     * @param X n*d c_contiguous array
     * @param n number of points
     * @param d dimensionality
     * @param squared true for the squared Euclidean distance
     */
    CDistanceEuclidean(const T* X, ssize_t n, ssize_t d, bool squared=false)
            : buf(n)//, sqnorm(n)
    {
        this->n = n;
        this->d = d;
        this->X = X;
        this->squared = squared;

//         T* __sqnorm = sqnorm.data();
// #ifdef _OPENMP
//         #pragma omp parallel for schedule(static)
// #endif
//         for (ssize_t i=0; i<n; ++i) {
//             __sqnorm[i] = 0.0;
//             for (ssize_t u=0; u<d; ++u) {
//                 __sqnorm[i] += X[d*i+u]*X[d*i+u];
//             }
//         }
    }

    CDistanceEuclidean()
        : CDistanceEuclidean(NULL, 0, 0) { }

    virtual const T* operator()(ssize_t i, const ssize_t* M, ssize_t k) {
        T* __buf = buf.data();
        // T* __sqnorm = sqnorm.data();
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (ssize_t j=0; j<k; ++j) {
            ssize_t w = M[j];
            // GENIECLUST_ASSERT(w>=0 && w < n)
            __buf[w] = 0.0;

//             const T* x = X+d*i;
//             const T* y = X+d*w;
//             for (ssize_t u=0; u<d; ++u) {
//                 __buf[w] += (*x-*y)*(*x-*y);
//                 ++x; ++y;
//             }

            // or we could use the BLAS snrm2() for increased numerical stability.
            for (ssize_t u=0; u<d; ++u) {
                __buf[w] += square(X[d*i+u]-X[d*w+u]);
            }


            // // did you know that (x-y)**2 = x**2 + y**2 - 2*x*y ?
            // const T* x = X+d*i;
            // const T* y = X+d*w;
            // for (ssize_t u=0; u<d; ++u) {
            //     __buf[w] -= (*(x++))*(*(y++));
            // }
            // __buf[w] = 2.0*__buf[w]+__sqnorm[i]+__sqnorm[w];

            if (!squared) __buf[w] = sqrt(__buf[w]);
        }
        return __buf;
    }
};



/*! A class to compute the CDistanceManhattan distances from the i-th point
 *  to all given k points.
 */
template<class T>
struct CDistanceManhattan : public CDistance<T>  {
    const T* X;
    ssize_t n;
    ssize_t d;
    std::vector<T> buf;

    /*!
     * @param X n*d c_contiguous array
     * @param n number of points
     * @param d dimensionality
     */
    CDistanceManhattan(const T* X, ssize_t n, ssize_t d)
            : buf(n)
    {
        this->n = n;
        this->d = d;
        this->X = X;
    }

    CDistanceManhattan()
        : CDistanceManhattan(NULL, 0, 0) { }

    virtual const T* operator()(ssize_t i, const ssize_t* M, ssize_t k) {
        T* __buf = buf.data();
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (ssize_t j=0; j<k; ++j) {
            ssize_t w = M[j];
            // GENIECLUST_ASSERT(w>=0 && w<n)
            __buf[w] = 0.0;

            for (ssize_t u=0; u<d; ++u) {
                __buf[w] += fabs(X[d*i+u]-X[d*w+u]);
            }
        }
        return __buf;
    }
};



/*! A class to compute the cosine distances from the i-th point
 *  to all given k points.
 */
template<class T>
struct CDistanceCosine : public CDistance<T>  {
    const T* X;
    ssize_t n;
    ssize_t d;
    std::vector<T> buf;
    std::vector<T> norm;

    /*!
     * @param X n*d c_contiguous array
     * @param n number of points
     * @param d dimensionality
     */
    CDistanceCosine(const T* X, ssize_t n, ssize_t d)
            : buf(n), norm(n)
    {
        this->n = n;
        this->d = d;
        this->X = X;

        T* __norm = norm.data();
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (ssize_t i=0; i<n; ++i) {
            __norm[i] = 0.0;
            for (ssize_t u=0; u<d; ++u) {
                __norm[i] += X[d*i+u]*X[d*i+u];
            }
            __norm[i] = sqrt(__norm[i]);
        }
    }

    CDistanceCosine()
        : CDistanceCosine(NULL, 0, 0) { }

    virtual const T* operator()(ssize_t i, const ssize_t* M, ssize_t k) {
        T*  __buf = buf.data();
        T* __norm = norm.data();
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (ssize_t j=0; j<k; ++j) {
            ssize_t w = M[j];
            // GENIECLUST_ASSERT(w>=0&&w<n)
            __buf[w] = 0.0;

            for (ssize_t u=0; u<d; ++u) {
                __buf[w] -= X[d*i+u]*X[d*w+u];
            }
            __buf[w] /= __norm[i];
            __buf[w] /= __norm[w];
            __buf[w] += 1.0;
        }
        return __buf;
    }
};



/*! A class to compute the "mutual reachability" (Campello et al., 2015)
 *  distances from the i-th point to all given k points based on the "core"
 *  distances and a CDistance class instance.
 *
 *  References:
 *  ==========
 *
 *  [1] R. Campello, D. Moulavi, A. Zimek, J. Sander, Hierarchical density
 *  estimates for data clustering, visualization, and outlier detection,
 *  ACM Transactions on Knowledge Discovery from Data 10(1):5:1–5:51, 2015.
 *  doi: 10.1145/2733381.
 *
 */
template<class T>
struct CDistanceMutualReachability : public CDistance<T>  {
    const T* d_core;
    ssize_t n;
    CDistance<T>* d_pairwise;
    std::vector<T> buf;

    CDistanceMutualReachability(const T* d_core, ssize_t n, CDistance<T>* d_pairwise)
            : buf(n) {
        this->d_core = d_core;
        this->n = n;
        this->d_pairwise = d_pairwise;
    }

    CDistanceMutualReachability() : CDistanceMutualReachability(NULL, 0, NULL) { }

    virtual const T* operator()(ssize_t i, const ssize_t* M, ssize_t k) {
        // pragma omp parallel for inside::
        const T* d = (*d_pairwise)(i, M, k);

        // NO pragma omp parallel for -- should be fast, no need for OMP?
        for (ssize_t j=0; j<k; ++j)  { //
            // buf[w] = max{d[w],d_core[i],d_core[w]}
            ssize_t w = M[j];
            if (w == i) buf[w] = 0.0;
            else {
                buf[w] = d[w];
                if (d_core[i] > buf[w]) buf[w] = d_core[i];
                if (d_core[w] > buf[w]) buf[w] = d_core[w];
            }
        }
        return buf.data();
    }
};

#endif
