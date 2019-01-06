/*  Various distances.
 *
 *  Copyright (C) 2018-2019 Marek.Gagolewski.com
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

#include <vector>
#include <cmath>


inline double square(double x) { return x*x; }



/*! Abstract base class for all distances */
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
    virtual const double* operator()(ssize_t i, const ssize_t* M, ssize_t k) = 0;
};



/*! A class to "compute" the distances from the i-th point
 *  to all n points based on a pre-computed n*n symmetric,
 *  complete pairwise matrix.
 */
struct CDistanceCompletePrecomputed : public CDistance {
    const double* dist;
    ssize_t n;

    /*!
     * @param dist n*n c_contiguous array, dist[i,j] is the distance between
     *    the i-th and the j-th point, the matrix is symmetric
     * @param n number of points
     */
    CDistanceCompletePrecomputed(const double* dist, ssize_t n) {
        this->n = n;
        this->dist = dist;
    }

    CDistanceCompletePrecomputed()
        : CDistanceCompletePrecomputed(NULL, 0) { }

    virtual const double* operator()(ssize_t i, const ssize_t* M, ssize_t k) {
        return &this->dist[i*n]; // the i-th row of dist
    }
};





/*! A class to compute the Euclidean distances from the i-th point
 *  to all given k points.
 */
struct CDistanceEuclidean : public CDistance  {
    const double* X;
    ssize_t n;
    ssize_t d;
    std::vector<double> buf;
    // std::vector<double> sqnorm;

    /*!
     * @param X n*d c_contiguous array
     * @param n number of points
     * @param d dimensionality
     */
    CDistanceEuclidean(const double* X, ssize_t n, ssize_t d)
            : buf(n)//, sqnorm(n)
    {
        this->n = n;
        this->d = d;
        this->X = X;

        // ssize_t cur = 0;
        // for (ssize_t i=0; i<n; ++i) {
        //     sqnorm[i] = 0.0;
        //     for (ssize_t u=0; u<d; ++u) {
        //         sqnorm[i] += X[cur]*X[cur];
        //         cur++;
        //     }
        // }
    }

    CDistanceEuclidean()
        : CDistanceEuclidean(NULL, 0, 0) { }

    virtual const double* operator()(ssize_t i, const ssize_t* M, ssize_t k) {
        for (ssize_t j=0; j<k; ++j) {
            ssize_t w = M[j];
            // if (w < 0 || w >= n)
            //     throw std::runtime_error("ASSERT FAIL: CDistanceEuclidean");
            buf[w] = 0.0;

            for (ssize_t u=0; u<d; ++u) {
                buf[w] += square(X[d*i+u]-X[d*w+u]);
            }

            // // did you know that (x-y)**2 = x**2 + y**2 - 2*x*y ?
            // for (ssize_t u=0; u<d; ++u) {
            //     buf[w] -= X[d*i+u]*X[d*w+u];
            // }
            // buf[w] = 2.0*buf[w]+sqnorm[i]+sqnorm[w];

            buf[w] = sqrt(buf[w]);
        }
        return buf.data();
    }
};



/*! A class to compute the CDistanceManhattan distances from the i-th point
 *  to all given k points.
 */
struct CDistanceManhattan : public CDistance  {
    const double* X;
    ssize_t n;
    ssize_t d;
    std::vector<double> buf;

    /*!
     * @param X n*d c_contiguous array
     * @param n number of points
     * @param d dimensionality
     */
    CDistanceManhattan(const double* X, ssize_t n, ssize_t d)
            : buf(n)
    {
        this->n = n;
        this->d = d;
        this->X = X;
    }

    CDistanceManhattan()
        : CDistanceManhattan(NULL, 0, 0) { }

    virtual const double* operator()(ssize_t i, const ssize_t* M, ssize_t k) {
        for (ssize_t j=0; j<k; ++j) {
            ssize_t w = M[j];
            // if (w < 0 || w >= n)
            //     throw std::ruantime_error("ASSERT FAIL: CDistanceManhattan");
            buf[w] = 0.0;

            for (ssize_t u=0; u<d; ++u) {
                buf[w] += fabs(X[d*i+u]-X[d*w+u]);
            }
        }
        return buf.data();
    }
};



/*! A class to compute the cosine distances from the i-th point
 *  to all given k points.
 */
struct CDistanceCosine : public CDistance  {
    const double* X;
    ssize_t n;
    ssize_t d;
    std::vector<double> buf;
    std::vector<double> norm;

    /*!
     * @param X n*d c_contiguous array
     * @param n number of points
     * @param d dimensionality
     */
    CDistanceCosine(const double* X, ssize_t n, ssize_t d)
            : buf(n), norm(n)
    {
        this->n = n;
        this->d = d;
        this->X = X;

        ssize_t cur = 0;
        for (ssize_t i=0; i<n; ++i) {
            norm[i] = 0.0;
            for (ssize_t u=0; u<d; ++u) {
                norm[i] += X[cur]*X[cur];
                cur++;
            }
            norm[i] = sqrt(norm[i]);
        }
    }

    CDistanceCosine()
        : CDistanceCosine(NULL, 0, 0) { }

    virtual const double* operator()(ssize_t i, const ssize_t* M, ssize_t k) {
        for (ssize_t j=0; j<k; ++j) {
            ssize_t w = M[j];
            // if (w < 0 || w >= n)
            //     throw std::runtime_error("ASSERT FAIL: CDistanceEuclidean");
            buf[w] = 0.0;

            for (ssize_t u=0; u<d; ++u) {
                buf[w] -= X[d*i+u]*X[d*w+u];
            }
            buf[w] /= norm[i];
            buf[w] /= norm[w];
            buf[w] += 1.0;
        }
        return buf.data();
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
struct CDistanceMutualReachability : public CDistance  {
    const double* d_core;
    ssize_t n;
    CDistance* d_pairwise;
    std::vector<double> buf;

    CDistanceMutualReachability(const double* d_core, ssize_t n, CDistance* d_pairwise)
            : buf(n) {
        this->d_core = d_core;
        this->n = n;
        this->d_pairwise = d_pairwise;
    }

    CDistanceMutualReachability() : CDistanceMutualReachability(NULL, 0, NULL) { }

    virtual const double* operator()(ssize_t i, const ssize_t* M, ssize_t k) {
        const double* d = (*d_pairwise)(i, M, k);
        for (ssize_t j=0; j<k; ++j)  {
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
