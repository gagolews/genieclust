/*  Various distances (Euclidean, mutual reachability distance, ...)
 *
 *  Copyleft (C) 2018-2020, Marek Gagolewski <https://www.gagolewski.com>
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
struct CDistancePrecomputedMatrix : public CDistance<T> {
    const T* dist;
    ssize_t n;

    /*!
     * @param dist n*n c_contiguous array, dist[i,j] is the distance between
     *    the i-th and the j-th point, the matrix is symmetric
     * @param n number of points
     */
    CDistancePrecomputedMatrix(const T* dist, ssize_t n) {
        this->n = n;
        this->dist = dist;
    }

    CDistancePrecomputedMatrix()
        : CDistancePrecomputedMatrix(NULL, 0) { }

    virtual const T* operator()(ssize_t i, const ssize_t* /*M*/, ssize_t /*k*/) {
        return &this->dist[i*n]; // the i-th row of dist
    }
};


/*! A class to "compute" the distances from the i-th point
 *  to all n points based on a pre-computed a vector-form
 *  c_contiguous distance vector.
 */
template<class T>
struct CDistancePrecomputedVector : public CDistance<T> {
    const T* dist;
    ssize_t n;
    std::vector<T> buf;

    /*!
     * @param dist n*(n-1)/2 c_contiguous vector, dist[ i*n - i*(i+1)/2 + w-i-1 ]
     *    where w is the distance between the i-th and the w-th point
     * @param n number of points
     */
    CDistancePrecomputedVector(const T* dist, ssize_t n)
            : buf(n)
    {
        this->n = n;
        this->dist = dist;
    }

    CDistancePrecomputedVector()
        : CDistancePrecomputedVector(NULL, 0) { }

    virtual const T* operator()(ssize_t i, const ssize_t* M, ssize_t k) {
        T* __buf = buf.data();
        for (ssize_t j=0; j<k; ++j) {
            ssize_t w = M[j];
            if (i == w)
                __buf[w] = 0.0;
            else if (i < w)
                __buf[w] = dist[ i*n - i*(i+1)/2 + w-i-1 ];
            else
                __buf[w] = dist[ w*n - w*(w+1)/2 + i-w-1 ];
        }
        return __buf;
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
    std::vector<T> buf;

    /*!
     * @param X n*d c_contiguous array
     * @param n number of points
     * @param d dimensionality
     */
    CDistanceEuclidean(const T* X, ssize_t n, ssize_t d)
            : buf(n)
    {
        this->n = n;
        this->d = d;
        this->X = X;
    }

    CDistanceEuclidean()
        : CDistanceEuclidean(NULL, 0, 0) { }

    virtual const T* operator()(ssize_t i, const ssize_t* M, ssize_t k) {
        T* __buf = buf.data();
        const T* x = X+d*i;

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (ssize_t j=0; j<k; ++j) {
            ssize_t w = M[j];
            const T* y = X+d*w;

            // or we could use the BLAS snrm2() for increased numerical
            // stability. are we building a rocket though?
            __buf[w] = 0.0;
            for (ssize_t u=0; u<d; ++u) {
                __buf[w] += (x[u]-y[u])*(x[u]-y[u]);
            }
            __buf[w] = sqrt(__buf[w]);
        }
        return __buf;
    }
};





/*! A class to compute the squared Euclidean distances from the i-th point
 *  to all given k points.
 */
template<class T>
struct CDistanceEuclideanSquared : public CDistance<T>  {
    const T* X;
    ssize_t n;
    ssize_t d;
    std::vector<T> buf;

    /*!
     * @param X n*d c_contiguous array
     * @param n number of points
     * @param d dimensionality
     */
    CDistanceEuclideanSquared(const T* X, ssize_t n, ssize_t d)
            : buf(n)
    {
        this->n = n;
        this->d = d;
        this->X = X;
    }

    CDistanceEuclideanSquared()
        : CDistanceEuclideanSquared(NULL, 0, 0) { }

    virtual const T* operator()(ssize_t i, const ssize_t* M, ssize_t k) {
        T* __buf = buf.data();
        const T* x = X+d*i;

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (ssize_t j=0; j<k; ++j) {
            ssize_t w = M[j];
            const T* y = X+d*w;

            // or we could use the BLAS snrm2() for increased numerical
            // stability. are we building a rocket though?
            __buf[w] = 0.0;
            for (ssize_t u=0; u<d; ++u) {
                __buf[w] += (x[u]-y[u])*(x[u]-y[u]);
            }
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
struct CDistanceMutualReachability : public CDistance<T>
{
    ssize_t n;
    CDistance<T>* d_pairwise;
    std::vector<T> buf;
    std::vector<T> d_core;

    CDistanceMutualReachability(const T* _d_core, ssize_t n, CDistance<T>* d_pairwise)
            : buf(n), d_core(_d_core, _d_core+n)
    {
        this->n = n;
        this->d_pairwise = d_pairwise;
    }

    CDistanceMutualReachability() : CDistanceMutualReachability(NULL, 0, NULL) { }

    virtual const T* operator()(ssize_t i, const ssize_t* M, ssize_t k) {
        // pragma omp parallel for inside::
        const T* d = (*d_pairwise)(i, M, k);
        T*  __buf = buf.data();

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (ssize_t j=0; j<k; ++j)  { //
            // buf[w] = max{d[w],d_core[i],d_core[w]}
            ssize_t w = M[j];
            if (w == i) __buf[w] = 0.0;
            else {
                __buf[w] = d[w];
                if (d_core[i] > __buf[w]) __buf[w] = d_core[i];
                if (d_core[w] > __buf[w]) __buf[w] = d_core[w];
            }
        }
        return __buf;
    }
};

#endif
