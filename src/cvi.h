/*  Internal cluster validity indices
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

#ifndef __CVI_H
#define __CVI_H

#include <cmath>
#include <algorithm>
#include <functional>
#include <vector>
#include <string>
#include "c_common.h"
#include "c_matrix.h"


template<class T>
inline T square(T x) { return x*x; }


/** Computes the squared Euclidean distance between two vectors.
 *
 * @param x c_contiguous vector of length d
 * @param y c_contiguous vector of length d
 * @param d length of both x and y
 * @return sum((x-y)^2)
 */
FLOAT_T distance_l2_squared(const FLOAT_T* x, const FLOAT_T* y, size_t d)
{
    FLOAT_T ret = 0.0;
    for (size_t i=0; i<d; i++) {
        ret += (x[i]-y[i])*(x[i]-y[i]);
    }
    return ret;
}




/** Stores the distance between the i-th and the j-th object,
 *  together with the indexes themselves;
 *  Used in DunnIndex; can be used for sorting.
 */
struct DistTriple {
    size_t i1;
    size_t i2;
    FLOAT_T d;

    DistTriple() { }

    DistTriple(size_t _i, size_t _j, FLOAT_T _d) {
        d = _d;
        if (_i < _j) {
            i1 = _i;
            i2 = _j;
        }
        else {
            i1 = _j;
            i2 = _i;
        }
    }

    bool operator<(const DistTriple& other) {
        return this->d < other.d;
    }

};






/** Computes Euclidean distances between pairs of points in the same dataset.
 *  Results might be precomputed for smaller datasets.
 */
class EuclideanDistance
{
private:
    const CMatrix<FLOAT_T>* X;
    std::vector<FLOAT_T> D;
    bool precomputed;
    bool squared;
    size_t n;
    size_t d;

public:
    EuclideanDistance(const CMatrix<FLOAT_T>* _X, bool _precompute=false, bool _square=false)
        : X(_X),
          D(_precompute?(_X->nrow()*(_X->nrow()-1)/2):0),
          precomputed(_precompute),
          squared(_square),
          n(_X->nrow()),
          d(_X->ncol())
    {
        if (!_precompute) return;

        size_t k = 0;
        for (size_t i=0; i<n-1; ++i) {
            for (size_t j=i+1; j<n; ++j) {
                D[k++] = distance_l2_squared(_X->row(i), _X->row(j), d);
            }
        }

        if (!_square) {
            for (k=0; k<D.size(); ++k)
                D[k] = sqrt(D[k]);
        }
    }


    const FLOAT_T operator()(size_t i, size_t j) const
    {
        if (i == j) return 0.0;
        if (precomputed) {
            if (i > j) std::swap(i, j);
            //GENIECLUST_ASSERT(i*n - i*(i+1)/2+(j-i-1) >= 0);
            //GENIECLUST_ASSERT(i*n - i*(i+1)/2+(j-i-1) < D.size());
            return D[i*n - i*(i+1)/2 + (j-i-1)];
        }
        else {
            if (squared)
                return distance_l2_squared(X->row(i), X->row(j), X->ncol());
            else
                return sqrt(distance_l2_squared(X->row(i), X->row(j), X->ncol()));
        }
    }
};




/** Base class for all the internal cluster validity indices implemented.
 */
class ClusterValidityIndex
{
protected:
    CMatrix<FLOAT_T> X;         ///< data matrix of size n*d
    std::vector<Py_ssize_t> L;    ///< current label vector of size n
    std::vector<size_t> count; ///< size of each of the K clusters
    const size_t K;           ///< number of clusters, max(L)
    const size_t n;            ///< number of points (for brevity of notation)
    const size_t d;            ///< dataset dimensionality (for brevity)
    const bool allow_undo;     ///< is the object's state preserved on modify()?

    Py_ssize_t last_i;             ///< for undo()
    Py_ssize_t last_j;            ///< for undo()

public:

    /** Constructor
     *
     * @param _X dataset
     * @param _K number of clusters
     * @param _allow_undo shall the object's state be preserved on a call to
     *      modify()?
     */
    ClusterValidityIndex(
            const CMatrix<FLOAT_T>& _X,
            const size_t _K,
            const bool _allow_undo
    )
        : X(_X), L(_X.nrow()), count(_K),
          K(_K), n(_X.nrow()), d(_X.ncol()), allow_undo(_allow_undo)
    {

    }



    /** Destructor
     */
    virtual ~ClusterValidityIndex() { }


    /** Returns the number of elements in the j-th cluster
     *
     * @param j
     * @return
     */
    size_t get_count(const size_t j) const
    {
        GENIECLUST_ASSERT(j >= 0 && j < K);
        return count[j];
    }

    /** Returns the i-th point's cluster label
     *
     * @param i
     * @return
     */
    Py_ssize_t get_label(const size_t i) const
    {
        GENIECLUST_ASSERT(i >= 0 && i < n);
        return L[i];
    }

    /** Returns the label vector
     *
     * @return
     */
    const std::vector<Py_ssize_t>& get_labels() const { return L; }


    /** Returns the number of clusters
     *
     * @return
     */
    const size_t get_K() const { return K; }


    /** Returns the number of data points
     *
     * @return
     */
    const size_t get_n() const { return n; }


    /** Assigns a new label vector
     *
     * @param _L
     */
    virtual void set_labels(const std::vector<Py_ssize_t>& _L)
    {
        GENIECLUST_ASSERT(X.nrow() == _L.size());
        for (size_t j=0; j<K; ++j) {
            count[j] = 0;
        }

        for (size_t i=0; i<n; ++i) {
            GENIECLUST_ASSERT(_L[i] >= 0 && _L[i] < (Py_ssize_t)K);
            L[i] = _L[i];
            count[ L[i] ]++;
        }

        for (size_t j=0; j<K; ++j) {
            GENIECLUST_ASSERT(count[j] > 0);
        }
    }


    /** Makes the i-th point a member of the j-th cluster
     *
     * The inheriting classes can overload this method and
     * compute the cluster validity index incrementally.
     *
     * @param i
     * @param j
     */
    virtual void modify(size_t i, Py_ssize_t j)
    {
        GENIECLUST_ASSERT(i >= 0 && i < n);
        GENIECLUST_ASSERT(j >= 0 && j < (Py_ssize_t)K);
        GENIECLUST_ASSERT(L[i] >= 0 && L[i] < (Py_ssize_t)K);
        GENIECLUST_ASSERT(count[L[i]] > 0);
        GENIECLUST_ASSERT(L[i] != j);


        if (allow_undo) {
            last_i = i;
            last_j = L[i];
        }

        count[L[i]]--;
        L[i] = j;
        count[L[i]]++;
    }

    /** Computes the cluster validity index for the current label vector, L
     */
    virtual FLOAT_T compute() = 0;




    /** Cancels the most recent modify() operation.
     */
    virtual void undo()
    {
        GENIECLUST_ASSERT(allow_undo);

        count[L[last_i]]--;
        L[last_i] = last_j;
        count[L[last_i]]++;
    }
};



/** Represents a cluster validity index that is based
 * on the notion of the clusters' centroid.
 */
class CentroidsBasedIndex : public ClusterValidityIndex
{
protected:
    CMatrix<FLOAT_T> centroids;     ///< centroids of all the clusters, size K*d


public:
    // Described in the base class
    CentroidsBasedIndex(
            const CMatrix<FLOAT_T>& _X,
            const size_t _K,
            const bool _allow_undo)
        : ClusterValidityIndex(_X, _K, _allow_undo),
          centroids(K, d)
    {
        ;
    }


    // Described in the base class
    virtual void set_labels(const std::vector<Py_ssize_t>& _L)
    {
        ClusterValidityIndex::set_labels(_L); // sets L and count

        for (size_t i=0; i<K; ++i) {
            for (size_t j=0; j<d; ++j) {
                centroids(i, j) = 0.0;
            }
        }
        for (size_t i=0; i<n; ++i) {
            for (size_t j=0; j<d; ++j) {
                centroids(L[i], j) += X(i, j);
            }
        }
        for (size_t i=0; i<K; ++i) {
            for (size_t j=0; j<d; ++j) {
                centroids(i, j) /= (FLOAT_T) count[i];
            }
        }
    }


    // Described in the base class
    virtual void modify(size_t i, Py_ssize_t j)
    {
        Py_ssize_t tmp = L[i];
        // tmp = old label for the i-th point
        // j   = new label for the i-th point

        // -----------------------------


        for (size_t k=0; k<d; ++k) {
            centroids(tmp, k) *= (FLOAT_T) count[tmp];
            centroids(tmp, k) -= X(i,k);
            centroids(tmp, k) /= (FLOAT_T) (count[tmp]-1.0);

            centroids(j, k)   *= (FLOAT_T) count[j];
            centroids(j, k)   += X(i,k);
            centroids(j, k)   /= (FLOAT_T) (count[j]+1.0);
        }

        ClusterValidityIndex::modify(i, j); // sets L[i]=j and updates count

        // -----------------------------
    }


    // Described in the base class
    virtual void undo()
    {
        size_t tmp = L[last_i];
        for (size_t k=0; k<d; ++k) {
            centroids(tmp, k) *= (FLOAT_T) count[tmp];
            centroids(tmp, k) -= X(last_i,k);
            centroids(tmp, k) /= (FLOAT_T) (count[tmp]-1.0);

            centroids(last_j, k)   *= (FLOAT_T) count[last_j];
            centroids(last_j, k)   += X(last_i,k);
            centroids(last_j, k)   /= (FLOAT_T) (count[last_j]+1.0);
        }

        ClusterValidityIndex::undo();
    }

};



/** Represents a cluster validity index that is based
 * on the notion of M-nearest neighbours between the input points,
 * for some M>=1.
 */
class NNBasedIndex : public ClusterValidityIndex
{
protected:
    const size_t M;       ///< number of nearest neighbours
    CMatrix<FLOAT_T> dist; ///< dist(i, j) is the L2 distance between i and its j-th NN
    CMatrix<size_t> ind;   ///< ind(i, j) is the index of the j-th NN of i

public:
    // Described in the base class
    NNBasedIndex(
            const CMatrix<FLOAT_T>& _X,
            const size_t _K,
            const bool _allow_undo,
            const size_t _M)
        : ClusterValidityIndex(_X, _K, _allow_undo),
          M((_M<=n-1)?_M:(n-1)),
          dist(n, M, INFTY),
          ind(n, M, n)
    {
        GENIECLUST_ASSERT(M>0 && M<n);

        for (size_t i=0; i<n-1; ++i) {
            for (size_t j=i+1; j<n; ++j) {
                FLOAT_T dij = sqrt(distance_l2_squared(X.row(i), X.row(j), d));

                if (dij < dist(i, M-1)) {
                    // j may be amongst M NNs of i
                    size_t l = M-1;
                    while (l > 0 && dij < dist(i, l-1)) {
                        dist(i, l) = dist(i, l-1);
                        ind(i, l)  = ind(i, l-1);
                        l--;
                    }
                    dist(i, l) = dij;
                    ind(i, l)  = j;
                }

                if (dij < dist(j, M-1)) {
                    // i may be amongst M NNs of j
                    size_t l = M-1;
                    while (l > 0 && dij < dist(j, l-1)) {
                        dist(j, l) = dist(j, l-1);
                        ind(j, l)  = ind(j, l-1);
                        l--;
                    }
                    dist(j, l) = dij;
                    ind(j, l)  = i;
                }
            }
        }
    }

};


#endif
