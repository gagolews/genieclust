/*  Internal cluster validity indices
 *
 *  Code originally contributed in <https://github.com/gagolews/optim_cvi>,
 *  see https://doi.org/10.1016/j.ins.2021.10.004.
 *  Copyleft (C) 2020, Maciej Bartoszuk
 *
 *  For the 'genieclust' version:
 *  Copyleft (C) 2020-2026, Marek Gagolewski <https://www.gagolewski.com>
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

#ifndef __CVI_GENERALIZED_DUNN_LOWERCASE_D3_H
#define __CVI_GENERALIZED_DUNN_LOWERCASE_D3_H

#include "cvi.h"
#include "cvi_generalized_dunn_delta.h"

class LowercaseDelta3 : public LowercaseDelta
{
protected:
    CMatrix<FLOAT_T> dist_sums; /**< intra-cluster sums:
        dist(i,j) = min( X(u,), X(v,) ), X(u,) in C_i, X(v,) in C_j  (i!=j)
        */
    CMatrix<FLOAT_T> last_dist_sums;      ///< for undo()
    bool last_chg; ///< for undo() (was dist changed at all?)

public:
    LowercaseDelta3(
        EuclideanDistance& D,
        const CMatrix<FLOAT_T>& X,
        std::vector<Py_ssize_t>& L,
        std::vector<size_t>& count,
        size_t K,
        size_t n,
        size_t d,
        CMatrix<FLOAT_T>* centroids=nullptr
        )
    : LowercaseDelta(D, X, L,count,K,n,d,centroids),
    dist_sums(K, K),
    last_dist_sums(K, K),
    last_chg(false)
    {
    }
    virtual void before_modify(size_t i, Py_ssize_t /*j*/) {

        for (size_t u=0; u<K; ++u) {
            for (size_t v=u+1; v<K; ++v) {
                last_dist_sums(u,v) = last_dist_sums(v,u) = dist_sums(u,v);
            }
        }

        // subtract a contribution of the point i to the old cluster L[i]
        for (size_t u=0; u<n; ++u) {
            if(L[i] != L[u])
            {
                FLOAT_T d = sqrt(D(i, u));
                dist_sums(L[i], L[u]) = dist_sums(L[u], L[i]) = dist_sums(L[u], L[i]) - d;
            }
        }

        last_chg = true;

    }
    virtual void after_modify(size_t i, Py_ssize_t /*j*/) {
        // add a contribution of the point i to the new cluster L[i]
        for (size_t u=0; u<n; ++u) {
            if(L[i] != L[u])
            {
                FLOAT_T d = sqrt(D(i, u));
                dist_sums(L[i], L[u]) = dist_sums(L[u], L[i]) = dist_sums(L[u], L[i]) + d;
            }
        }
    }
    virtual void undo() {
        if (last_chg) {
            for (size_t i=0; i<K; ++i) {
                for (size_t j=i+1; j<K; ++j) {
                    dist_sums(i,j) = dist_sums(j,i) = last_dist_sums(i,j);
                }
            }
        }
    }
    virtual void recompute_all() {
        for (size_t i=0; i<K; ++i) {
            for (size_t j=i+1; j<K; ++j) {
                dist_sums(i,j) = dist_sums(j,i) = 0;
            }
        }

        for (size_t i=0; i<n-1; ++i) {
            for (size_t j=i+1; j<n; ++j) {
                FLOAT_T d = sqrt(D(i, j));
                if (L[i] != L[j]) {
                    dist_sums(L[i], L[j]) = dist_sums(L[j], L[i]) = dist_sums(L[j], L[i]) + d;
                }
            }
        }
    }

    virtual FLOAT_T compute(size_t k, size_t l) {
        return dist_sums(k, l)/((FLOAT_T)count[k]*count[l]);
    }

};

class LowercaseDelta3Factory : public LowercaseDeltaFactory
{
public:
    virtual bool IsCentroidNeeded() { return false; }

    virtual LowercaseDelta* create(EuclideanDistance& D,
           const CMatrix<FLOAT_T>& X,
           std::vector<Py_ssize_t>& L,
           std::vector<size_t>& count,
           size_t K,
           size_t n,
           size_t d,
           CMatrix<FLOAT_T>* centroids=nullptr) {
               return new LowercaseDelta3(D, X, L, count, K, n, d, centroids);
           }
};

#endif
