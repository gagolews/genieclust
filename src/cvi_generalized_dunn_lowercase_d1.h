/*  Internal cluster validity indices
 *
 *  Code originally contributed in <https://github.com/gagolews/optim_cvi>,
 *  see https://doi.org/10.1016/j.ins.2021.10.004.
 *  Copyleft (C) 2020, Maciej Bartoszuk
 *
 *  For the 'genieclust' version:
 *  Copyleft (C) 2020-2025, Marek Gagolewski <https://www.gagolewski.com>
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

#ifndef __CVI_GENERALIZED_DUNN_LOWERCASE_D1_H
#define __CVI_GENERALIZED_DUNN_LOWERCASE_D1_H

#include "cvi.h"
#include "cvi_generalized_dunn_delta.h"

class LowercaseDelta1 : public LowercaseDelta
{
protected:
    CMatrix<DistTriple> dist; /**< intra-cluster distances:
        dist(i,j) = min( X(u,), X(v,) ), X(u,) in C_i, X(v,) in C_j  (i!=j)
        */
    CMatrix<DistTriple> last_dist;      ///< for undo()
    bool last_chg; ///< for undo() (was dist changed at all?)
    bool needs_recompute; ///< for before and after modify
    std::function< bool(FLOAT_T, FLOAT_T) > comparator;

public:
    LowercaseDelta1(
        EuclideanDistance& D,
        const CMatrix<FLOAT_T>& X,
        std::vector<Py_ssize_t>& L,
        std::vector<size_t>& count,
        size_t K,
        size_t n,
        size_t d,
        CMatrix<FLOAT_T>* centroids=nullptr
        )
    : LowercaseDelta(D, X, L, count,K,n,d,centroids),
    dist(K, K),
    last_dist(K, K)
    {
        comparator = std::less<FLOAT_T>();

    }
    virtual void before_modify(size_t i, Py_ssize_t /*j*/) {
        needs_recompute = false;
        for (size_t u=0; u<K; ++u) {

            for (size_t v=u+1; v<K; ++v) {
                // if the point being modified determines intra-cluster distance:
                if (dist(u,v).i1 == i || dist(u,v).i2 == i)
                    needs_recompute = true;

                last_dist(u,v) = last_dist(v,u) = dist(u,v);
            }
        }
    }
    virtual void after_modify(size_t i, Py_ssize_t /*j*/) {
        if (needs_recompute) {
            last_chg = true;
            recompute_all();
        }
        else {
            last_chg = false;
            for (size_t u=0; u<n; ++u) {
                if (i == u) continue;

                FLOAT_T d = D(i, u);
                if (L[i] != L[u]) {
                    if (comparator(d, dist(L[i], L[u]).d)) {
                        dist(L[i], L[u]) = dist(L[u], L[i]) = DistTriple(i, u, d);
                        last_chg = true;
                    }
                }
            }
        }
    }
    virtual void undo() {
        if (last_chg) {
            for (size_t i=0; i<K; ++i) {
                for (size_t j=i+1; j<K; ++j) {
                    dist(i,j) = dist(j,i) = last_dist(i,j);
                }
            }
        }
    }
    virtual void recompute_all() {
        for (size_t i=0; i<K; ++i) {
            for (size_t j=i+1; j<K; ++j) {
                dist(i,j) = dist(j,i) = DistTriple(0, 0, INFINITY);
            }
        }

        for (size_t i=0; i<n-1; ++i) {
            for (size_t j=i+1; j<n; ++j) {
                FLOAT_T d = D(i, j);
                if (L[i] != L[j]) {
                    if (comparator(d, dist(L[i], L[j]).d))
                        dist(L[i], L[j]) = dist(L[j], L[i]) = DistTriple(i, j, d);
                }
            }
        }
    }

    virtual FLOAT_T compute(size_t k, size_t l) {
        return sqrt(dist(k, l).d);
    }

};

class LowercaseDelta1Factory : public LowercaseDeltaFactory
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
               return new LowercaseDelta1(D, X, L, count, K, n, d, centroids);
           }
};

#endif
