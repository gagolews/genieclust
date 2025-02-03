/*  Internal cluster validity indices
 *
 *  Code originally contributed in <https://github.com/gagolews/optim_cvi>,
 *  see https://doi.org/10.1016/j.ins.2021.10.004.
 *  Copyleft (C) 2020, Maciej Bartoszuk <http://bartoszuk.rexamine.com>
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

#ifndef __CVI_GENERALIZED_DUNN_UPPERCASE_D1_H
#define __CVI_GENERALIZED_DUNN_UPPERCASE_D1_H

#include "cvi.h"
#include "cvi_generalized_dunn_delta.h"

class UppercaseDelta1 : public UppercaseDelta
{
protected:
    std::vector<DistTriple> diam; /**< cluster diameters:
        diam[i] = max( X(u,), X(v,) ), X(u,), X(v,) in C_i
        */
    std::vector<DistTriple> last_diam; ///< for undo()
    bool last_chg; ///< for undo() (was diam changed at all?)
    bool needs_recompute; ///< for before and after modify
public:
    UppercaseDelta1(
        EuclideanDistance& D,
        const CMatrix<FLOAT_T>& X,
        std::vector<Py_ssize_t>& L,
        std::vector<size_t>& count,
        size_t K,
        size_t n,
        size_t d,
        CMatrix<FLOAT_T>* centroids=nullptr
        )
    : UppercaseDelta(D,X,L,count,K,n,d,centroids),
    diam(K),
    last_diam(K)
    { }
    virtual void before_modify(size_t i, Py_ssize_t j) {
        needs_recompute = false;
        for (size_t u=0; u<K; ++u) {
            last_diam[u] = diam[u];

            // if the point being modified determines its cluster's diameter:
            if (diam[u].i1 == i || diam[u].i2 == i)
                needs_recompute = true;
        }

    }

    virtual void after_modify(size_t i, Py_ssize_t j) {
        if (needs_recompute) {
            last_chg = true;
            recompute_all();
        }
        else {
            last_chg = false;
            for (size_t u=0; u<n; ++u) {
                if (i == u) continue;

                FLOAT_T d = D(i, u);
                if (L[i] == L[u]) {
                    if (d > diam[L[i]].d) {
                        diam[L[i]] = DistTriple(i, u, d);
                        last_chg = true;
                    }
                }
            }
        }
    }

    virtual void undo(){
        if (last_chg) {
            for (size_t i=0; i<K; ++i) {
                diam[i] = last_diam[i];
            }
        }
    }

    virtual void recompute_all(){
        for (size_t i=0; i<K; ++i) {
            diam[i] = DistTriple(0, 0, 0.0);
        }

        for (size_t i=0; i<n-1; ++i) {
            for (size_t j=i+1; j<n; ++j) {
                double d = D(i, j);
                if (L[i] == L[j]) {
                    if (d > diam[L[i]].d)
                        diam[L[i]] = DistTriple(i, j, d);
                }
            }
        }
    }

    virtual FLOAT_T compute(size_t k){
        return sqrt(diam[k].d);
    }
};


class UppercaseDelta1Factory : public UppercaseDeltaFactory
{
public:
    virtual bool IsCentroidNeeded() { return false; }

    virtual UppercaseDelta* create(EuclideanDistance& D,
           const CMatrix<FLOAT_T>& X,
           std::vector<Py_ssize_t>& L,
           std::vector<size_t>& count,
           size_t K,
           size_t n,
           size_t d,
           CMatrix<FLOAT_T>* centroids=nullptr) {
               return new UppercaseDelta1(D, X, L, count, K, n, d, centroids);
           }
};

#endif
