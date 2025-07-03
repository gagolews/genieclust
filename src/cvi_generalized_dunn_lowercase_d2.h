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

#ifndef __CVI_GENERALIZED_DUNN_LOWERCASE_D2_H
#define __CVI_GENERALIZED_DUNN_LOWERCASE_D2_H

#include "cvi.h"
#include "cvi_generalized_dunn_delta.h"

class LowercaseDelta2 : public LowercaseDelta1
{
public:
    LowercaseDelta2(
        EuclideanDistance& D,
        const CMatrix<FLOAT_T>& X,
        std::vector<Py_ssize_t>& L,
        std::vector<size_t>& count,
        size_t K,
        size_t n,
        size_t d,
        CMatrix<FLOAT_T>* centroids=nullptr
        )
    : LowercaseDelta1(D, X, L, count, K, n, d, centroids)
    {
        comparator = std::greater<FLOAT_T>();
    }

    virtual void recompute_all() {
        for (size_t i=0; i<K; ++i) {
            for (size_t j=i+1; j<K; ++j) {
                dist(i,j) = dist(j,i) = DistTriple(0, 0, 0); //the only reason for this method is initialization by 0, not INFINITY
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
};

class LowercaseDelta2Factory : public LowercaseDeltaFactory
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
               return new LowercaseDelta2(D, X, L, count, K, n, d, centroids);
           }
};

#endif
