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

#ifndef __CVI_GENERALIZED_DUNN_LOWERCASE_D6_H
#define __CVI_GENERALIZED_DUNN_LOWERCASE_D6_H

#include "cvi.h"
#include "cvi_generalized_dunn_delta.h"

class LowercaseDelta6 : public LowercaseDelta
{
protected:
    CMatrix<DistTriple> dist; /**< intra-cluster distances:
        dist(i,j) = min( X(u,), X(v,) ), X(u,) in C_i, X(v,) in C_j  (i!=j)
        */
    CMatrix<DistTriple> last_dist;      ///< for undo()
    std::vector<DistTriple> min_dists; ///< helper for calculating minimum distances to clusters for a single point
    bool last_chg; ///< for undo() (was dist changed at all?)
    bool needs_recompute; ///< for before and after modify
    Py_ssize_t cluster1;
    Py_ssize_t cluster2;

public:
    LowercaseDelta6(
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
    last_dist(K, K),
    min_dists(K)
    { }
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

        cluster1 = L[i];
    }
    virtual void after_modify(size_t i, Py_ssize_t /*j*/) {
        if (needs_recompute) {
            last_chg = true;
            recompute_all();
        }
        else {
            //recompute_all();
            last_chg = false;
            cluster2 = L[i];

            for (Py_ssize_t i1=0; i1<(Py_ssize_t)K; ++i1) {
                for (Py_ssize_t j=i1+1; j<(Py_ssize_t)K; ++j) {
                    if(i1 == cluster1 || i1 == cluster2 || j == cluster1 || j == cluster2)
                        dist(i1,j) = dist(j,i1) = DistTriple(0, 0, 0);
                }
            }


            for (size_t i1=0; i1<n; ++i1) {
                if (L[i1] != cluster1 && L[i1] != cluster2)
                    continue;
                // for every point i we find its minimum distance to every other cluster
                std::fill(min_dists.begin(), min_dists.end(), DistTriple(0, 0, INFINITY));
                for (size_t j=0; j<n; ++j) {
                    if (L[i1] != L[j]) {
                        FLOAT_T d = D(i1, j);
                        if (d < min_dists[L[j]].d)
                            min_dists[L[j]] = DistTriple(i1, j, d);
                    }
                }

                // update maximum minimum distance on cluster level
                for (Py_ssize_t l=0; l<(Py_ssize_t)K; ++l) {
                    if (l != L[i1] && dist(L[i1],l).d < min_dists[l].d) {
                        dist(L[i1],l) = min_dists[l];
                        last_chg = true;
                    }
                }
            }

            for (size_t i1=0; i1<n; ++i1) {
                // for every point i we find its minimum distance to every other cluster
                std::fill(min_dists.begin(), min_dists.end(), DistTriple(0, 0, INFINITY));
                for (size_t j=0; j<n; ++j) {
                    if (L[j] != cluster1 && L[j] != cluster2)
                        continue;
                    if (L[i1] != L[j]) {
                        FLOAT_T d = D(i1, j);
                        if (d < min_dists[L[j]].d)
                            min_dists[L[j]] = DistTriple(i1, j, d);
                    }
                }

                // update maximum minimum distance on cluster level
                for (Py_ssize_t l=0; l<(Py_ssize_t)K; ++l) {
                    if (l != cluster1 && l != cluster2)
                        continue;

                    if ( l != L[i1] && dist(L[i1],l).d < min_dists[l].d) {
                        dist(L[i1],l) = min_dists[l];
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
                dist(i,j) = dist(j,i) = DistTriple(0, 0, 0);
            }
        }


        for (size_t i=0; i<n; ++i) {
            // for every point i we find its minimum distance to every other cluster
            std::fill(min_dists.begin(), min_dists.end(), DistTriple(0, 0, INFINITY));
            for (size_t j=0; j<n; ++j) {
                if (L[i] != L[j]) {
                    FLOAT_T d = D(i, j);
                    if (d < min_dists[L[j]].d)
                        min_dists[L[j]] = DistTriple(i, j, d);
                }
            }

            // update maximum minimum distance on cluster level
            for (Py_ssize_t l=0; l<(Py_ssize_t)K; ++l) {
                if ( l != L[i] && dist(L[i],l).d < min_dists[l].d) {
                    dist(L[i],l) = min_dists[l];
                }
            }
        }
    }

    virtual FLOAT_T compute(size_t k, size_t l) {
        FLOAT_T maxx = std::max(dist(k, l).d, dist(l, k).d);
        return sqrt(maxx);
    }

};

class LowercaseDelta6Factory : public LowercaseDeltaFactory
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
               return new LowercaseDelta6(D, X, L, count, K, n, d, centroids);
           }
};

#endif
