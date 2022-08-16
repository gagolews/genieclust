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

#ifndef __CVI_SILHOUETTE_H
#define __CVI_SILHOUETTE_H

#include "cvi.h"





/** The Silhouette Coefficient
 *
 *  Overall average per-point silhouette scores (widths=false)
 *  or mean of the cluster average silhouette widths (widths=true) as defined
 *  in Sec.2 of (Rousseeuw, 1987).
 *
 *
 *  P.J. Rousseeuw, Silhouettes: A graphical aid to the interpretation and
 *  validation of cluster analysis, Computational and Applied Mathematics 20,
 *  1987, pp. 53-65, doi:10.1016/0377-0427(87)90125-7.
 *
 *
 *  See the following paper for the formula and further discussion:
 *  M. Gagolewski, M. Bartoszuk, A. Cena,
 *  Are cluster validity measures (in)valid?, Information Sciences 581,
 *  620-636, 2021, DOI:10.1016/j.ins.2021.10.004
 */
class SilhouetteIndex : public ClusterValidityIndex
{
protected:
    std::vector<FLOAT_T> A; ///< cluster "radius"
    std::vector<FLOAT_T> B; ///< distance to "nearest" cluster
    CMatrix<FLOAT_T> C;      ///< auxiliary array; Let C(i,j) == sum of
                ///< distances between X(i,:) and all points in the j-th cluster
    EuclideanDistance D;    ///< D(i, j) gives the Euclidean distance
                ///< between X(i,:) and X(j,:) /can be precomputed for speed/
    bool widths;

public:
    // Described in the base class
    SilhouetteIndex(
           const CMatrix<FLOAT_T>& _X,
           const size_t _K,
           const bool _allow_undo=false,
           bool _widths=false)
        : ClusterValidityIndex(_X, _K, _allow_undo),
          A(n),
          B(n),
          C(n, K),
          D(&X, n<=CVI_MAX_N_PRECOMPUTE_DISTANCE)
    {
        widths = _widths;
    }

    // Described in the base class
    virtual void set_labels(const std::vector<ssize_t>& _L)
    {
        ClusterValidityIndex::set_labels(_L); // sets L, count and centroids

        for (size_t i=0; i<n; ++i) {
            for (size_t j=0; j<K; ++j) C(i, j) = 0.0;
        }

        for (size_t i=0; i<n-1; ++i) {
            for (size_t j=i+1; j<n; ++j) {
                FLOAT_T dist = D(i, j);
                C(i, L[j]) += dist;
                C(j, L[i]) += dist;
            }
        }
    }


    // Described in the base class
    virtual void modify(size_t i, ssize_t j)
    {
        for (size_t u=0; u<n; ++u) {
            FLOAT_T dist = D(i, u);
            C(u, L[i]) -= dist;
            C(u, j)    += dist;
        }


        // sets L[i]=j and updates count as well as centroids
        ClusterValidityIndex::modify(i, j);
    }


    // Described in the base class
    virtual void undo()
    {
        for (size_t u=0; u<n; ++u) {
            double dist = D(last_i, u);
            C(u, L[last_i]) -= dist;
            C(u, last_j)    += dist;
        }

        ClusterValidityIndex::undo();
    }


    // Described in the base class
    virtual FLOAT_T compute()
    {
        for (size_t i=0; i<n; ++i) {
            // Let C(i,j) == sum of distances between X(i,) and all points in the j-th cluster
            B[i] = INFTY;
            for (ssize_t j=0; j<(ssize_t)K; ++j) {
                if (j == L[i]) {
                    A[i] = C(i,j)/(FLOAT_T)(count[j]-1);
                }
                else {
                    if (C(i,j)/(FLOAT_T)(count[j]) < B[i])
                        B[i] = C(i,j)/(FLOAT_T)(count[j]);
                }
            }
        }

        // compute the mean of silhouette scores of each point
        FLOAT_T ret = 0.0;
        if (widths) {
            size_t num_singletons = 0;
            for (size_t i=0; i<n; ++i) {
                if (count[L[i]] > 1) { // silhouette score of 0 for singleton clusters
                    FLOAT_T cur = (B[i]-A[i])/std::max(B[i], A[i]);
                    ret += cur/(FLOAT_T)count[L[i]];
                }
                else
                    num_singletons++;
            }
            ret = ret/(FLOAT_T)(K-num_singletons);
        }
        else {
            for (size_t i=0; i<n; ++i) {
                if (count[L[i]] > 1) { // silhouette score of 0 for singleton clusters
                    FLOAT_T cur = (B[i]-A[i])/std::max(B[i], A[i]);
                    ret += cur;
                }
            }
            ret = ret/(FLOAT_T)n;
        }

        GENIECLUST_ASSERT(std::fabs(ret) < 1.0+1e-12);

        return ret;
    }
};



#endif
