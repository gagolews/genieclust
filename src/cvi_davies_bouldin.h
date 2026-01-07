/*  Internal cluster validity indices
 *
 *  Code originally contributed in <https://github.com/gagolews/optim_cvi>,
 *  see https://doi.org/10.1016/j.ins.2021.10.004.
 *
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

#ifndef __CVI_DAVIES_BOULDIN_H
#define __CVI_DAVIES_BOULDIN_H

#include "cvi.h"



/** The Negated Davies-Bouldin Index
 *
 *  See Def.5 in (Davies, Bouldin, 1979). Singletons are assumed
 *  to have infinite dispersion (see discussion on p.225 therein).
 *
 *  D.L. Davies, D.W. Bouldin,
 *  A cluster separation measure,
 *  IEEE Transactions on Pattern Analysis and Machine Intelligence. PAMI-1 (2),
 *  1979, pp. 224-227, doi:10.1109/TPAMI.1979.4766909
 *
 *  See the following paper for the formula and further discussion:
 *  M. Gagolewski, M. Bartoszuk, A. Cena,
 *  Are cluster validity measures (in)valid?, Information Sciences 581,
 *  620-636, 2021, DOI:10.1016/j.ins.2021.10.004
 */
class DaviesBouldinIndex : public CentroidsBasedIndex
{
protected:
    std::vector<FLOAT_T> R; ///< average distance between
                            ///< cluster centroids and their members

public:
    // Described in the base class
    DaviesBouldinIndex(
           const CMatrix<FLOAT_T>& _X,
           const size_t _K,
           const bool _allow_undo=false)
        : CentroidsBasedIndex(_X, _K, _allow_undo),
          R(_K)
    {

    }

//     // Described in the base class
//     virtual void set_labels(const std::vector<Py_ssize_t>& _L)
//     {
//         CentroidsBasedIndex::set_labels(_L); // sets L, count and centroids
//     }


//     // Described in the base class
//     virtual void modify(size_t i, Py_ssize_t j)
//     {
//         // sets L[i]=j and updates count as well as centroids
//         CentroidsBasedIndex::modify(i, j);
//     }


//     // Described in the base class
//     virtual void undo() {
//         CentroidsBasedIndex::undo();
//     }


    // Described in the base class
    virtual FLOAT_T compute()
    {
        // Compute the average distances between the cluster centroids
        // and their members.
        // The centroids are up-to-date.
        for (size_t i=0; i<K; ++i) {
            if (count[i] <= 1)  // singletons not permitted
                return -INFINITY;  // negative!!
            R[i] = 0.0;
        }
        for (size_t i=0; i<n; ++i) {
            FLOAT_T dist = 0.0;
            for (size_t u=0; u<d; ++u) {
                dist += square(centroids(L[i],u)-X(i,u));
            }
            R[L[i]] += sqrt(dist);
        }
        for (size_t i=0; i<K; ++i) R[i] /= (FLOAT_T)count[i];

        FLOAT_T ret = 0.0;
        for (size_t i=0; i<K; ++i) {
            FLOAT_T max_r = 0.0;
            for (size_t j=0; j<K; ++j) {
                if (j == i) continue;

                // compute the distance between the i-th and the j-th centroid:
                FLOAT_T cur_d = 0.0;
                for (size_t u=0; u<d; ++u)
                    cur_d += square(centroids(i,u)-centroids(j,u));
                cur_d = sqrt(cur_d);

                FLOAT_T cur_r = (R[i]+R[j])/cur_d;
                if (cur_r > max_r)
                    max_r = cur_r;
            }
            ret += max_r;
        }
        ret = -ret/(FLOAT_T)K; // negative!!
        GENIECLUST_ASSERT(ret < 1e-12);
        return ret;
    }

};



#endif
