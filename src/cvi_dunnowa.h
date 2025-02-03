/*  Internal cluster validity indices
 *
 *  Code originally contributed in <https://github.com/gagolews/optim_cvi>,
 *  see https://doi.org/10.1016/j.ins.2021.10.004.
 *
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

#ifndef __CVI_DUNNOWA_H
#define __CVI_DUNNOWA_H

#include "cvi.h"
#include "c_argfuns.h"

#define OWA_ERROR 0
#define OWA_MEAN 1
#define OWA_CONST 666
#define OWA_MIN 2
#define OWA_MAX 3
#define OWA_SMIN_START 100000
#define OWA_SMIN_LIMIT 199999
#define OWA_SMAX_START 200000
#define OWA_SMAX_LIMIT 299999


int DuNNOWA_get_OWA(std::string owa_name)
{
    if      (owa_name == "Mean")   return OWA_MEAN;
    else if (owa_name == "Min")    return OWA_MIN;
    else if (owa_name == "Max")    return OWA_MAX;
    else if (owa_name == "Const")  return OWA_CONST;
    else if (owa_name.substr(0, 5) == "SMin:") {
        int delta = std::atoi(owa_name.substr(5).c_str());
        GENIECLUST_ASSERT(delta > 0 && delta < OWA_SMIN_LIMIT-OWA_SMIN_START);
        return OWA_SMIN_START+delta;
    }
    else if (owa_name.substr(0, 5) == "SMax:") {
        int delta = std::atoi(owa_name.substr(5).c_str());
        GENIECLUST_ASSERT(delta > 0 && delta < OWA_SMAX_LIMIT-OWA_SMAX_START);
        return OWA_SMAX_START+delta;
    }
    else {
        return OWA_ERROR;
    };
}


#define REC_SQRT_2_PI 0.3989422804014326779399460599343818684758586311649346576659258296

FLOAT_T dnorm(FLOAT_T x, FLOAT_T m, FLOAT_T s) {
    return REC_SQRT_2_PI*exp(-0.5*square((x-m)/s))/s;
}


/** OWA-based Dunn-like Indices Based on Near Neighbours
 *
 *
 *  Proposed by Gagolewski
 *
 *  See the following paper for the formula and further discussion:
 *  M. Gagolewski, M. Bartoszuk, A. Cena,
 *  Are cluster validity measures (in)valid?, Information Sciences 581,
 *  620-636, 2021, DOI:10.1016/j.ins.2021.10.004
 *
 *  Inspired by generalised Dunn indexes:
 *
 *  J.C. Dunn, A fuzzy relative of the ISODATA process and its use in detecting
 *  Compact Well-Separated Clusters, Journal of Cybernetics 3(3), 1974,
 *  pp. 32-57, doi:10.1080/01969727308546046.
 */
class DuNNOWAIndex : public NNBasedIndex
{
protected:
    const int owa_numerator;
    const int owa_denominator;
    std::vector<Py_ssize_t> order;
    std::vector<FLOAT_T> pq;  ///< for SMin and SMax - aux storage of size 3*delta

    FLOAT_T aggregate(int owa, bool same_cluster)
    {
        if (owa == OWA_MEAN) {
            FLOAT_T ret = 0.0;
            size_t count = 0;
            for (size_t i=0; i<n; ++i) {
                for (size_t j=0; j<M; ++j) {
                    if ((bool)same_cluster == (bool)(L[i] == L[ind(i, j)])) {
                        ++count;
                        ret += dist(i, j);
                    }
                }
            }
            if (count == 0) return INFTY;
            else return ret/(FLOAT_T)count;
        }
        else if (owa == OWA_MIN) {
//             FLOAT_T ret = INFTY;
//             for (size_t i=0; i<n; ++i) {
//                 for (size_t j=0; j<M; ++j) {
//                     if ((bool)same_cluster == (bool)(L[i] == L[ind(i, j)])) {
//                         if (dist(i, j) < ret)
//                             ret = dist(i, j);
//                         break; // dist(i, :) is sorted increasingly
//                     }
//                 }
//             }
//             return ret;
            for (size_t u=0; u<n*M; ++u) {
                Py_ssize_t i = order[u]/M;
                Py_ssize_t j = order[u]%M;
                if ((bool)same_cluster == (bool)(L[i] == L[ind(i, j)])) {
                    return dist(i, j);
                }
            }
            return INFTY;
        }
        else if (owa == OWA_MAX) {
//             FLOAT_T ret = -INFTY;
//             for (size_t i=0; i<n; ++i) {
//                 for (Py_ssize_t j=M-1; j>=0; --j) { /* yep, a signed type */
//                     if ((bool)same_cluster == (bool)(L[i] == L[ind(i, j)])) {
//                         if (dist(i, j) > ret)
//                             ret = dist(i, j);
//                         break; // dist(i, :) is sorted increasingly
//                     }
//                 }
//             }
//            return ret;
            for (Py_ssize_t u=n*M-1; u>=0; --u) { /* yep, a signed type */
                Py_ssize_t i = order[u]/M;
                Py_ssize_t j = order[u]%M;
                if ((bool)same_cluster == (bool)(L[i] == L[ind(i, j)])) {
                    return dist(i, j);
                }
            }
            return -INFTY;
        }
        else if (owa == OWA_CONST) {
            return 1.0;
        }
        else if (owa > OWA_SMIN_START && owa <= OWA_SMIN_LIMIT) {
            Py_ssize_t delta = owa-OWA_SMIN_START;
            Py_ssize_t pq_cur = 0;
            for (size_t u=0; u<n*M; ++u) {
                Py_ssize_t i = order[u]/M;
                Py_ssize_t j = order[u]%M;
                if ((bool)same_cluster == (bool)(L[i] == L[ind(i, j)])) {
                    pq[pq_cur++] = dist(i, j);
                    if (pq_cur == 3*delta) break;
                }
            }
            if (pq_cur == 0) return INFTY;
            FLOAT_T sum_wx = 0.0, sum_w = 0.0;
            for (Py_ssize_t u=0; u<pq_cur; ++u) {
                FLOAT_T cur_w = dnorm((FLOAT_T)u+1, 1, delta);
                sum_w  += cur_w;
                sum_wx += cur_w*pq[u];
            }
            return sum_wx/sum_w;
        }
        else if (owa > OWA_SMAX_START && owa <= OWA_SMAX_LIMIT) {
            Py_ssize_t delta = owa-OWA_SMAX_START;
            Py_ssize_t pq_cur = 0;
             for (Py_ssize_t u=n*M-1; u>=0; --u) { /* yep, a signed type */
                Py_ssize_t i = order[u]/M;
                Py_ssize_t j = order[u]%M;
                if ((bool)same_cluster == (bool)(L[i] == L[ind(i, j)])) {
                    pq[pq_cur++] = dist(i, j);
                    if (pq_cur == 3*delta) break;
                }
            }
            if (pq_cur == 0) return INFTY;
            FLOAT_T sum_wx = 0.0, sum_w = 0.0;
            for (Py_ssize_t u=0; u<pq_cur; ++u) {
                FLOAT_T cur_w = dnorm((FLOAT_T)u+1, 1, delta);
                sum_w  += cur_w;
                sum_wx += cur_w*pq[u];
            }
            return sum_wx/sum_w;
        }
        else {
            GENIECLUST_ASSERT(false);
            return -1.0;  // whatever
        }
    }


public:
    // Described in the base class
    DuNNOWAIndex(
           const CMatrix<FLOAT_T>& _X,
           const size_t _K,
           const bool _allow_undo=false,
           const size_t _M=10,
           const int _owa_numerator=OWA_MIN,
           const int _owa_denominator=OWA_MAX
             )
        : NNBasedIndex(_X, _K, _allow_undo, _M),
        owa_numerator(_owa_numerator),
        owa_denominator(_owa_denominator),
        order(n*M)
    {
//         Rprintf("%d_%d_%d\n", M, owa_numerator, owa_denominator);

        int delta = 0;
        if (owa_numerator > OWA_SMIN_START && owa_numerator <= OWA_SMIN_LIMIT) {
            delta = std::max(delta, owa_numerator-OWA_SMIN_START);
        }
        else if (owa_numerator > OWA_SMAX_START && owa_numerator <= OWA_SMAX_LIMIT) {
            delta = std::max(delta, owa_numerator-OWA_SMAX_START);
        }
        if (owa_denominator > OWA_SMIN_START && owa_denominator <= OWA_SMIN_LIMIT) {
            delta = std::max(delta, owa_denominator-OWA_SMIN_START);
        }
        else if (owa_denominator > OWA_SMAX_START && owa_denominator <= OWA_SMAX_LIMIT) {
            delta = std::max(delta, owa_denominator-OWA_SMAX_START);
        }
        pq = std::vector<FLOAT_T>(3*delta);



        Cargsort(order.data(), dist.data(), n*M);
    }


    virtual FLOAT_T compute()
    {
        for (size_t i=0; i<K; ++i)
            if (count[i] <= M)
                return -INFTY;

        FLOAT_T numerator = aggregate(owa_numerator, /*same_cluster*/false);
        if (!std::isfinite(numerator)) return INFTY;

        FLOAT_T denominator = aggregate(owa_denominator, /*same_cluster*/true);
        if (!std::isfinite(denominator)) return -INFTY;

        return numerator/denominator;
    }

};



#endif
