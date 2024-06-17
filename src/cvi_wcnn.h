/*  Internal cluster validity indices
 *
 *  Code originally contributed in <https://github.com/gagolews/optim_cvi>,
 *  see https://doi.org/10.1016/j.ins.2021.10.004.
 *
 *  Copyleft (C) 2020-2024, Marek Gagolewski <https://www.gagolewski.com>
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

#ifndef __CVI_WCNN_H
#define __CVI_WCNN_H

#include "cvi.h"



/** Within-Cluster Nearest-Neighbours
 *
 *  For given M, returns the overall proportion of
 *  each point's M nearest neighbours belonging to the same cluster.
 *
 *  If there is a cluster of size <= M, the index is -INFTY.
 *
 *  See the following paper for the formula and further discussion:
 *  M. Gagolewski, M. Bartoszuk, A. Cena,
 *  Are cluster validity measures (in)valid?, Information Sciences 581,
 *  620-636, 2021, DOI:10.1016/j.ins.2021.10.004
 */
class WCNNIndex : public NNBasedIndex
{
public:
    // Described in the base class
    WCNNIndex(
           const CMatrix<FLOAT_T>& _X,
           const size_t _K,
           const bool _allow_undo=false,
           const size_t _M=10
             )
        : NNBasedIndex(_X, _K, _allow_undo, _M)
    {
        ;
    }


    virtual FLOAT_T compute()
    {
        for (size_t i=0; i<K; ++i)
            if (count[i] <= M)
                return -INFTY;

        size_t wcnn = 0;
        for (size_t i=0; i<n; ++i) {
            for (size_t j=0; j<M; ++j) {
                if (L[i] == L[ind(i, j)])
                    wcnn++;
            }
        }
        return wcnn/(FLOAT_T)(n*M);
    }

};



#endif
