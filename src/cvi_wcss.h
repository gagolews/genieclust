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

#ifndef __CVI_WCSS_H
#define __CVI_WCSS_H

#include "cvi.h"



/** Negated Within-Cluster Sum of Squares and the Ball-Hall Index
 *
 *  The Ball-Hall index is weighted by the cluster cardinality (weighted=true).
 *
 *  WCSS is the objective function used, amongst others, in the k-means and
 *  the Ward and Calinski&Harabasz algorithms.
 *
 *  G.H. Ball, D.J. Hall,
 *  ISODATA: A novel method of data analysis and pattern classification,
 *  Technical report No. AD699616, Stanford Research Institute, 1965.
 *
 *  T. Calinski, J. Harabasz, A dendrite method for cluster analysis,
 *  Communications in Statistics, 3(1), 1974, pp. 1-27,
 *  doi:10.1080/03610927408827101.
 *
 *
 *  See the following paper for the formula and further discussion:
 *  M. Gagolewski, M. Bartoszuk, A. Cena,
 *  Are cluster validity measures (in)valid?, Information Sciences 581,
 *  620-636, 2021, DOI:10.1016/j.ins.2021.10.004
 */
class WCSSIndex : public CentroidsBasedIndex
{
protected:
    bool weighted;          ///< false for WCSS, true for the Ball-Hall index

public:
    // Described in the base class
    WCSSIndex(
           const CMatrix<FLOAT_T>& _X,
           const size_t _K,
           const bool _allow_undo=false,
           bool _weighted=false)
        : CentroidsBasedIndex(_X, _K, _allow_undo)
    {
        weighted = _weighted;
    }



    // Described in the base class
    virtual FLOAT_T compute()
    {
        // sum of within-cluster squared L2 distances
        FLOAT_T wcss = 0.0;
        for (size_t i=0; i<n; ++i) {
            for (size_t j=0; j<d; ++j) {
                wcss += square(centroids(L[i],j)-X(i,j))/((weighted)?count[L[i]]:1.0);
            }
        }
        return -wcss;  // negative!!!
    }

};



#endif
