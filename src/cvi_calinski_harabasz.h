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

#ifndef __CVI_CALINSKI_HARABASZ_H
#define __CVI_CALINSKI_HARABASZ_H

#include "cvi.h"



/** The Calinski-Harabasz Index (Variance Ratio Criterion)
 *
 *  See Eq.(3) in (Calinski, Harabasz, 1974)
 *
 *  T. Calinski, J. Harabasz, A dendrite method for cluster analysis,
 *  Communications in Statistics, 3(1), 1974, pp. 1-27,
 *  doi:10.1080/03610927408827101.
 *
 *  See the following paper for the formula and further discussion:
 *  M. Gagolewski, M. Bartoszuk, A. Cena,
 *  Are cluster validity measures (in)valid?, Information Sciences 581,
 *  620-636, 2021, DOI:10.1016/j.ins.2021.10.004
 */
class CalinskiHarabaszIndex : public CentroidsBasedIndex
{
protected:
    std::vector<FLOAT_T> centroid; ///< the centroid of the whole X, size d
    FLOAT_T numerator;             ///< sum of intra-cluster squared L2 distances
    FLOAT_T denominator;           ///< sum of within-cluster squared L2 distances

    FLOAT_T last_numerator;        ///< for undo()
    FLOAT_T last_denominator;      ///< for undo()

public:
    // Described in the base class
    CalinskiHarabaszIndex(
           const matrix<FLOAT_T>& _X,
           const uint8_t _K,
           const bool _allow_undo=false)
        : CentroidsBasedIndex(_X, _K, _allow_undo),
          centroid(d, 0.0)
    {
        // centroid[i,j] == 0.0 already

        // compute the centroid of the whole dataset
        for (size_t i=0; i<n; ++i) {
            for (size_t j=0; j<d; ++j) {
                centroid[j] += X(i, j);
            }
        }
        for (size_t j=0; j<d; ++j) {
            centroid[j] /= (FLOAT_T)n;
        }
    }


    // Described in the base class
    virtual void set_labels(const std::vector<uint8_t>& _L)
    {
        CentroidsBasedIndex::set_labels(_L); // sets L, count and centroids

        // sum of intra-cluster squared L2 distances
        numerator = 0.0;
        for (size_t i=0; i<K; ++i) {
            for (size_t j=0; j<d; ++j) {
                numerator += square(centroid[j]-centroids(i,j))*count[i];
            }
        }

        // sum of within-cluster squared L2 distances
        denominator = 0.0;
        for (size_t i=0; i<n; ++i) {
            for (size_t j=0; j<d; ++j) {
                denominator += square(centroids(L[i],j)-X(i,j));
            }
        }
    }


    // Described in the base class
    virtual void modify(size_t i, uint8_t j)
    {
        uint8_t tmp = L[i];
        // tmp = old label for the i-th point
        // j   = new label for the i-th point

        if (allow_undo) {
            last_numerator = numerator;
            last_denominator = denominator;
        }

        for (size_t k=0; k<d; ++k) {
            numerator -= square(centroid[k]-centroids(j,k))*count[j];
            numerator -= square(centroid[k]-centroids(tmp,k))*count[tmp];
        }


        // sets L[i]=j and updates count as well as centroids
        CentroidsBasedIndex::modify(i, j);


        for (size_t k=0; k<d; ++k) {
            numerator += square(centroid[k]-centroids(j,k))*count[j];
            numerator += square(centroid[k]-centroids(tmp,k))*count[tmp];
        }


        // centroids(j,:) and centroids(tmp,:) have changed
        // -- we need to update within-cluster distances for the j-th
        // and the tmp-th cluster -- we need to iterate over all members
        // of those two clusters -- for small K (which we assume here)
        // it'll be more efficient to actually compute the denominator from
        // scratch
        denominator = 0.0;
        for (size_t i=0; i<n; ++i) {
            for (size_t j=0; j<d; ++j) {
                denominator += square(centroids(L[i],j)-X(i,j));
            }
        }
    }


    // Described in the base class
    virtual FLOAT_T compute()
    {
        return numerator*FLOAT_T(n-K)/(denominator*FLOAT_T(K-1.0));
    }


    // Described in the base class
    virtual void undo()
    {
        numerator = last_numerator;
        denominator = last_denominator;
        CentroidsBasedIndex::undo();
    }

};



#endif
