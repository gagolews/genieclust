/*  Internal cluster validity indices
 *
 *  Code originally contributed in <https://github.com/gagolews/optim_cvi>,
 *  see https://doi.org/10.1016/j.ins.2021.10.004.
 *  Copyleft (C) 2020, Maciej Bartoszuk <http://bartoszuk.rexamine.com>
 *
 *  For the 'genieclust' version:
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

#ifndef __CVI_GENERALIZED_DUNN_H
#define __CVI_GENERALIZED_DUNN_H

#include "cvi.h"
#include "cvi_generalized_dunn_delta.h"


/** Dunn's index for measuring the degree to which clusters are
 *  compact and well-separated
 *
 *  The index is defined by Eq.(3) in (Dunn, 1974).
 *
 *  J.C. Dunn, A fuzzy relative of the ISODATA process and its use in detecting
 *  Compact Well-Separated Clusters, Journal of Cybernetics 3(3), 1974,
 *  pp. 32-57, doi:10.1080/01969727308546046.
 *
 *  See the following paper for the formula and further discussion:
 *  M. Gagolewski, M. Bartoszuk, A. Cena,
 *  Are cluster validity measures (in)valid?, Information Sciences 581,
 *  620-636, 2021, DOI:10.1016/j.ins.2021.10.004
 */
class GeneralizedDunnIndex : public ClusterValidityIndex
{
protected:
    EuclideanDistance D; ///< squared Euclidean
    LowercaseDelta* numeratorDelta;
    UppercaseDelta* denominatorDelta;

public:
    // Described in the base class
    GeneralizedDunnIndex(
           const CMatrix<FLOAT_T>& _X,
           const size_t _K,
           LowercaseDeltaFactory* numeratorDeltaFactory,
           UppercaseDeltaFactory* denominatorDeltaFactory,
           const bool _allow_undo=false)
        : ClusterValidityIndex(_X, _K, _allow_undo),
          D(&X, n<=CVI_MAX_N_PRECOMPUTE_DISTANCE, true/*squared*/),
          numeratorDelta(numeratorDeltaFactory->create(D, X, L, count, K, n, d)),
          denominatorDelta(denominatorDeltaFactory->create(D, X, L, count, K, n, d))
    { }

    ~GeneralizedDunnIndex()
    {
        delete numeratorDelta;
        delete denominatorDelta;
    }

    // Described in the base class
    virtual void set_labels(const std::vector<ssize_t>& _L)
    {
        ClusterValidityIndex::set_labels(_L); // sets L, count and centroids
        numeratorDelta->recompute_all();
        denominatorDelta->recompute_all();
    }


    // Described in the base class
    virtual void modify(size_t i, ssize_t j)
    {
        numeratorDelta->before_modify(i, j);
        denominatorDelta->before_modify(i, j);
        // sets L[i]=j and updates count as well as centroids
        ClusterValidityIndex::modify(i, j);
        numeratorDelta->after_modify(i, j);
        denominatorDelta->after_modify(i, j);
    }


    // Described in the base class
    virtual void undo()
    {
        numeratorDelta->undo();
        denominatorDelta->undo();
        ClusterValidityIndex::undo();
    }


    // Described in the base class
    virtual FLOAT_T compute()
    {
        FLOAT_T max_denominator = 0.0;
        FLOAT_T min_numerator = INFTY;
        for (size_t i=0; i<K; ++i) {
            FLOAT_T denom_i = denominatorDelta->compute(i);
            if (denom_i > max_denominator)
                max_denominator = denom_i;
            for (size_t j=i+1; j<K; ++j) {
                FLOAT_T num_ij = numeratorDelta->compute(i, j);
                if (num_ij < min_numerator)
                    min_numerator = num_ij;
            }
        }

        // remember to do sqrt in deltas!
        return min_numerator/max_denominator;
    }
};


class GeneralizedDunnIndexCentroidBased : public CentroidsBasedIndex
{
protected:
    EuclideanDistance D; ///< squared Euclidean
    LowercaseDelta* numeratorDelta;
    UppercaseDelta* denominatorDelta;

public:
    // Described in the base class
    GeneralizedDunnIndexCentroidBased(
           const CMatrix<FLOAT_T>& _X,
           const size_t _K,
           LowercaseDeltaFactory* numeratorDeltaFactory,
           UppercaseDeltaFactory* denominatorDeltaFactory,
           const bool _allow_undo=false)
        : CentroidsBasedIndex(_X, _K, _allow_undo),
          D(&X, n<=CVI_MAX_N_PRECOMPUTE_DISTANCE, true/*squared*/),
          numeratorDelta(numeratorDeltaFactory->create(D, X, L, count, K, n, d, &centroids)),
          denominatorDelta(denominatorDeltaFactory->create(D, X, L, count, K, n, d, &centroids))
    { }

    ~GeneralizedDunnIndexCentroidBased()
    {
        delete numeratorDelta;
        delete denominatorDelta;
    }

    // Described in the base class
    virtual void set_labels(const std::vector<ssize_t>& _L)
    {
        CentroidsBasedIndex::set_labels(_L); // sets L, count and centroids
        numeratorDelta->recompute_all();
        denominatorDelta->recompute_all();
    }


    // Described in the base class
    virtual void modify(size_t i, ssize_t j)
    {
        numeratorDelta->before_modify(i, j);
        denominatorDelta->before_modify(i, j);
        // sets L[i]=j and updates count as well as centroids
        CentroidsBasedIndex::modify(i, j);
        numeratorDelta->after_modify(i, j);
        denominatorDelta->after_modify(i, j);
    }


    // Described in the base class
    virtual void undo()
    {
        numeratorDelta->undo();
        denominatorDelta->undo();
        CentroidsBasedIndex::undo();
    }


    // Described in the base class
    virtual FLOAT_T compute()
    {
        FLOAT_T max_denominator = 0.0;
        FLOAT_T min_numerator = INFTY;
        for (size_t i=0; i<K; ++i) {
            FLOAT_T denom_i = denominatorDelta->compute(i);
            if (denom_i > max_denominator)
                max_denominator = denom_i;
            for (size_t j=i+1; j<K; ++j) {
                FLOAT_T num_ij = numeratorDelta->compute(i, j);
                if (num_ij < min_numerator)
                    min_numerator = num_ij;
            }
        }

        // remember to do sqrt in deltas!
        return min_numerator/max_denominator;
    }
};



#endif
