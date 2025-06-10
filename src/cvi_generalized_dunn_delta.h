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

#ifndef __CVI_GENERALIZED_DUNN_DELTA_H
#define __CVI_GENERALIZED_DUNN_DELTA_H

#include "cvi.h"

class Delta
{
protected:
    EuclideanDistance& D; ///< squared Euclidean
    const CMatrix<FLOAT_T>& X;
    //CMatrix<FLOAT_T>& X;         ///< data matrix of size n*d
    std::vector<Py_ssize_t>& L;    ///< current label vector of size n
    std::vector<size_t>& count; ///< size of each of the K clusters
    size_t K;
    size_t n;
    size_t d;
    CMatrix<FLOAT_T>* centroids; ///< centroids, can be NULL

public:
    Delta(
           EuclideanDistance& D,
           const CMatrix<FLOAT_T>& X,
           std::vector<Py_ssize_t>& L,
           std::vector<size_t>& count,
           size_t K,
           size_t n,
           size_t d,
           CMatrix<FLOAT_T>* centroids=nullptr
           )
        : D(D),
          X(X),
          L(L),
          count(count),
          K(K),
          n(n),
          d(d),
          centroids(centroids)
    { }

    virtual void before_modify(size_t i, Py_ssize_t /*j*/) = 0;
    virtual void after_modify(size_t i, Py_ssize_t /*j*/) = 0;
    virtual void undo() = 0;
    virtual void recompute_all() = 0;

    virtual ~Delta() {}
};

class LowercaseDelta : public Delta
{
public:
    LowercaseDelta(
        EuclideanDistance& D,
        const CMatrix<FLOAT_T>& X,
        std::vector<Py_ssize_t>& L,
        std::vector<size_t>& count,
        size_t K,
        size_t n,
        size_t d,
        CMatrix<FLOAT_T>* centroids=nullptr
        )
    : Delta(D,X,L,count,K,n,d,centroids)
    { }

    virtual FLOAT_T compute(size_t k, size_t l) = 0;

    virtual ~LowercaseDelta() {}
};


class UppercaseDelta : public Delta
{
public:
    UppercaseDelta(
        EuclideanDistance& D,
        const CMatrix<FLOAT_T>& X,
        std::vector<Py_ssize_t>& L,
        std::vector<size_t>& count,
        size_t K,
        size_t n,
        size_t d,
        CMatrix<FLOAT_T>* centroids=nullptr
        )
    : Delta(D,X,L,count,K,n,d,centroids)
    { }

    virtual FLOAT_T compute(size_t k) = 0;

    virtual ~UppercaseDelta() {}
};

class DeltaFactory
{
public:
    virtual bool IsCentroidNeeded() = 0;

    virtual ~DeltaFactory() {}
};


class LowercaseDeltaFactory : public DeltaFactory
{
public:
    // cannot be in DeltaFactory since result type is different, even if parameter list is the same
    virtual LowercaseDelta* create(EuclideanDistance& D,
           const CMatrix<FLOAT_T>& X,
           std::vector<Py_ssize_t>& L,
           std::vector<size_t>& count,
           size_t K,
           size_t n,
           size_t d,
           CMatrix<FLOAT_T>* centroids=nullptr) = 0;


    // static LowercaseDeltaFactory* GetSpecializedFactory(std::string lowercaseDeltaName);
};

class UppercaseDeltaFactory : public DeltaFactory
{
public:
    // cannot be in DeltaFactory since result type is different, even if parameter list is the same
    virtual UppercaseDelta* create(EuclideanDistance& D,
           const CMatrix<FLOAT_T>& X,
           std::vector<Py_ssize_t>& L,
           std::vector<size_t>& count,
           size_t K,
           size_t n,
           size_t d,
           CMatrix<FLOAT_T>* centroids=nullptr) = 0;

    // static UppercaseDeltaFactory* GetSpecializedFactory(std::string uppercaseDeltaName);
};

#endif
