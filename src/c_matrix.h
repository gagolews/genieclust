/*  Lightweight matrix class - KISS
 *
 *  Copyright (C) 2018-2020 Marek Gagolewski (https://www.gagolewski.com)
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *  contributors may be used to endorse or promote products derived from this
 *  software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 *  OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef __c_genie_h
#define __c_genie_h
#include <vector>


/**
 * Represents a matrix as a C-contiguous array,
 * i.e., in a row-major order.
 */
template <typename T> class matrix {
private:
    size_t n, d;
    std::vector<T> elems;

public:
    /** Initialises a new matrix of size _nrow*_ncol, filled with 0s
     *
     * @param _nrow
     * @param _ncol
     */
    matrix(size_t _nrow, size_t _ncol)
        : n(_nrow), d(_ncol), elems(_nrow*_ncol)
    {
        ;
    }

    /** Initialises a new matrix of size _nrow*_ncol, filled with _ts
     *
     * @param _nrow
     * @param _ncol
     * @param _t
     */
    matrix(size_t _nrow, size_t _ncol, T _t)
        : n(_nrow), d(_ncol), elems(_nrow*_ncol, _t)
    {
        ;
    }


    /** Initialises a new matrix of size _nrow*_ncol based on a contiguous
     * C- or Fortran-style array
     *
     * @param _data
     * @param _nrow
     * @param _ncol
     * @param _c_order whether the first _ncol elements in _data constitute the first row
     * or the first _nrow elements define the first column
     */
    template<class S> matrix(const S* _data, size_t _nrow, size_t _ncol, bool _c_order)
        : n(_nrow), d(_ncol), elems(_nrow*_ncol)
    {
        if (_c_order) {
            for (size_t i; i<_nrow*_ncol; ++i)
                elems[i] = (T)(_data[i]);
        }
        else {
            for (size_t i=0; i<_nrow; i++) {
                for (size_t j=0; j<_ncol; j++) {
                    elems[_ncol*i + j] = (T)_data[i+_nrow*j];
                }
            }
        }
    }


    /** Read/write access to an element in the i-th row and the j-th column
     *
     * @param i
     * @param j
     * @return a reference to the indicated matrix element
     */
    T& operator()(const size_t i, const size_t j) {
        return elems[d*i + j];
    }

    const T& operator()(const size_t i, const size_t j) const {
        return elems[d*i + j];
    }


    /** Returns a direct pointer to the underlying C-contiguous data array:
     * the first ncol elements give the 1st row,
     * the next ncol element give the 2nd row,
     * and so forth.
     *
     * @return pointer
     */
    T* data() {
        return elems.data();
    }

    const T* data() const {
        return elems.data();
    }


    /** Returns a direct pointer to the start of the i-th row
     *
     * @param i
     * @return pointer
     */
    T* row(const size_t i) {
        return elems.data()+i*d;
    }

    const T* row(const size_t i) const {
        return elems.data()+i*d;
    }


    /** Returns the number of rows
     *
     * @return
     */
    size_t nrow() const {
        return n;
    }


    /** Returns the number of columns
     *
     * @return
     */
    size_t ncol() const {
        return d;
    }
};

#endif
