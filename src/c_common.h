/*  Common functions, macros, includes
 *
 *  Copyleft (C) 2018-2024, Marek Gagolewski <https://www.gagolewski.com>
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


#ifndef __c_common_h
#define __c_common_h


#ifdef Py_PYTHON_H
#define GENIECLUST_PYTHON 1
#endif

#ifdef GENIECLUST_R
#undef GENIECLUST_R
#define GENIECLUST_R 1
#endif


#include <stdexcept>
#include <string>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#define OPENMP_ENABLED 1
#else
#define OPENMP_ENABLED 0
#endif

#ifndef GENIECLUST_ASSERT
#define __GENIECLUST_STR(x) #x
#define GENIECLUST_STR(x) __GENIECLUST_STR(x)

#define GENIECLUST_ASSERT(EXPR) { if (!(EXPR)) \
    throw std::runtime_error( "genieclust: Assertion " #EXPR " failed in "\
        __FILE__ ":" GENIECLUST_STR(__LINE__) ); }
#endif




#if GENIECLUST_R
#include <Rcpp.h>
#else
#include <cstdio>
#endif


#if GENIECLUST_R
#define GENIECLUST_PRINT(fmt) REprintf((fmt));
#else
#define GENIECLUST_PRINT(fmt) fprintf(stderr, (fmt));
#endif

#if GENIECLUST_R
#define GENIECLUST_PRINT_int(fmt, val) REprintf((fmt), (int)(val));
#else
#define GENIECLUST_PRINT_int(fmt, val) fprintf(stderr, (fmt), (int)(val));
#endif

#if GENIECLUST_R
#define GENIECLUST_PRINT_float(fmt, val) REprintf((fmt), (double)(val));
#else
#define GENIECLUST_PRINT_float(fmt, val) fprintf(stderr, (fmt), (double)(val));
#endif


#if GENIECLUST_R
typedef ssize_t         Py_ssize_t;
#endif



typedef double FLOAT_T; ///< float type we are working internally with

#ifndef INFTY
#define INFTY (std::numeric_limits<FLOAT_T>::infinity())
#endif

#define IS_PLUS_INFTY(x)  ((x) > 0.0 && !std::isfinite(x))
#define IS_MINUS_INFTY(x) ((x) < 0.0 && !std::isfinite(x))

#define CVI_MAX_N_PRECOMPUTE_DISTANCE 10000


#endif
