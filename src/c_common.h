/*  Common functions, macros, includes
 *
 *  Copyleft (C) 2018-2025, Marek Gagolewski <https://www.gagolewski.com>
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
#include <cmath>


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
#define GENIECLUST_PRINT(...) REprintf(__VA_ARGS__);
#else
#define GENIECLUST_PRINT(...) fprintf(stderr, __VA_ARGS__);
#endif



#ifdef GENIECLUST_PROFILER
#include <chrono>

#define GENIECLUST_PROFILER_START \
    _genieclust_profiler_t0 = std::chrono::high_resolution_clock::now();

#define GENIECLUST_PROFILER_GETDIFF  \
    _genieclust_profiler_td = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-_genieclust_profiler_t0);

#define GENIECLUST_PROFILER_USE \
    auto GENIECLUST_PROFILER_START \
    auto GENIECLUST_PROFILER_GETDIFF \
    char _genieclust_profiler_strbuf[256];

#define GENIECLUST_PROFILER_STOP(...) \
    GENIECLUST_PROFILER_GETDIFF; \
    snprintf(_genieclust_profiler_strbuf, sizeof(_genieclust_profiler_strbuf), __VA_ARGS__); \
    GENIECLUST_PRINT("%-64s: time=%12.3lf s\n", _genieclust_profiler_strbuf, _genieclust_profiler_td.count()/1000.0);

/* use like:
GENIECLUST_PROFILER_USE
GENIECLUST_PROFILER_START
GENIECLUST_PROFILER_STOP("message %d", 7)
*/
#else
#define GENIECLUST_PROFILER_START ; /* no-op */
#define GENIECLUST_PROFILER_STOP(...) ; /* no-op */
#define GENIECLUST_PROFILER_GETDIFF ; /* no-op */
#define GENIECLUST_PROFILER_USE ; /* no-op */
#endif


#if GENIECLUST_R
typedef ssize_t         Py_ssize_t;
#endif



typedef double FLOAT_T; ///< float type we are working internally with

// #ifndef INFTY
// #define INFTY (std::numeric_limits<FLOAT_T>::infinity())
// #endif

template<class T>
inline T square(T x) { return x*x; }

template <class T>
inline T min3(const T a, const T b, const T c)
{
    T m = a;
    if (b < m) m = b;
    if (c < m) m = c;
    return m;
}

template <class T>
inline T med3(const T a, const T b, const T c)
{
    if ((b < a)^(c < a)) return a;      // b < a  && a <= c= || c < a && a <= b
    else if ((b < c)^(b < a)) return b; // c <= b && b < a   || c > b && b >= a
    else return c;
}

template <class T>
inline T max3(const T a, const T b, const T c)
{
    T m = a;
    if (b > m) m = b;
    if (c > m) m = c;
    return m;
}


#define IS_PLUS_INFINITY(x)  ((x) > 0.0 && !std::isfinite(x))
#define IS_MINUS_INFINITY(x) ((x) < 0.0 && !std::isfinite(x))



#ifdef OPENMP_DISABLED
    #define OPENMP_IS_ENABLED 0
    #ifdef _OPENMP
        #undef _OPENMP
    #endif
#else
    #ifdef _OPENMP
        #include <omp.h>
        #define OPENMP_IS_ENABLED 1
    #else
        #define OPENMP_IS_ENABLED 0
    #endif
#endif


inline void Comp_set_num_threads(Py_ssize_t n_threads)
{
#if OPENMP_IS_ENABLED
    if (n_threads <= 0)
        n_threads = omp_get_max_threads();
    omp_set_num_threads(n_threads);
#endif
}

inline int Comp_get_max_threads()
{
#if OPENMP_IS_ENABLED
    return omp_get_max_threads();
#else
    return 1;
#endif
}


#endif
