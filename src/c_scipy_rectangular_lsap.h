/*
This file is adapted from
scipy/scipy/optimize/rectangular_lsap/rectangular_lsap.cpp
(version last updated on 25 May 2022; 6d310c8)
See https://github.com/scipy/scipy/ and https://scipy.org/scipylib/.

This code implements the shortest augmenting path algorithm for the
rectangular assignment problem.  This implementation is based on the
pseudocode described in pages 1685-1686 of:

    Crouse D.F., On implementing 2D rectangular assignment algorithms,
    *IEEE Transactions on Aerospace and Electronic Systems* **52**(4), 2016,
    pp. 1679-1696, doi:10.1109/TAES.2016.140952.

Original author: Peter M. Larsen (https://github.com/pmla/).
Thanks!!!

Keywords: the Hungarian algorithm, Kuhn-Munkres algorithm,
a modified Jonker-Volgenant algorithm.



Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef __c_scipy_rectangular_lsap_h
#define __c_scipy_rectangular_lsap_h

#include "c_common.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <numeric>

#define RECTANGULAR_LSAP_INFEASIBLE -1
#define RECTANGULAR_LSAP_INVALID -2


template <typename T> std::vector<Py_ssize_t> argsort_iter(const std::vector<T> &v)
{
    std::vector<Py_ssize_t> index(v.size());
    std::iota(index.begin(), index.end(), 0);
    std::sort(index.begin(), index.end(), [&v](Py_ssize_t i, Py_ssize_t j)
              {return v[i] < v[j];});
    return index;
}


Py_ssize_t augmenting_path(Py_ssize_t nc, double* cost,
    std::vector<double>& u, std::vector<double>& v,
    std::vector<Py_ssize_t>& path, std::vector<Py_ssize_t>& row4col,
    std::vector<double>& shortestPathCosts, Py_ssize_t i,
    std::vector<bool>& SR, std::vector<bool>& SC,
    std::vector<Py_ssize_t>& remaining, double* p_minVal
) {
    double minVal = 0;

    // Crouse's pseudocode uses set complements to keep track of remaining
    // nodes.  Here we use a vector, as it is more efficient in C++.
    Py_ssize_t num_remaining = nc;
    for (Py_ssize_t it = 0; it < nc; it++) {
        // Filling this up in reverse order ensures that the solution of a
        // constant cost matrix is the identity matrix (c.f. #11602).
        remaining[it] = nc - it - 1;
    }

    std::fill(SR.begin(), SR.end(), false);
    std::fill(SC.begin(), SC.end(), false);
    std::fill(shortestPathCosts.begin(), shortestPathCosts.end(), INFINITY);

    // find shortest augmenting path
    Py_ssize_t sink = -1;
    while (sink == -1) {

        Py_ssize_t index = -1;
        double lowest = INFINITY;
        SR[i] = true;

        for (Py_ssize_t it = 0; it < num_remaining; it++) {
            Py_ssize_t j = remaining[it];

            double r = minVal + cost[i * nc + j] - u[i] - v[j];
            if (r < shortestPathCosts[j]) {
                path[j] = i;
                shortestPathCosts[j] = r;
            }

            // When multiple nodes have the minimum cost, we select one which
            // gives us a new sink node. This is particularly important for
            // integer cost matrices with small co-efficients.
            if (shortestPathCosts[j] < lowest ||
                (shortestPathCosts[j] == lowest && row4col[j] == -1)) {
                lowest = shortestPathCosts[j];
                index = it;
            }
        }

        minVal = lowest;
        if (minVal == INFINITY) { // infeasible cost matrix
            return -1;
        }

        Py_ssize_t j = remaining[index];
        if (row4col[j] == -1) {
            sink = j;
        } else {
            i = row4col[j];
        }

        SC[j] = true;
        remaining[index] = remaining[--num_remaining];
    }

    *p_minVal = minVal;
    return sink;
}


/**
 *  Solves the 2D rectangular assignment problem
 *  using the algorithm described in <doi:10.1109/TAES.2016.140952>
 *
 *  The procedure is adapted from
 *  scipy/scipy/optimize/rectangular_lsap/rectangular_lsap.cpp
 *  (version last updated  on 5 Mar; c050fd9)
 *  See https://github.com/scipy/scipy/ and https://scipy.org/scipylib/.
 *  Author: P.M. Larsen.
 *
 *
 *
 *  References
 *  ==========
 *
 *  Crouse, D.F., On implementing 2D rectangular assignment algorithms,
 *  *IEEE Transactions on Aerospace and Electronic Systems* **52**(4), 2016,
 *  pp. 1679-1696, doi:10.1109/TAES.2016.140952.
 *
 *
 *  @param C c_contiguous cost matrix; shape nr*nc
 *  @param nr number of rows in C
 *  @param nc number of cols in C
 *  @param a [output] c_contiguous vector of length min(nr,nc);
 *  @param b [output] c_contiguous vector of length min(nr,nc);
 *      (a[i], b[i]) gives location of the items in C with the smallest sum.
 *  @param minimise false if we seek the maximum
 *
 *  @return 0 on success
 */
template<class T1, class T2> Py_ssize_t linear_sum_assignment(
    T1* C,
    Py_ssize_t nr,
    Py_ssize_t nc,
    /*out*/T2* a,
    /*out*/T2* b,
    bool minimise=false
) {
    // handle trivial inputs
    if (nr == 0 || nc == 0) {
        return 0;
    }

    // build the non-negative cost matrix
    std::vector<double> cost(nr * nc);
    bool transpose = nc < nr;
    if (!transpose) {
        if (minimise) {
            double minval = *std::min_element(C, C + nr * nc);
            for (Py_ssize_t i = 0; i < nr * nc; i++) {
                cost[i] = C[i] - minval;
            }
        }
        else {
            double maxval = *std::max_element(C, C + nr * nc);
            for (Py_ssize_t i = 0; i < nr * nc; i++) {
                cost[i] = maxval-C[i];
            }
        }
    }
    else {
        if (minimise) {
            double minval = *std::min_element(C, C + nr * nc);
            for (Py_ssize_t i = 0; i < nr; i++)
                for (Py_ssize_t j = 0; j < nc; j++)
                    cost[j*nr+i] = C[i*nc+j] - minval;
        }
        else {
            double maxval = *std::max_element(C, C + nr * nc);
            for (Py_ssize_t i = 0; i < nr; i++)
                for (Py_ssize_t j = 0; j < nc; j++)
                    cost[j*nr+i] = maxval - C[i*nc+j];
        }
        std::swap(nr, nc);
    }


    // // make a copy of the cost matrix if we need to modify it
    // std::vector<double> temp;
    // if (transpose /*|| !minimise*/) {
    //     temp.resize(nr * nc);
    //
    //     // if (transpose) {
    //         for (intptr_t i = 0; i < nr; i++) {
    //             for (intptr_t j = 0; j < nc; j++) {
    //                 temp[j * nr + i] = cost[i * nc + j];
    //             }
    //         }
    //
    //         std::swap(nr, nc);
    //     // }
    //     // else {
    //         // std::copy(cost, cost + nr * nc, temp.begin());
    //     // }
    //
    //     // // negate cost matrix for maximization
    //     // if (!minimise) {
    //     //     for (intptr_t i = 0; i < nr * nc; i++) {
    //     //         temp[i] = -temp[i];
    //     //     }
    //     // }
    //
    //     cost = temp;//.data();
    // }


    // initialize variables
    std::vector<double> u(nr, 0);
    std::vector<double> v(nc, 0);
    std::vector<double> shortestPathCosts(nc);
    std::vector<Py_ssize_t> path(nc, -1);
    std::vector<Py_ssize_t> col4row(nr, -1);
    std::vector<Py_ssize_t> row4col(nc, -1);
    std::vector<bool> SR(nr);
    std::vector<bool> SC(nc);
    std::vector<Py_ssize_t> remaining(nc);

    // iteratively build the solution
    for (Py_ssize_t curRow = 0; curRow < nr; curRow++) {

        double minVal;
        Py_ssize_t sink = augmenting_path(
            nc, cost.data(), u, v, path, row4col,
            shortestPathCosts, curRow, SR, SC,
            remaining, &minVal
        );
        if (sink < 0) {
            return RECTANGULAR_LSAP_INFEASIBLE;
        }

        // update dual variables
        u[curRow] += minVal;
        for (Py_ssize_t i = 0; i < nr; i++) {
            if (SR[i] && i != curRow) {
                u[i] += minVal - shortestPathCosts[col4row[i]];
            }
        }

        for (Py_ssize_t j = 0; j < nc; j++) {
            if (SC[j]) {
                v[j] -= minVal - shortestPathCosts[j];
            }
        }

        // augment previous solution
        Py_ssize_t j = sink;
        while (1) {
            Py_ssize_t i = path[j];
            row4col[j] = i;
            std::swap(col4row[i], j);
            if (i == curRow) {
                break;
            }
        }
    }

    if (transpose) {  // nc < nr;
        Py_ssize_t i = 0;
        for (auto v: argsort_iter(col4row)) {
            a[i] = (T2)col4row[v];
            b[i] = (T2)v;
            i++;
        }
    }
    else {  // nc >= nr;
        for (Py_ssize_t i = 0; i < nr; i++) {
            a[i] = (T2)i;
            b[i] = (T2)col4row[i];
        }
    }

    return 0;
}


#endif
