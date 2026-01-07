/*  An implementation of the *Kneedle* algorithm to detect knee/elbow points,
 *  with exponential moving average smoothing
 *
 *  Based on V. Satopaa, J. Albrecht, D. Irwin, B. Raghavan
 *  Finding a “Kneedle” in a haystack: Detecting knee points in system behavior,
 *  31st Intl. Conf. Distributed Computing Systems Workshops, 2011, pp. 166-171,
 *  DOI: 10.1109/ICDCSW.2011.20
 *
 *
 *  Copyleft (C) 2018-2026, Marek Gagolewski <https://www.gagolewski.com>
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


#ifndef __c_kneedle_h
#define __c_kneedle_h

#include <cmath>
#include "c_common.h"


/**
 * Exponential moving average with
 * smoothing parameter alpha = 1-exp(-dt)
 *
 * y[0] = x[0],
 * y[i] = alpha*x[i]+(1-alpha)*y[i-1]
 *
 * @param x [in] input array of length n
 * @param y [out] output array of length n
 * @param n length of x and y
 * @param dt controls the smoothing parameter alpha = 1-exp(-dt)
 */
template <class FLOAT>
void Cema(const FLOAT* x, Py_ssize_t n, FLOAT dt, FLOAT* y)
{
    FLOAT alpha = -std::expm1(-dt);  // 1-np.exp(-dt)
    FLOAT alpham1 = 1.0-alpha;

    y[0] = x[0];
    for (Py_ssize_t i=1; i<n; ++i)
        y[i] = alpha*x[i] + alpham1*y[i-1];
}


/**
 * Find the most significant knee/elbow using the Kneedle method
 * of an increasing sequence with exponential moving average smoothing
 *
 * Based on V. Satopaa, J. Albrecht, D. Irwin, B. Raghavan
 * Finding a “Kneedle” in a haystack: Detecting knee points in system behavior,
 * 31st Intl. Conf. Distributed Computing Systems Workshops, 2011, pp. 166-171,
 * DOI: 10.1109/ICDCSW.2011.20
 *
 *
 * @param x [in] input array of length n
 * @param n length of x and y
 * @param convex whether the data in x are convex-ish (elbow detection) or not (knee)
 * @param dt controls the smoothing parameter alpha = 1-exp(-dt)
 *
 * @return the location of the knee/elbow, 0 if not found
 */
template <class FLOAT>
Py_ssize_t Ckneedle_increasing(const FLOAT* x, Py_ssize_t n, bool convex, FLOAT dt)
{
    std::vector<FLOAT> _y(n);
    FLOAT* y = _y.data();

    Cema(x, n, dt, y);  // sets y

    // normalise to [0,1], subtract i/(n-1)
    FLOAT miny = y[0], maxy = y[0];
    for (Py_ssize_t i=1; i<n; ++i) {
        if (miny > y[i]) miny = y[i];
        else if (maxy < y[i]) maxy = y[i];
    }
    FLOAT rngy = maxy-miny;
    for (Py_ssize_t i=0; i<n; ++i)
        y[i] = (y[i]-miny)/rngy - (FLOAT)i/(FLOAT)(n-1);


    Py_ssize_t peak_i = 0;
    FLOAT peak_y = -INFINITY;

    if (convex) {
        for (Py_ssize_t i=1; i<n-1; ++i) {
            if (y[i-1] > y[i] and y[i] < y[i+1]) {
                if (y[i] >= peak_y) {
                    peak_y = y[i];
                    peak_i = i;
                }
            }
        }
    }
    else {
        for (Py_ssize_t i=1; i<n-1; ++i) {
            if (y[i-1] < y[i] and y[i] > y[i+1]) {
                if (y[i] >= peak_y) {
                    peak_y = y[i];
                    peak_i = i;
                }
            }
        }
    }

    return peak_i;
}

#endif
