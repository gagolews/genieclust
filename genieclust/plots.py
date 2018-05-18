"""
Various plotting functions

Copyright (C) 2018 Marek.Gagolewski.com
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(X, labels, **kwargs):
    """
    Draws a scatter plot.

    Unlike `matplitlib.pyplot.scatter()`
    `labels == i` is always drawn in the same way, no matter the `max(labels)`.


    Parameters:
    ----------

    X : ndarray, shape (n, 2)
        A two-column matrix giving the X and Y coordinates of the points.

    labels : ndarray, shape (n,)
        A vector of integer labels corresponding to each point in X,
        giving its plot style.

    **kwargs : Collection properties
        Further arguments to `matplotlib.pyplot.scatter()`.
    """
    col = ["k", "r", "g", "b", "c", "m", "y"]+list(plt.cm.get_cmap("tab10").colors)
    mrk = ["o", "v", "^", "s", "P", "*", "<", ">"]
    for i in np.unique(labels): # -1 is black, 0 is red, etc.
        plt.scatter(X[labels==i,0], X[labels==i,1],
            c=col[(i+1) % len(col)], marker=mrk[(i+1) % len(mrk)], **kwargs)


def plot_segments(X, pairs, **kwargs):
    """
    Draws a set of disjoint line segments given by
    (X[pairs[i,0],0], X[pairs[i,0],1])--(X[pairs[i,1],0], X[pairs[i,1],1]),
    i = 0, ...., pairs.shape[0]-1.

    Calls `matplotlib.pyplot.plot()` once => it's fast.


    Parameters:
    ----------

    X : ndarray, shape (n, 2)
        A two-column matrix giving the X and Y coordinates of the points.

    pairs : ndarray, shape (m, 2)
        A two-column matrix, giving the pairs of indexes
        defining the line segments.

    **kwargs : Collection properties
        Further arguments to `matplotlib.pyplot.plot()`.
    """
    xcoords = np.insert(X[pairs.ravel(),0].reshape(-1,2), 2, None, 1).ravel()
    ycoords = np.insert(X[pairs.ravel(),1].reshape(-1,2), 2, None, 1).ravel()
    plt.plot(xcoords, ycoords, "k-", **kwargs)
