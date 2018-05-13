"""
Various plotting functions

Copyright (C) 2018 Marek.Gagolewski.com


This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
    for i in np.unique(labels)+1: # -1 is black, 0 is red, etc.
        plt.scatter(X[labels==i,0], X[labels==i,1],
            c=col[i % len(col)], marker=mrk[i % len(mrk)], **kwargs)


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
