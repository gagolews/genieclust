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
    Draws a scatter plot

    `labels == i` is always drawn in the same way, no matter the `max(labels)`.

    Arguments:
    * X - a two-column matrix giving the X and Y coordinates of the points
    * labels - vector of non-negative integer labels corresponding
       to each point in X
    """
    col = ["k", "r", "g", "b", "c", "m", "y"]+list(plt.cm.get_cmap("tab10").colors)
    mrk = ["o", "v", "^", "s", "P", "*", "<", ">"]
    for i in np.unique(labels):
        plt.scatter(X[labels==i,0], X[labels==i,1],
            c=col[i % len(col)], marker=mrk[i % len(mrk)], **kwargs)


def plot_segments(X, pairs, **kwargs):
    """
    Draws a set of disjoint line segments given by
    (X[pairs[i,0],0], X[pairs[i,0],1])--(X[pairs[i,1],0], X[pairs[i,1],1]),
    i = 0, ...., pairs.shape[0]-1.

    Calls `matplotlib.pyplot.plot()` once => it's fast.

    Arguments:
    * X - a two-column matrix giving the X and Y coordinates of the points
    * pairs - a two-column integer matrix
    """
    xcoords = np.insert(X[pairs.ravel(),0].reshape(-1,2), 2, None, 1).ravel()
    ycoords = np.insert(X[pairs.ravel(),1].reshape(-1,2), 2, None, 1).ravel()
    plt.plot(xcoords, ycoords, "k-", **kwargs)
