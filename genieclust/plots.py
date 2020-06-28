# Various plotting functions
#
# Copyright (C) 2018-2020 Marek Gagolewski (https://www.gagolewski.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License
# Version 3, 19 November 2007, published by the Free Software Foundation.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License Version 3 for more details.
# You should have received a copy of the License along with this program.
# If not, see <https://www.gnu.org/licenses/>.


import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np


# module globals:
col = ["k", "r", "g", "b", "c", "m", "y"]+\
    [matplotlib.colors.to_hex(c) for c in plt.cm.get_cmap("tab10").colors]+\
    [matplotlib.colors.to_hex(c) for c in plt.cm.get_cmap("tab20").colors]+\
    [matplotlib.colors.to_hex(c) for c in plt.cm.get_cmap("tab20b").colors]+\
    [matplotlib.colors.to_hex(c) for c in plt.cm.get_cmap("tab20c").colors]

mrk = ["o", "^", "+", "x", "D", "v", "s", "*", "<", ">", "2"]


def plot_scatter(X, y=None, labels=None, **kwargs):
    """
    Draws a scatter plot.

    Unlike in `matplitlib.pyplot.scatter()`, all points in `X`
    corresponding to `labels == i` are always drawn in the same way,
    no matter the `max(labels)`.


    Parameters:
    ----------

    X : ndarray, shape (n, 2) or ndarray, shape (n,)
        A two-column matrix giving the x and y coordinates of the points.
        Optionally, these can be given by both X and y.

    y : None or ndarray, shape (n,)
        y coordinates in the case of X being a vector

    labels : ndarray, shape (n,) or None
        A vector of integer labels corresponding to each point in `X`,
        giving its plot style.

    **kwargs : Collection properties
        Further arguments to `matplotlib.pyplot.scatter()`.
    """
    if labels is None: labels = np.repeat(0, X.shape[0])
    if X.ndim == 2:
        if not X.shape[1] == 2:
            raise ValueError("X must have 2 columns or y must be given")
    elif X.ndim == 1:
        if y is None or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same shape")
        X = np.column_stack((X, y))
    else:
        raise ValueError("invalid X")

    if not X.shape[0] == labels.shape[0]:
        raise ValueError("incorrect number of labels")
    for i in np.unique(labels): # 0 is black, 1 is red, etc.
        plt.scatter(X[labels==i,0], X[labels==i,1],
            c=col[(i) % len(col)], marker=mrk[(i) % len(mrk)], **kwargs)



def plot_segments(X, pairs, style="k-", **kwargs):
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
        A two-column matrix, giving the pairs of indices
        defining the line segments.

    style: see `matplotlib.pyplot.plot()`

    **kwargs : Collection properties
        Further arguments to `matplotlib.pyplot.plot()`.
    """
    if not X.shape[1] == 2: raise ValueError("X must have 2 columns")
    if not pairs.shape[1] == 2: raise ValueError("pairs must have 2 columns")

    xcoords = np.insert(X[pairs.ravel(),0].reshape(-1,2), 2, None, 1).ravel()
    ycoords = np.insert(X[pairs.ravel(),1].reshape(-1,2), 2, None, 1).ravel()
    plt.plot(xcoords, ycoords, style, **kwargs)
