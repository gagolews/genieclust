"""
Various plotting functions
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2020-2021, Marek Gagolewski <https://www.gagolewski.com>      #
#                                                                              #
#                                                                              #
#   This program is free software: you can redistribute it and/or modify       #
#   it under the terms of the GNU Affero General Public License                #
#   Version 3, 19 November 2007, published by the Free Software Foundation.    #
#   This program is distributed in the hope that it will be useful,            #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the               #
#   GNU Affero General Public License Version 3 for more details.              #
#   You should have received a copy of the License along with this program.    #
#   If this is not the case, refer to <https://www.gnu.org/licenses/>.         #
#                                                                              #
# ############################################################################ #


import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np


# module globals:
col = ["k", "r", "g", "b", "c", "m", "y"]                                   + \
    [matplotlib.colors.to_hex(c) for c in plt.cm.get_cmap("tab10").colors]  + \
    [matplotlib.colors.to_hex(c) for c in plt.cm.get_cmap("tab20").colors]  + \
    [matplotlib.colors.to_hex(c) for c in plt.cm.get_cmap("tab20b").colors] + \
    [matplotlib.colors.to_hex(c) for c in plt.cm.get_cmap("tab20c").colors]

mrk = ["o", "^", "+", "x", "D", "v", "s", "*", "<", ">", "2"]



def _get_xy(X, y):
    # auxiliary function
    X = np.array(X)
    if X.ndim == 2:
        if not X.shape[1] == 2:
            raise ValueError("`X` must have 2 columns.")
        if y is not None:
            raise ValueError("If `X` is a matrix, `y` should not be given.")
    elif X.ndim == 1:
        if y is None:
            raise ValueError("If `X` is a vector, `y` should be provided.")

        y = np.array(y)
        if y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("`X` and `y` must have the same shape.")
        X = np.column_stack((X, y))
    else:
        raise ValueError("Invalid `X`.")

    return X


def plot_scatter(X, y=None, labels=None, **kwargs):
    """
    Draws a scatter plot


    Parameters
    ----------

    X : array_like
        Either a two-column matrix that gives the *x* and *y* coordinates of the points
        or a vector of length *n*. In the latter case, `y` must be a
        vector of length *n* as well.

    y : None or array_like
        The *y* coordinates of the *n* points in the case where `X` is a vector.

    labels : None or array_like
        A vector of *n* integer labels that correspond to each point in `X`,
        that gives its plot style.

    **kwargs : Collection properties
        Further arguments to `matplotlib.pyplot.scatter`.


    Notes
    -----

    If `X` is a two-column matrix, then ``plot_scatter(X)``
    is equivalent to ``plot_scatter(X[:,0], X[:,1])``.

    Unlike in `matplotlib.pyplot.scatter`,
    for any fixed ``j``, all points ``X[i,:]`` such that ``labels[i] == j``
    are always drawn in the same way, no matter the ``max(labels)``.
    In particular, labels 0, 1, 2, and 3 correspond to
    black, red, green, and blue, respectively.

    This function was inspired by the ``plot()`` function
    from the R package ``graphics``.


    Examples
    --------

    .. plot::

        An example scatter plots where each point is assigned one of
        two distinct labels:

        >>> n = np.r_[100, 50]
        >>> X = np.r_[np.random.randn(n[0], 2), np.random.randn(n[1], 2)+2.0]
        >>> l = np.repeat([0, 1], n)
        >>> genieclust.plots.plot_scatter(X, labels=l)
        >>> plt.show()                                   # doctest: +SKIP


    .. plot::

        Here are the first 10 plotting styles:

        >>> ncol = len(genieclust.plots.col)
        >>> nmrk = len(genieclust.plots.mrk)
        >>> mrk_recycled = np.tile(
        ...     genieclust.plots.mrk,
        ...     int(np.ceil(ncol/nmrk)))[:ncol]
        >>> for i in range(10):                               # doctest: +SKIP
        ...     plt.text(                                     # doctest: +SKIP
        ...         i, 0, i, horizontalalignment="center")    # doctest: +SKIP
        ...     plt.plot(                                     # doctest: +SKIP
        ...         i, 1, marker=mrk_recycled[i],             # doctest: +SKIP
        ...         color=genieclust.plots.col[i],            # doctest: +SKIP
        ...         markersize=25)                            # doctest: +SKIP
        >>> plt.title("Plotting styles for labels=0,1,...,9") # doctest: +SKIP
        >>> plt.ylim(-3,4)                                    # doctest: +SKIP
        >>> plt.axis("off")                                   # doctest: +SKIP
        >>> plt.show()                                        # doctest: +SKIP

    """
    X = _get_xy(X, y)

    if labels is None:
        labels = np.repeat(0, X.shape[0])
    else:
        labels = np.array(labels)
    if labels.ndim != 1 or X.shape[0] != labels.shape[0]:
        raise ValueError("Incorrect shape of `labels`.")

    for i in np.unique(labels):  # 0 is black, 1 is red, etc.
        plt.scatter(
            X[labels == i, 0],
            X[labels == i, 1],
            c=col[(i) % len(col)],
            marker=mrk[(i) % len(mrk)],
            **kwargs)



def plot_segments(pairs, X, y=None, style="k-", **kwargs):
    """
    Draws a set of disjoint line segments


    Parameters
    ----------

    pairs : array_like
        A two-column matrix that gives the pairs of indices
        defining the line segments to draw.

    X : array_like
        Either a two-column matrix that gives the *x* and *y* coordinates of the points
        or a vector of length *n*. In the latter case, `y` must be a
        vector of length *n* as well.

    y : None or array_like
        The *y* coordinates of the *n* points in the case where `X` is a vector.

    style:
        See `matplotlib.pyplot.plot`.

    **kwargs : Collection properties
        Further arguments to `matplotlib.pyplot.plot`.


    Notes
    -----

    The function draws a set of disjoint line segments from
    ``(X[pairs[i,0],0], X[pairs[i,0],1])`` to
    ``(X[pairs[i,1],0], X[pairs[i,1],1])``
    for all ``i`` from ``0`` to ``pairs.shape[0]-1``.

    `matplotlib.pyplot.plot` is called only once.
    Therefore, you can expect it to be pretty pretty fast.


    Examples
    --------

    .. plot::

        Plotting the convex hull of a point set:

        >>> import scipy.spatial
        >>> X = np.random.randn(100, 2)
        >>> hull = scipy.spatial.ConvexHull(X)
        >>> genieclust.plots.plot_scatter(X)
        >>> genieclust.plots.plot_segments(hull.simplices, X, style="r--")
        >>> plt.show()                                 # doctest: +SKIP


    .. plot::

        Plotting the minimum spanning tree:

        >>> X = np.random.randn(100, 2)
        >>> mst = genieclust.internal.mst_from_distance(X, "euclidean")
        >>> genieclust.plots.plot_scatter(X)
        >>> genieclust.plots.plot_segments(mst[1], X, style="m-.")
        >>> plt.axis("equal")                          # doctest: +SKIP
        >>> plt.show()                                 # doctest: +SKIP



    """
    X = _get_xy(X, y)

    pairs = np.array(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("`pairs` must be a matrix with 2 columns.")

    xcoords = np.insert(X[pairs.ravel(), 0].reshape(-1, 2), 2, None, 1).ravel()
    ycoords = np.insert(X[pairs.ravel(), 1].reshape(-1, 2), 2, None, 1).ravel()
    plt.plot(xcoords, ycoords, style, **kwargs)
