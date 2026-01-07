"""
A stupid clustering algorithm. Do not use it. Don't trip.
(Despite many attempts to come up with some useful heuristics, the simple
robust single linkage-like method is similarly fine)
March 2025
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2018-2026, Marek Gagolewski <https://www.gagolewski.com>      #
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


import numpy as np
import scipy.spatial
import genieclust
from sklearn.base import BaseEstimator, ClusterMixin
import matplotlib.pyplot as plt

SKIP = np.iinfo(int).min  # must be negative
UNSET = -1 # must be negative
NOISE = 0  # must be 0


# TODO: 0-based -> 1-based!!!



def _lumberdupa(
    X,
    n_clusters,
    noise_cluster=False,
    n_neighbors=5,
    outlier_factor=1.5,
    min_cluster_size=10,
    min_cluster_factor=0.2,
    max_twig_size=5,
    twig_size_factor=0.01,
    M=1,
    cut_internodes=True,
    mst_skiplist=None,      # default = empty
    verbose=False
):
    assert n_clusters > 0
    assert M >= 0
    assert n_neighbors >= 0

    n = X.shape[0]

    min_cluster_size = max(min_cluster_size, min_cluster_factor*n/n_clusters)

    if mst_skiplist is None:
        mst_skiplist = []

    max_twig_size = max(max_twig_size, twig_size_factor*n/n_clusters)

    if n_neighbors <= 0:
        max_twig_size = 0
        cut_internodes = False

    if max_twig_size <= 0:
        max_twig_size = 0
        noise_cluster = False

    if n_neighbors > 0 or M > 1:
        kd = scipy.spatial.KDTree(X)
        nn_w, nn_a = kd.query(X, max(n_neighbors+1, M+1))
        nn_w = np.ascontiguousarray(np.array(nn_w)[:, 1:])  # exclude self
        nn_a = np.ascontiguousarray(np.array(nn_a)[:, 1:])


    if M <= 1:
        mst_w, mst_e = genieclust.internal.mst_from_distance(X, "euclidean")
    else:
        d_core = genieclust.internal.get_d_core(nn_w, nn_a, M)

        mst_w, mst_e = genieclust.internal.mst_from_distance(
            X, metric="euclidean", d_core=d_core, verbose=verbose
        )

    _o = np.argsort(mst_w)[::-1]
    mst_w = mst_w[_o]  # weights sorted decreasingly
    mst_e = mst_e[_o, :]

    # TODO: store as two arrays: indices, indptr (compressed sparse row format)
    mst_a = [ [] for i in range(n) ]
    for i in range(n-1):
        mst_a[mst_e[i, 0]].append(i)
        mst_a[mst_e[i, 1]].append(i)
    for i in range(n):
        mst_a[i] = np.array(mst_a[i])
    mst_a = np.array(mst_a, dtype="object")



    def visit(v, e, c):  # v->w  where mst_e[e,:]={v,w}
        if mst_labels[e] == SKIP or (c != NOISE and mst_labels[e] == NOISE):
            return 0
        iv = int(mst_e[e, 1] == v)
        w = mst_e[e, 1-iv]
        tot = 1
        for e2 in mst_a[w]:
            if mst_e[e2, 0] != v and mst_e[e2, 1] != v:
                tot += visit(w, e2, c)
        mst_s[e, iv] = tot if c != NOISE else -1
        mst_s[e, 1-iv] = 0

        labels[w] = c
        mst_labels[e] = c

        return tot


    def mark_clusters():
        c = 0
        for v in range(n):
            if labels[v] == UNSET:
                c += 1
                labels[v] = c
                for e in mst_a[v]:
                    visit(v, e, c)

        # c is the number of clusters
        counts = np.bincount(labels)  # cluster counts, counts[0] == number of noise points

        # fix mst_s now that we know the cluster sizes
        for i in range(n-1):
            if mst_labels[i] > 0:
                j = int(mst_s[i, 1] == 0)
                mst_s[i, j] = counts[np.abs(mst_labels[i])]-mst_s[i, 1-j]
        min_mst_s = np.min(mst_s, axis=1)

        return c, counts, min_mst_s



    def mark_noise(v):
        if not is_outlier[v]: return

        e = -1
        for e2 in mst_a[v]:
            if mst_labels[e2] > 0:
                if e < 0:
                    e = e2  # first non-NOISE, non-SKIP edge
                else:
                    return  # we want a vertex that's incident to only one such edge

        if e == -1:
            return

        if min_mst_s[e] > max_twig_size:
            return

        labels[v] = NOISE
        mst_labels[e] = NOISE
        # print(v, e, mst_e[e,:])

        iv = int(mst_e[e, 1] == v)
        v = mst_e[e, 1-iv]
        mark_noise(v)  # tail call


    while True:
        mst_s = np.zeros((n-1, 2), dtype=int)  # sizes: mst_s[e, 0] is the size of the cluster that appears when
        labels = np.repeat(UNSET, n)
        mst_labels = np.repeat(UNSET, n-1)
        for s in mst_skiplist:
            mst_labels[s] = SKIP  # mst_skiplist
        c, counts, min_mst_s = mark_clusters()

        last_iteration = (c >= n_clusters)

        # leaves are vertices of degree 1 (in the forest with the SKIP edges removed!)
        which_leaves = mst_e[:,::-1][mst_s == 1]

        if n_neighbors == 0:
            is_outlier = np.repeat(False, n)
        else:
            # a point is an outlier if its k-th nearest neighbour is just too far away -
            # - relative to the "typical" distances to k-nearest neighbours within the point's cluster
            # (each cluster can be of different density, so we take this into account)
            # or may be considered as an outlier after a cluster breaks down to smaller ones
            outlier_k = n_neighbors-1  # which nearest neighbour do we take into account?
            Q13 = np.array([np.percentile(nn_w[labels==j+1, outlier_k], [25, 75]) for j in range(c)])
            bnd_outlier = np.r_[np.nan, (Q13[:, 1]+outlier_factor*(Q13[:, 1]-Q13[:, 0]))]
            is_outlier = (nn_w[:, outlier_k] > bnd_outlier[labels])
            # plt.clf()
            # plt.boxplot([nn_w[labels==j+1, outlier_k] for j in range(c)])
            # plt.show()
            # stop()

        which_outliers = np.flatnonzero(is_outlier)


        if max_twig_size > 0 and (not last_iteration or noise_cluster):
            # cut out all twigs ("small" branches of size <= max_twig_size)
            # that solely consist of outliers, starting from leaves
            for v in which_leaves:
                mark_noise(v)

            mst_s[:,:] = 0
            labels[labels != NOISE] = UNSET
            mst_labels[(mst_labels != SKIP) & (mst_labels != NOISE)] = UNSET

            c, counts, min_mst_s = mark_clusters()


        # the longest node that yields not too small a cluster
        is_limb = (mst_labels > 0)
        is_limb &= (min_mst_s >= min(min_cluster_size, np.max(min_mst_s[is_limb])))

        which_limbs = np.flatnonzero(is_limb)  # TODO: we just need the first non-zero here....
        cutting_e = which_limbs[0]
        cutting_w = mst_w[cutting_e]


        if verbose: print("cand=(%g,%g,[%d,%d])." % (cutting_e, cutting_w, mst_s[cutting_e,0], mst_s[cutting_e,1]))

        if not cut_internodes:
            is_internode = np.repeat(False, n-1)
            which_internodes = []
        else:
            # an internode is an edge incident to an outlier, whose removal leads to a new cluster of "considerable" size
            is_internode = (mst_labels > 0)
            is_internode &= (is_outlier[mst_e[:,0]] | is_outlier[mst_e[:,1]])
            is_internode &= (min_mst_s >= min_cluster_size)
            # is_internode &= (np.min(mst_s, axis=1)/np.max(mst_s, axis=1)) >= cluster_size_factor

            which_internodes = np.flatnonzero(is_internode)

            # OLD version: internodes have priority over limbs
            #if len(which_internodes) > 0:
            #    cutting = which_internodes[0]

            # NEW version: "merge" incident internodes (those sharing a vertex of degree 2), competing against the longest limb
            def _visit(e):
                assert not _visited[e]

                if not is_internode[e]:
                    return 0.0

                tot = mst_w[e]
                _visited[e] = True

                if verbose: print("(%d,%g)"%(e, tot), end=", ")

                for i in [0, 1]:
                    (v, w) = mst_e[e, i], mst_e[e, 1-i]

                    enext = -1
                    for e2 in mst_a[w]:
                        if e2 != e and is_internode[e2]:
                            if enext >= 0:
                                enext = -1
                                break
                            else:
                                enext = e2

                    if enext >= 0 and not _visited[enext]: tot += _visit(enext)

                return tot

            _visited = np.repeat(False, n-1)
            for e in which_internodes:
                # edges are sorted decreasingly

                if _visited[e]: continue

                cur_w = _visit(e)

                if verbose: print("total=%g." % cur_w)

                if cur_w > cutting_w:
                    cutting_w = cur_w
                    cutting_e = e



        if verbose: print("%3d: cutting e=%d w=%g" % (c, cutting_e, cutting_w))

        # if verbose:
        #     _x = []
        #     import pandas as pd
        #     for _w in [which_internodes, which_limbs]:
        #         _x.append(pd.DataFrame(dict(
        #             i=_w,
        #             w=mst_w[_w],
        #             s0=np.sort(mst_s, axis=1)[_w, 0],
        #             s1=np.sort(mst_s, axis=1)[_w, 1],
        #             label=mst_labels[_w]
        #         )))
        #     _x = pd.concat(_x, keys=["internode", "limb"])#.reset_index(level=0)#.groupby(["level_0", "label"]).head(1)
        #     print(_x)
        #     #print(_x.loc[["internode"],:])
        #     #print(_x.loc[["limb"],:])



        if last_iteration:
            assert c == n_clusters
            return (
                labels, mst_w, mst_e, mst_labels, mst_s, mst_skiplist,
                which_internodes, cutting_e,
                which_leaves, which_outliers
            )
        else:
            mst_skiplist.append(cutting_e)



class Lumberdupa(BaseEstimator, ClusterMixin):

    def __init__(
        self,
        *,
        n_clusters,
        noise_cluster=False,
        n_neighbors=5,
        outlier_factor=1.5,
        min_cluster_size=10,
        min_cluster_factor=0.2,
        max_twig_size=5,
        twig_size_factor=0.01,
        M=1,
        cut_internodes=True,
        verbose=False
    ):
        """
        set n_neighbors to 0 to disable internode cutting and noise point detection

        noise_cluster applies to the final labelling

        twig = any small, leafless branch of a woody plant
        internode =  the part of a plant stem between two nodes
        limb = one of the larger branches of a tree
        """

        self.n_clusters          = n_clusters
        self.noise_cluster       = noise_cluster
        self.n_neighbors         = n_neighbors
        self.outlier_factor      = outlier_factor
        self.min_cluster_size    = min_cluster_size
        self.min_cluster_factor  = min_cluster_factor
        self.max_twig_size       = max_twig_size
        self.twig_size_factor    = twig_size_factor
        self.M                   = M
        self.cut_internodes      = cut_internodes
        self.verbose             = verbose


    def fit(self, X, y=None, mst_skiplist=None):
        """
        Perform cluster analysis of a dataset.


        Parameters
        ----------

        X : object
            Typically a matrix with ``n_samples`` rows and ``n_features``
            columns.

        y : None
            Ignored.

        mst_skiplist : None


        Returns
        -------

        self : genieclust.Genie
            The object that the method was called on.
        """

        self.X = np.ascontiguousarray(X)
        self.n_samples_ = self.X.shape[0]

        (
            self.labels_, self._mst_w, self._mst_e, self._mst_labels,
            self._mst_s, self._mst_skiplist,
            self._mst_internodes, self._mst_cutting,
            self._which_leaves, self._which_outliers
        ) = _lumberdupa(
            self.X,
            n_clusters=self.n_clusters,
            noise_cluster=self.noise_cluster,
            n_neighbors=self.n_neighbors,
            outlier_factor=self.outlier_factor,
            min_cluster_size=self.min_cluster_size,
            min_cluster_factor=self.min_cluster_factor,
            max_twig_size=self.max_twig_size,
            twig_size_factor=self.twig_size_factor,
            M=self.M,
            cut_internodes=self.cut_internodes,
            verbose=self.verbose,
            mst_skiplist=mst_skiplist
        )

        return self



    def fit_predict(self, X, y=None, mst_skiplist=None):
        """
        Perform cluster analysis of a dataset and return the predicted labels.


        Parameters
        ----------

        X : object
            See `genieclust.Genie.fit`.

        y : None
            See `genieclust.Genie.fit`.

        mst_skiplist : None


        Returns
        -------

        labels_ : ndarray
            `self.labels_` attribute.


        See also
        --------

        genieclust.Genie.fit

        """
        self.fit(X, y, mst_skiplist)
        return self.labels_
