"""
The Lumbermark Clustering Algorithm
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2018-2025, Marek Gagolewski <https://www.gagolewski.com>      #
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



def lumbermark(
    X,
    n_clusters,
    min_cluster_size=None,
    max_twig_size=None,
    noise_cluster=False,
    n_neighbors=5,
    cut_internodes=True,
    skiplist=None
):
    """
    set n_neighbors to 0 to disable cutting a internodes and noise point detection

    twig = any small, leafless branch of a woody plant
    internode =  the part of a plant stem between two nodes
    limb = one of the larger branches of a tree
    """
    X = np.ascontiguousarray(X)
    n = X.shape[0]

    assert n_clusters > 0
    assert n_neighbors >= 0

    if n_neighbors == 0:
        cut_internodes = False
        mark_noise = False

    if min_cluster_size is None:
        min_cluster_size = max(15, 0.1*n/n_clusters)

    if max_twig_size is None:
        max_twig_size = max(5, 0.01*n/n_clusters)

    if skiplist is None:
        skiplist = []

    SKIP = np.iinfo(int).min  # must be negative
    UNSET = -1 # must be negative
    NOISE = 0  # must be 0


    if n_neighbors > 0:
        kd = scipy.spatial.KDTree(X)
        nn_w, nn_a = kd.query(X, n_neighbors+1)
        nn_w = np.array(nn_w)[:, 1:]  # exclude self
        nn_a = np.array(nn_a)[:, 1:]


    mst_w, mst_e = genieclust.internal.mst_from_distance(X, "euclidean")
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
                    return  # we want a vertex that's incident of only one such edge

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
        for s in skiplist:
            mst_labels[s] = SKIP  # skiplist


        c, counts, min_mst_s = mark_clusters()

        if c >= n_clusters and not noise_cluster:
            assert c == n_clusters
            return labels


        if n_neighbors == 0:
            is_outlier = np.repeat(False, n)
        else:
            # a point is an outlier if its k-th nearest neighbour is just too far away -
            # - relative to the "typical" distances to k-nearest neighbours within the point's cluster
            # (each cluster can be of different density, so we take this into account)
            # or may be considered as an outlier after a cluster breaks down to smaller ones
            outlier_k = -1  # which nearest neighbour do we take into account?
            Q13 = np.array([np.percentile(nn_w[labels==j+1, outlier_k], [25, 75]) for j in range(c)])
            bnd_outlier = np.r_[np.nan, (Q13[:, 1]+1.5*(Q13[:, 1]-Q13[:, 0]))]
            is_outlier = (nn_w[:, outlier_k] > bnd_outlier[labels])


            # cut out all twigs ("small" branches of size <= max_twig_size)
            # that solely consist of outliers, starting from leaves

            if max_twig_size > 0:
                # leaves are vertices of degree 1 (in the forest with the SKIP edges removed!)
                leaves = mst_e[:,::-1][mst_s == 1]
                #is_leaf = np.repeat(False, n)
                #is_leaf[leaves] = True

                for v in leaves:  #np.flatnonzero(is_leaf):
                    mark_noise(v)


                mst_s[:,:] = 0
                labels[labels != NOISE] = UNSET
                mst_labels[(mst_labels != SKIP) & (mst_labels != NOISE)] = UNSET
                c, counts, min_mst_s = mark_clusters()


        if c >= n_clusters:
            assert c == n_clusters
            return labels


        if not cut_internodes:
            internodes = np.repeat(False, n-1)
        else:
            # an internode is an edge incident to an outlier, whose removal leads to a new cluster of "considerable" size
            internodes  = (is_outlier[mst_e[:,0]] | is_outlier[mst_e[:,1]])
            internodes &= (min_mst_s >= min_cluster_size)
            # internodes &= (min_mst_s >= min_cluster_size_frac*np.sum(mst_s, axis=1))
            internodes &= (mst_labels != SKIP)

        which_internodes = np.flatnonzero(internodes)
        #print("which_internodes=%s" % (which_internodes, ))


        limbs = (mst_labels > 0)
        limbs &= (min_mst_s >= min_cluster_size)
        # limbs &= (min_mst_s >= min_cluster_size_frac*np.sum(mst_s, axis=1))
        limbs &= (mst_labels != SKIP)
        limbs &= ~internodes

        which_limbs = np.flatnonzero(limbs)
        #print("which_limbs=%s" % (which_limbs, ))

        # _x = []
        # import pandas as pd
        # for _w in [which_internodes, which_limbs]:
        #     _x.append(pd.DataFrame(np.c_[_w, mst_w[_w], np.sort(mst_s, axis=1)[_w,:], mst_labels[_w]], columns=["i", "w", "s0", "s1", "label"]))
        # _x = pd.concat(_x, keys=["cut", "outlier"])#.reset_index(level=0)#.groupby(["level_0", "label"]).head(1)
        # print(_x)


        cutting = which_internodes[0] if len(which_internodes) > 0 else which_limbs[0]
        #print("cutting=%d" % cutting)

        skiplist.append(cutting)

