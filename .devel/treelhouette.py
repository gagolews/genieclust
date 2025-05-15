"""
Treelhouette Score
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
import mst_examples



def get_intercluster_distances(L):
    X = L.X
    y_pred = L.labels_+1
    n = X.shape[0]
    mst_e = L._tree_e
    mst_w = L._tree_w
    n_clusters = L.n_clusters
    skiplist = L._tree_cutlist

    mst_a = [ [] for i in range(n) ]
    for i in range(n-1):
        mst_a[mst_e[i, 0]].append(i)
        mst_a[mst_e[i, 1]].append(i)
    for i in range(n):
        mst_a[i] = np.array(mst_a[i])
    mst_a = np.array(mst_a, dtype="object")

    def visit(v, e):  # from v along e
        iv = int(mst_e[e, 1] == v)
        w = mst_e[e, 1-iv]

        if y_pred[w] > 0:
            # reached a coloured vertex - stop
            return [(y_pred[w], 0.0)]
        if len(mst_a[w]) == 1:
            # reached a leaf - stop
            return []

        res = []
        for e2 in mst_a[w]:
            if mst_e[e2, 0] != v and mst_e[e2, 1] != v:
                res += [(l, w+mst_w[e2]) for (l, w) in visit(w, e2)]

        return res

    D = np.ones((n_clusters, n_clusters))*np.inf
    for e in skiplist:
        res = []
        v, w = mst_e[e, :]
        res_v = visit(v, e)
        res_w = visit(w, e)
        for (lv, dv) in res_v:
            for (lw, dw) in res_w:
                D[lv-1, lw-1] = np.minimum(D[lv-1, lw-1], dv+dw+mst_w[e])
                D[lw-1, lv-1] = D[lv-1, lw-1]
    return D



def treelhouette_score(L, skip_leaves=False):
    X = L.X
    y_pred = L.labels_.copy()+1
    mst_labels = L._tree_labels.copy()

    mst_e = L._tree_e
    mst_w = L._tree_w
    mst_s = L._tree_s
    min_mst_s = np.min(mst_s, axis=1)
    n = X.shape[0]

    if skip_leaves:
        y_pred[mst_e[(mst_s[:,0] <= 1) & (mst_labels > 0), 1]] = 0
        y_pred[mst_e[(mst_s[:,1] <= 1) & (mst_labels > 0), 0]] = 0
        mst_labels[   (min_mst_s <= 1) & (mst_labels > 0)] = 0

    cluster_distances = get_intercluster_distances(L)
    print(cluster_distances)

    # leave the diagonal to inf
    min_intercluster_distances = np.min(cluster_distances, axis=0)
    a = mst_w[mst_labels > 0]
    l = mst_labels[mst_labels > 0]
    b = min_intercluster_distances[l - 1]
    s = np.where(a<b, 1.0 - a/b, b/a - 1.0)
    treelhouette_score = np.mean(s)
    weighted_treelhouette_score = np.mean(mst_examples.aggregate(s, mst_labels[mst_labels>0], np.mean)[0])
    return treelhouette_score, weighted_treelhouette_score

