"""
A generalised version of normalised clustering accuracy
and normalised pivoted accuracy allowing k_pred >= k_true.

Solves a 0-1 integer programming problem; hence, may be slow.

Requires GLPK (GNU Linear Programming Kit)

M. Gagolewski, Normalised Clustering Accuracy: An Asymmetric External Cluster
Validity Measure. Journal of Classification 42, 2–30, 2025.
https://doi.org/10.1007/s00357-024-09482-2
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
import glpk
import genieclust


def generalized_lsap(C):

    k_true = C.shape[0]
    k_pred = C.shape[1]
    n = np.sum(C)

    assert k_true <= k_pred

    lp = glpk.LPX()                     # Create empty problem instance
    lp.obj.maximize = True
    lp.cols.add(k_true*k_pred)
    for i in range(k_true*k_pred):
        lp.cols[i].bounds = 0.0, 1.0     # 0 <= s_ij <= 1
        lp.cols[i].kind = int

    lp.obj[:] = C.ravel().tolist()

    lp.rows.add(k_true+k_pred+1)
    for i in range(k_true+k_pred):
        lp.rows[i].bounds = (1, None)
    lp.rows[k_true+k_pred].bounds = max(k_pred, k_true) #(None, max(k_pred, k_true))


    D = np.zeros((k_true+k_pred+1, k_true*k_pred))
    for i in range(k_true):
        D[i,((i)*k_pred):((i+1)*k_pred)] = 1
    for i in range(k_pred):
        D[i+k_true, i::k_pred] = 1
    D[-1,:] = 1

    lp.matrix = D.ravel().tolist()

    res = lp.simplex()
    assert res is None
    R = np.array([col.value for col in lp.cols]).reshape(k_true, k_pred)

    return R


def generalized_normalized_pivoted_accuracy(y_true, y_pred):
    C = genieclust.compare_partitions.confusion_matrix(y_true, y_pred, force_double=True)
    k_true = C.shape[0]
    n = C.sum()
    R = generalized_lsap(C)
    return (np.sum(C*R)/n-1.0/k_true)/(1.0-1.0/k_true)


def generalized_normalized_clustering_accuracy(y_true, y_pred):
    C = genieclust.compare_partitions.confusion_matrix(y_true, y_pred, force_double=True)
    k_true = C.shape[0]
    C_rowsum = C.sum(axis=1)
    Cp = C/C_rowsum.reshape(-1,1)
    R = generalized_lsap(Cp)
    return (np.sum(Cp*R)/k_true-1.0/k_true)/(1.0-1.0/k_true)


# y_true = [1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1]
# y_pred = [1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 3, 3, 4]
# print(genieclust.compare_partitions.confusion_matrix(y_true, y_pred, force_double=True))
#
# print(genieclust.compare_partitions.normalized_pivoted_accuracy(y_true, y_pred))
# print(genieclust.compare_partitions.normalized_clustering_accuracy(y_true, y_pred))
# print(genieclust.compare_partitions.adjusted_rand_score(y_true, y_pred))
# print(genieclust.compare_partitions.normalized_mi_score(y_true, y_pred))
#
# print(generalized_normalized_pivoted_accuracy(y_true, y_pred))
# print(generalized_normalized_clustering_accuracy(y_true, y_pred))
