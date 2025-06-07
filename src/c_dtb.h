/*  A dual-tree Boruvka algorithm for finding minimum spanning trees
 *  wrt the Euclidean dist based on "Fast Euclidean Minimum Spanning Tree:
 *  Algorithm, Analysis, and Applications"
 *  by W.B. March, P. Ram, A.G. Gray
 *
 *  Copyleft (C) 2025, Marek Gagolewski <https://www.gagolewski.com>
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


#ifndef __c_dtb_h
#define __c_dtb_h

#include "c_kdtree.h"

namespace mgtree {

template <typename FLOAT, size_t D>
struct kdtree_node_clusterable : public kdtree_node_base<FLOAT, D>
{
    kdtree_node_clusterable* left;
    kdtree_node_clusterable* right;

    Py_ssize_t cluster_repr;  //< representative point index if all descendants are in the same cluster
    FLOAT cluster_max_dist;  //< minimal distance so far from the current node to another node

    kdtree_node_clusterable() {
        left = nullptr;
        right = nullptr;
    }

    bool is_leaf() const { return left == nullptr /*&& right == nullptr*/; }  // either both null or none
};



template <typename FLOAT, size_t D, typename NODE=kdtree_node_clusterable<FLOAT, D> >
class dtb : public kdtree<FLOAT, D, NODE>
{
protected:
    FLOAT* tree_dist;  //< size n-1
    size_t* tree_ind;  //< size 2*(n-1)
    CDisjointSets ds;
    size_t k;  // current number of MST nodes determined

    std::vector<FLOAT> nn_dist;
    std::vector<size_t> nn_ind;
    std::vector<size_t> nn_from;


    FLOAT dist_between_nodes(const NODE* roota, const NODE* rootb) const
    {
        FLOAT dist = 0.0;
        for (size_t u=0; u<D; ++u) {
            if (rootb->bbox_min[u] > roota->bbox_max[u])
                dist += square(rootb->bbox_min[u] - roota->bbox_max[u]);
            else if (roota->bbox_min[u] > rootb->bbox_max[u])
                dist += square(roota->bbox_min[u] - rootb->bbox_max[u]);
            // else dist += 0.0;
        }
        return dist;
    }


    void update_cluster_data()
    {
        for (Py_ssize_t i=(Py_ssize_t)this->nodes.size()-1; i>=0; --i)
        {
            NODE* curnode = &(this->nodes[i]);
            curnode->cluster_max_dist = INFINITY;

            if (curnode->cluster_repr >= 0) {
                //curnode->cluster_repr = ds.find(curnode->cluster_repr);
                continue;
            }

            if (curnode->is_leaf()) {
                curnode->cluster_repr = ds.find(curnode->idx_from);
                for (size_t j=curnode->idx_from+1; j<curnode->idx_to; ++j) {
                    if (curnode->cluster_repr != ds.find(j)) {
                        curnode->cluster_repr = -1;  // not all are members of the same cluster
                        break;
                    }
                }
            }
            else {
                // descendants were already processed because children in `nodes` occur after their parents

                // if both children only feature members of the same cluster, update the cluster repr for the current node
                if (curnode->left->cluster_repr >= 0 && curnode->right->cluster_repr >= 0) {
                    if (ds.find(curnode->left->cluster_repr) == ds.find(curnode->right->cluster_repr))
                        curnode->cluster_repr = ds.find(curnode->left->cluster_repr);
                }
            }
        }
    }


    void find_mst_first()
    {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t i=0; i<this->n; ++i) {
            kdtree_kneighbours<FLOAT, D, NODE> nn(this->data, this->n, i, nn_dist.data()+i, nn_ind.data()+i, 1);
            nn.find(this->root);
        }

        for (size_t i=0; i<this->n; ++i) {
            if (ds.find(i) != ds.find(nn_ind[i])) {
                tree_ind[k*2+0] = i;
                tree_ind[k*2+1] = nn_ind[i];
                tree_dist[k] = nn_dist[i];
                ds.merge(i, nn_ind[i]);
                k++;
            }
        }
    }


    void find_mst_next(NODE* roota, NODE* rootb)
    {
        GENIECLUST_ASSERT(roota);
        GENIECLUST_ASSERT(rootb);

        if (roota->cluster_repr >= 0 && rootb->cluster_repr >= 0) {
            if (ds.find(roota->cluster_repr) == ds.find(rootb->cluster_repr)) {
                // both consist of members of the same cluster - nothing to do
                return;
            }
        }

        FLOAT dist = dist_between_nodes(roota, rootb);

        if (roota->cluster_max_dist < dist) {
            // we've a better candidate already - nothing to do
            return;
        }

        if (roota->is_leaf()) {
            if (rootb->is_leaf()) {
                for (size_t i=roota->idx_from; i<roota->idx_to; ++i) {
                    Py_ssize_t ds_find_i = ds.find(i);
                    for (size_t j=rootb->idx_from; j<rootb->idx_to; ++j) {
                        if (ds_find_i != ds.find(j)) {
                            FLOAT dij = 0.0;
                            for (size_t u=0; u<D; ++u)
                                dij += square(this->data[i*D+u]-this->data[j*D+u]);
                            if (dij < nn_dist[ds_find_i]) {
                                nn_dist[ds_find_i] = dij;
                                nn_ind[ds_find_i]  = j;
                                nn_from[ds_find_i] = i;
                            }
                        }
                    }
                }

                if (roota->cluster_repr >= 0) {
                    roota->cluster_max_dist = nn_dist[ds.find(roota->cluster_repr)];
                }
                else {
                    roota->cluster_max_dist = -INFINITY;
                    for (size_t i=roota->idx_from; i<roota->idx_to; ++i) {
                        FLOAT dist_cur = nn_dist[ds.find(i)];
                        if (dist_cur > roota->cluster_max_dist)
                            roota->cluster_max_dist = dist_cur;
                    }
                }
            }
            else {
                find_mst_next(roota, rootb->left);
                find_mst_next(roota, rootb->right);
            }
        }
        else {
            if (rootb->is_leaf()) {
                find_mst_next(roota->left, rootb);
                find_mst_next(roota->right, rootb);
            }
            else {
                find_mst_next(roota->left,  rootb->left);
                find_mst_next(roota->left,  rootb->right);
                find_mst_next(roota->right, rootb->left);
                find_mst_next(roota->right, rootb->right);
            }
            roota->cluster_max_dist = std::max(
                roota->left->cluster_max_dist,
                roota->right->cluster_max_dist
            );
        }
    }


    void find_mst()
    {
        // 1st iteration: connect nearest neighbours with each other
        find_mst_first();

        // TODO: start with max_leaf_size=~32, then switch to a smaller one?


        while (k < this->n-1) {
            update_cluster_data();  // reset cluster_max_dist and set up cluster_repr

            // NOTE: this could be a fancy data structure that holds only
            // the representatives of the current clusters, but why bother,
            // time'll be >= ~(n\log n) anyway
            for (size_t i=0; i<this->n; ++i) nn_dist[i] = INFINITY;
            for (size_t i=0; i<this->n; ++i) nn_ind[i]  = this->n;
            for (size_t i=0; i<this->n; ++i) nn_from[i] = this->n;

            find_mst_next(this->root, this->root);

            for (size_t i=0; i<this->n; ++i) {
                if (nn_ind[i] < this->n && ds.find(i) != ds.find(nn_ind[i])) {
                    GENIECLUST_ASSERT(ds.find(i) == ds.find(nn_from[i]));
                    tree_ind[k*2+0] = nn_from[i];
                    tree_ind[k*2+1] = nn_ind[i];
                    tree_dist[k] = nn_dist[i];
                    ds.merge(i, nn_ind[i]);
                    k++;
                }
            }
        }
    }


public:
    dtb()
        : kdtree<FLOAT, D, NODE>()
    {

    }


    dtb(FLOAT* data, const size_t n, const size_t max_leaf_size=32)
        : kdtree<FLOAT, D, NODE>(data, n, max_leaf_size), ds(n), k(0), nn_dist(n), nn_ind(n), nn_from(n)
    {

    }



    void boruvka(FLOAT* tree_dist, size_t* tree_ind)
    {
        this->tree_dist = tree_dist;
        this->tree_ind = tree_ind;

        if (ds.get_k() != (Py_ssize_t)this->n) ds.reset();
        k = 0;

        for (size_t i=0; i<this->n-1; ++i)     tree_dist[i] = INFINITY;
        for (size_t i=0; i<2*(this->n-1); ++i) tree_ind[i]  = this->n;

        for (Py_ssize_t i=(Py_ssize_t)this->nodes.size()-1; i>=0; i--)
            this->nodes[i].cluster_repr = -1;

        find_mst();
    }
};




template <typename FLOAT, size_t D, typename TREE>
void mst(
    TREE& tree,
    FLOAT* tree_dist,  // size n-1
    size_t* tree_ind   // size 2*(n-1)
) {
    size_t n = tree.get_n();
    const size_t* perm = tree.get_perm().data();

    tree.boruvka(tree_dist, tree_ind);

    std::vector< CMstTriple<FLOAT> > mst(n-1);

    for (size_t i=0; i<n-1; ++i) {
        GENIECLUST_ASSERT(tree_ind[2*i+0] != tree_ind[2*i+1]);
        GENIECLUST_ASSERT(tree_ind[2*i+0] < n);
        GENIECLUST_ASSERT(tree_ind[2*i+1] < n);

        mst[i] = CMstTriple<FLOAT>(perm[tree_ind[2*i+0]], perm[tree_ind[2*i+1]], tree_dist[i]);
    }

    std::sort(mst.begin(), mst.end());

    for (size_t i=0; i<n-1; ++i) {
        tree_dist[i]    = sqrt(mst[i].d);
        tree_ind[2*i+0] = mst[i].i1;  // i1 < i2
        tree_ind[2*i+1] = mst[i].i2;
    }
}



};  // namespace

#endif
