/*  A dual-tree Boruvka algorithm for finding minimum spanning trees
 *      wrt the Euclidean dist
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
    Py_ssize_t cluster_min_dist;  //< minimal distance so far from the current node to another node

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



    FLOAT dist_between_nodes(const NODE* roota, const NODE* rootb) const
    {
        FLOAT dist = 0.0;
        GENIECLUST_ASSERT(false);  // TODO .........................................................
        // for (size_t u=0; u<D; ++u) {
        //     if (x[u] < root->bbox_min[u])
        //         dist += square(x[u] - root->bbox_min[u]);
        //     else if (x[u] > root->bbox_max[u])
        //         dist += square(x[u] - root->bbox_max[u]);
        //     // else dist += 0.0;
        // }
        return dist;
    }


    void update_cluster_repr(NODE*& root, bool reset_all)
    {
        if (!reset_all && root->cluster_repr >= 0) {
            root->cluster_repr = ds.find(root->cluster_repr);
            return;
        }

        if (root->is_leaf()) {
            root->cluster_repr = ds.find(root->idx_from);
            for (size_t i=root->idx_from+1; i<root->idx_to; ++i) {
                if (root->cluster_repr != ds.find(i)) {
                    root->cluster_repr = -1;  // not all are members of the same cluster
                    break;
                }
            }
        }
        else {
            update_cluster_repr(root->left,  reset_all);
            update_cluster_repr(root->right, reset_all);

            // if both children only feature members of the same cluster, update the cluster repr for the current node
            if (root->left->cluster_repr >= 0 && root->right->cluster_repr >= 0) {
                if (ds.find(root->left->cluster_repr) == ds.find(root->right->cluster_repr))
                    root->cluster_repr = ds.find(root->left->cluster_repr);
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


    void find_mst_next(const NODE* roota, const NODE* rootb)
    {
        if (roota->cluster_repr >= 0 && rootb->cluster_repr >= 0) {
            if (ds.find(roota->cluster_repr) == ds.find(rootb->cluster_repr)) {
                // both consist of members of the same cluster - nothing to do
                return;
            }
        }

        FLOAT dist = dist_between_nodes(roota, rootb);  // ........ TODO.....

          // ........ TODO.....
    }


    void find_mst()
    {
        // 1st iteration: connect nearest neighbours with each other
        find_mst_first();

        // TODO: use std::deque for node allocation/storage?


        // TODO: start with max_leaf_size=~32, then switch to a smaller one?

        // A dual-tree Boruvka algorithm
        // based on "Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis, and Applications"
        // by W.B. March, P. Ram, A.G. Gray

        return;  // TODO ....

        bool first_iter = true;
        while (k < this->n-1) {
            update_cluster_repr(this->root, first_iter);
            first_iter = false;

            // NOTE: this could be a fancy data structure that holds only
            // the representatives of the current clusters, but why bother
            for (size_t i=0; i<this->n; ++i) nn_dist[i] = INFINITY;
            for (size_t i=0; i<this->n; ++i) nn_ind[i] = this->n;

            find_mst_next(this->root, this->root);

            for (size_t i=0; i<this->n; ++i) {
                if (nn_ind[i] >= this->n) continue;
                GENIECLUST_ASSERT(i == (size_t)ds.find(i));

                if (ds.find(i) != ds.find(nn_ind[i])) {
                    tree_ind[k*2+0] = i;
                    tree_ind[k*2+1] = nn_ind[i];
                    tree_dist[k] = nn_dist[i];
                    ds.merge(i, nn_ind[i]);
                    k++;
                }
            }
        }

        // TODO: delme:
        update_cluster_repr(this->root, false);// TODO: delme
        GENIECLUST_ASSERT(this->root->cluster_repr == 0);// TODO: delme
        update_cluster_repr(this->root, true);// TODO: delme
        GENIECLUST_ASSERT(this->root->cluster_repr == 0);// TODO: delme
    }


public:
    dtb()
        : kdtree<FLOAT, D, NODE>()
    {

    }


    dtb(FLOAT* data, const size_t n, const size_t max_leaf_size=32)
        : kdtree<FLOAT, D, NODE>(data, n, max_leaf_size), ds(n), k(0), nn_dist(n), nn_ind(n)
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
