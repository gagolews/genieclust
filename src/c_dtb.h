/*  A dual-tree Boruvka algorithm for finding minimum spanning trees
 *  wrt the Euclidean distance.
 *
 *  It is based on "Fast Euclidean Minimum Spanning Tree:
 *  Algorithm, Analysis, and Applications"
 *  by W.B. March, P. Ram, A.G. Gray, ACM SIGKDD 2010.
 *  It features some further performance enhancements.
 *
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
#include "c_disjoint_sets.h"
#include "c_mst_triple.h"

#define DCORE_DIST_ADJ (1<<26)


namespace mgtree {

template <typename FLOAT, Py_ssize_t D>
struct kdtree_node_clusterable : public kdtree_node_base<FLOAT, D>
{
    kdtree_node_clusterable* left;
    kdtree_node_clusterable* right;

    Py_ssize_t cluster_repr;  //< representative point index if all descendants are in the same cluster, -1 otherwise
    FLOAT cluster_max_dist;

    FLOAT min_dcore;

    kdtree_node_clusterable() {
        left = nullptr;
        right = nullptr;
        //cluster_repr = -1;
        //cluster_max_dist = INFINITY;
        //min_dcore = 0.0;
    }

    inline bool is_leaf() const {
        return left == nullptr /*&& right == nullptr*/; // either both null or none
    }
};




template <
    typename FLOAT,
    Py_ssize_t D,
    typename DISTANCE=kdtree_distance_sqeuclid<FLOAT,D>,
    typename NODE=kdtree_node_clusterable<FLOAT, D>
>
class dtb : public kdtree<FLOAT, D, DISTANCE, NODE>
{
protected:
    FLOAT*  tree_dist;     //< size n-1
    Py_ssize_t* tree_ind;  //< size 2*(n-1)
    Py_ssize_t  tree_num;  // number of MST edges already found
    CDisjointSets ds;

    std::vector<FLOAT>  nn_dist;
    std::vector<Py_ssize_t> nn_ind;
    std::vector<Py_ssize_t> nn_from;

    const Py_ssize_t first_pass_max_brute_size;  // used in the first iter (finding 1-nns)

    const Py_ssize_t M;  // mutual reachability distance - "smoothing factor"
    std::vector<FLOAT> dcore;  // distances to the (M-1)-th nns of each point if M>1


    struct kdtree_node_orderer {
        NODE* closer_node;
        NODE* farther_node;
        FLOAT closer_dist;
        FLOAT farther_dist;

        kdtree_node_orderer(NODE* from, NODE* to1, NODE* to2, bool use_min_dcore=false)
        {
            closer_dist  = DISTANCE::node_node(
                from->bbox_min.data(), from->bbox_max.data(),
                 to1->bbox_min.data(),  to1->bbox_max.data()
            );

            farther_dist = DISTANCE::node_node(
                from->bbox_min.data(), from->bbox_max.data(),
                 to2->bbox_min.data(),  to2->bbox_max.data()
            );

            if (use_min_dcore) {
                closer_dist = max3(closer_dist, from->min_dcore, to1->min_dcore);
                farther_dist = max3(farther_dist, from->min_dcore, to2->min_dcore);
            }

            if (closer_dist <= farther_dist) {
                closer_node  = to1;
                farther_node = to2;
            }
            else {
                std::swap(closer_dist, farther_dist);
                closer_node  = to2;
                farther_node = to1;
            }
        }
    };


    template <bool USE_DCORE>
    inline void leaf_vs_leaf(NODE* roota, NODE* rootb)
    {
        // assumes ds.find(i) == ds.get_parent(i) for all i!
        const FLOAT* _x = this->data + roota->idx_from*D;
        for (Py_ssize_t i=roota->idx_from; i<roota->idx_to; ++i, _x += D) {
            Py_ssize_t ds_find_i = ds.get_parent(i);
            for (Py_ssize_t j=rootb->idx_from; j<rootb->idx_to; ++j) {
                Py_ssize_t ds_find_j = ds.get_parent(j);
                if (ds_find_i != ds_find_j) {
                    FLOAT dij = DISTANCE::point_point(_x, this->data+j*D);
                    if (USE_DCORE) {
                        // pulled-away from each other, but ordered w.r.t. the original pairwise distances (increasingly)
                        FLOAT dcore_max = std::max(dcore[i], dcore[j]);
                        if (dij <= dcore_max)
                            dij = dcore_max+dij/DCORE_DIST_ADJ;
                    }
                    if (dij < nn_dist[ds_find_i]) {
                        nn_dist[ds_find_i] = dij;
                        nn_ind[ds_find_i]  = j;
                        nn_from[ds_find_i] = i;
                    }
                }
            }
        }
    }


    void update_min_dcore()
    {
        for (Py_ssize_t i=(Py_ssize_t)this->nodes.size()-1; i>=0; --i)
        {
            NODE* curnode = &(this->nodes[i]);
            if (M <= 2)
                curnode->min_dcore = 0.0;
            else {
                if (curnode->is_leaf()) {
                    curnode->min_dcore = dcore[curnode->idx_from];
                    for (Py_ssize_t j=curnode->idx_from+1; j<curnode->idx_to; ++j) {
                        if (dcore[j] < curnode->min_dcore)
                            curnode->min_dcore = dcore[j];
                    }
                }
                else {
                    // all descendants have already been processed as children in `nodes` occur after their parents
                    curnode->min_dcore = std::min(
                        curnode->left->min_dcore,
                        curnode->right->min_dcore
                    );
                }
            }
        }
    }


    void update_cluster_data()
    {
        for (Py_ssize_t i=0; i<this->n; ++i)
            this->ds.find(i);
        // now ds.find(i) == ds.get_parent(i) for all i

        for (Py_ssize_t i=(Py_ssize_t)this->nodes.size()-1; i>=0; --i)
        {
            NODE* curnode = &(this->nodes[i]);
            curnode->cluster_max_dist = INFINITY;

            if (curnode->cluster_repr >= 0) {
                curnode->cluster_repr = ds.get_parent(curnode->cluster_repr);  // helpful
                continue;
            }

            if (curnode->is_leaf()) {
                curnode->cluster_repr = ds.get_parent(curnode->idx_from);
                for (Py_ssize_t j=curnode->idx_from+1; j<curnode->idx_to; ++j) {
                    if (curnode->cluster_repr != ds.get_parent(j)) {
                        curnode->cluster_repr = -1;  // not all are members of the same cluster
                        break;
                    }
                }
            }
            else {
                // all descendants have already been processed as children in `nodes` occur after their parents
                if (curnode->left->cluster_repr >= 0 && curnode->right->cluster_repr >= 0) {
                    // if both children only feature members of the same cluster, update the cluster repr for the current node;
                    Py_ssize_t left_cluster_id  = curnode->left->cluster_repr;  //ds.find(curnode->left->cluster_repr); <- done above
                    Py_ssize_t right_cluster_id = curnode->right->cluster_repr; //ds.find(curnode->right->cluster_repr);<- done above
                    if (left_cluster_id == right_cluster_id)
                        curnode->cluster_repr = left_cluster_id;
                }
                // else curnode->cluster_repr = -1;  // it already is
            }
        }
    }


    void find_mst_first_1()
    {
        GENIECLUST_ASSERT(M <= 2);
        const Py_ssize_t k = 1;

        // find 1-nns of each point using max_brute_size,
        // preferably with max_brute_size>max_leaf_size
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (Py_ssize_t i=0; i<this->n; ++i) {
            kdtree_kneighbours<FLOAT, D, DISTANCE, NODE> nn(
                this->data, this->n, i, nn_dist.data()+i, nn_ind.data()+i, k,
                first_pass_max_brute_size
            );
            nn.find(this->root);
            if (M == 2) dcore[i]   = nn_dist[i];
        }

        // connect nearest neighbours with each other
        for (Py_ssize_t i=0; i<this->n; ++i) {
            if (ds.find(i) != ds.find(nn_ind[i])) {
                tree_ind[tree_num*2+0] = i;
                tree_ind[tree_num*2+1] = nn_ind[i];
                tree_dist[tree_num] = nn_dist[i];
                ds.merge(i, nn_ind[i]);
                tree_num++;
            }
        }
    }


    void find_mst_first_M()
    {
        GENIECLUST_ASSERT(M>1);
        const Py_ssize_t k = M-1;
        // find (M-1)-nns of each point
        std::vector<FLOAT> knn_dist(this->n*k);
        std::vector<Py_ssize_t> knn_ind(this->n*k);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (Py_ssize_t i=0; i<this->n; ++i) {
            kdtree_kneighbours<FLOAT, D, DISTANCE, NODE> nn(
                this->data, this->n, i, knn_dist.data()+k*i, knn_ind.data()+k*i, k,
                first_pass_max_brute_size
            );
            nn.find(this->root);
            dcore[i]   = knn_dist[i*k+(k-1)];
        }

        // k-nns wrt Euclidean distances are not necessarily k-nns wrt mutreach
        for (Py_ssize_t i=0; i<this->n; ++i) {
            for (Py_ssize_t v=0; v<k; ++v) {
                Py_ssize_t j = knn_ind[i*k+v];
                if (dcore[i] >= dcore[j]) {
                    // j is the nearest neighbour of i wrt mutreach dist.
                    if (ds.find(i) != ds.find(j)) {
                        tree_ind[tree_num*2+0] = i;
                        tree_ind[tree_num*2+1] = j;
                        tree_dist[tree_num] = dcore[i]+knn_dist[i*k+v]/DCORE_DIST_ADJ;//dcore[i];
                        ds.merge(i, j);
                        tree_num++;
                    }
                    break;
                }
            }
        }


        // // try to reuse the computed k-nns (no benefit)
        // std::vector<Py_ssize_t> knn_next(this->n, 0);
        // bool changed;
        // do {
        //     changed = false;
        //     for (Py_ssize_t i=0; i<this->n; ++i) nn_dist[i] = INFINITY;
        //     for (Py_ssize_t i=0; i<this->n; ++i) nn_ind[i]  = this->n;
        //     for (Py_ssize_t i=0; i<this->n; ++i) nn_from[i] = this->n;
        //
        //     for (Py_ssize_t i=0; i<this->n; ++i) {
        //         bool found = false;
        //         Py_ssize_t ds_find_i = ds.find(i);
        //         if (nn_dist[ds_find_i] < 0) continue;
        //
        //         while (knn_next[i] < k) {
        //             Py_ssize_t j = knn_ind[i*k+knn_next[i]];
        //             if (dcore[i] >= dcore[j]) {
        //                 if (ds_find_i != ds.find(j)) {
        //                     FLOAT d = dcore[i]+knn_dist[i*k+knn_next[i]]/DCORE_DIST_ADJ;//dcore[i];
        //                     if (d < nn_dist[ds_find_i]) {
        //                         nn_dist[ds_find_i] = d;
        //                         nn_ind[ds_find_i] = j;
        //                         nn_from[ds_find_i] = i;
        //                     }
        //                     found = true;
        //                     break;
        //                 }
        //             }
        //             knn_next[i]++;
        //         }
        //         if (!found) {
        //             nn_dist[ds_find_i] = -INFINITY;  // disable
        //             nn_ind[ds_find_i] = this->n;
        //             nn_from[ds_find_i] = this->n;
        //         }
        //     }
        //
        //     for (Py_ssize_t i=0; i<this->n; ++i) {
        //         if (nn_ind[i] < this->n && ds.find(i) != ds.find(nn_ind[i])) {
        //             GENIECLUST_ASSERT(ds.find(i) == ds.find(nn_from[i]));
        //             tree_ind[tree_num*2+0] = nn_from[i];
        //             tree_ind[tree_num*2+1] = nn_ind[i];
        //             tree_dist[tree_num] = nn_dist[i];
        //             ds.merge(i, nn_ind[i]);
        //             tree_num++;
        //
        //             //knn_next[nn_from[i]]++;
        //             changed = true;
        //         }
        //     }
        // } while (changed && tree_num < this->n-1);
    }


    void find_mst_first()
    {
        // the 1st iteration: connect nearest neighbours with each other
        if (M <= 2) find_mst_first_1();
        else        find_mst_first_M();
    }


    void find_mst_next(NODE* roota, NODE* rootb)
    {
        //GENIECLUST_ASSERT(roota);
        //GENIECLUST_ASSERT(rootb);

        // we have ds.find(i) == ds.get_parent(i) for all i!

        if (roota->cluster_repr >= 0 && roota->cluster_repr == rootb->cluster_repr) {
            // both consist of members of the same cluster - nothing to do
            return;
        }

        // pruning below!
        //FLOAT dist = distance_node_node_sqeuclid(roota, rootb);
        //if (roota->cluster_max_dist < dist) {
        //    // we've a better candidate already - nothing to do
        //    return;
        //}

        if (roota->is_leaf()) {
            if (rootb->is_leaf()) {

                if (M>2) leaf_vs_leaf<true>(roota, rootb);
                else     leaf_vs_leaf<false>(roota, rootb);

                if (roota->cluster_repr >= 0) {  // all points are in the same cluster
                    roota->cluster_max_dist = nn_dist[roota->cluster_repr];
                }
                else {
                    roota->cluster_max_dist = nn_dist[ds.get_parent(roota->idx_from)];
                    for (Py_ssize_t i=roota->idx_from+1; i<roota->idx_to; ++i) {
                        FLOAT dist_cur = nn_dist[ds.get_parent(i)];
                        if (dist_cur > roota->cluster_max_dist)
                            roota->cluster_max_dist = dist_cur;
                    }
                }
            }
            else {
                // closer node first -> faster!
                kdtree_node_orderer sel(roota, rootb->left, rootb->right, (M>2));

                // prune nodes too far away if we have better candidates
                if (roota->cluster_max_dist >= sel.closer_dist) {
                    find_mst_next(roota, sel.closer_node);
                    if (roota->cluster_max_dist >= sel.farther_dist)
                        find_mst_next(roota, sel.farther_node);
                }


                // roota->cluster_max_dist updated above
            }
        }
        else {  // roota is not a leaf
            if (rootb->is_leaf()) {
                kdtree_node_orderer sel(rootb, roota->left, roota->right, (M>2));
                if (sel.closer_node->cluster_max_dist >= sel.closer_dist)
                    find_mst_next(sel.closer_node, rootb);
                if (sel.farther_node->cluster_max_dist >= sel.farther_dist)  // separate if!
                    find_mst_next(sel.farther_node, rootb);
            }
            else {
                kdtree_node_orderer sel(roota->left, rootb->left, rootb->right, (M>2));
                if (roota->left->cluster_max_dist >= sel.closer_dist) {
                    find_mst_next(roota->left, sel.closer_node);
                    if (roota->left->cluster_max_dist >= sel.farther_dist)
                        find_mst_next(roota->left, sel.farther_node);
                }

                sel = kdtree_node_orderer(roota->right, rootb->left, rootb->right, (M>2));
                if (roota->right->cluster_max_dist >= sel.closer_dist) {
                    find_mst_next(roota->right, sel.closer_node);
                    if (roota->right->cluster_max_dist >= sel.farther_dist)
                        find_mst_next(roota->right, sel.farther_node);
                }
            }

            roota->cluster_max_dist = std::max(
                roota->left->cluster_max_dist,
                roota->right->cluster_max_dist
            );
        }
    }


    void find_mst()
    {
        // the 1st iteration: connect nearest neighbours with each other
        find_mst_first();

        if (M>2) update_min_dcore();


        std::vector<Py_ssize_t> ds_parents(this->n);
        Py_ssize_t ds_k;

        while (tree_num < this->n-1) {
            // reset cluster_max_dist and set up cluster_repr,
            // ensure ds.find(i) == ds.get_parent(i) for all i
            update_cluster_data();

            ds_k = 0;
            for (Py_ssize_t i=0; i<this->n; ++i) {
                if (i == ds.get_parent(i)) {
                    nn_dist[i] = INFINITY;
                    nn_ind[i]  = this->n;
                    nn_from[i] = this->n;
                    ds_parents[ds_k++] = i;
                }
            }
            // for (Py_ssize_t i=0; i<this->n; ++i) nn_dist[i] = INFINITY;
            // for (Py_ssize_t i=0; i<this->n; ++i) nn_ind[i]  = this->n;
            // for (Py_ssize_t i=0; i<this->n; ++i) nn_from[i] = this->n;

            find_mst_next(this->root, this->root);

            for (Py_ssize_t j=0; j<ds_k; ++j) {
                Py_ssize_t i = ds_parents[j];
                GENIECLUST_ASSERT(nn_ind[i] < this->n);
                if (ds.find(i) != ds.find(nn_ind[i])) {
                    GENIECLUST_ASSERT(ds.find(i) == ds.find(nn_from[i]));
                    tree_ind[tree_num*2+0] = nn_from[i];
                    tree_ind[tree_num*2+1] = nn_ind[i];
                    tree_dist[tree_num] = nn_dist[i];
                    ds.merge(i, nn_ind[i]);
                    tree_num++;
                }
            }
        }
    }


public:
    dtb()
        : kdtree<FLOAT, D, DISTANCE, NODE>()
    {

    }


    dtb(
        FLOAT* data, const Py_ssize_t n, const Py_ssize_t M=1,
        const Py_ssize_t max_leaf_size=4,
        const Py_ssize_t first_pass_max_brute_size=16
    ) :
        kdtree<FLOAT, D, DISTANCE, NODE>(data, n, max_leaf_size), tree_num(0),
        ds(n), nn_dist(n), nn_ind(n), nn_from(n),
        first_pass_max_brute_size(first_pass_max_brute_size), M(M)
    {
        if (M > 2) dcore.resize(n);
    }



    void boruvka(FLOAT* tree_dist, Py_ssize_t* tree_ind)
    {
        this->tree_dist = tree_dist;
        this->tree_ind = tree_ind;

        if (ds.get_k() != (Py_ssize_t)this->n) ds.reset();
        tree_num = 0;

        for (Py_ssize_t i=0; i<this->n-1; ++i)     tree_dist[i] = INFINITY;
        for (Py_ssize_t i=0; i<2*(this->n-1); ++i) tree_ind[i]  = this->n;

        for (Py_ssize_t i=(Py_ssize_t)this->nodes.size()-1; i>=0; i--)
            this->nodes[i].cluster_repr = -1;

        find_mst();
    }


    inline const FLOAT* get_dcore() const { return this->dcore.data(); }
};




template <typename FLOAT, Py_ssize_t D, typename TREE>
void mst(
    TREE& tree,
    FLOAT* tree_dist,   // size n-1
    Py_ssize_t* tree_ind,   // size 2*(n-1),
    FLOAT* d_core=nullptr,  // size n
    bool order=true
) {
    Py_ssize_t n = tree.get_n();
    const Py_ssize_t* perm = tree.get_perm().data();

    tree.boruvka(tree_dist, tree_ind);

    if (d_core) {
        const FLOAT* _d_core = tree.get_dcore();
        const FLOAT* _data = tree.get_data();
        // we need to recompute the distances as we applied a correction for ambiguity
        for (Py_ssize_t i=0; i<n-1; ++i) {
            tree_dist[i] = 0.0;
            for (Py_ssize_t j=0; j<D; ++j)
                tree_dist[i] += square(_data[tree_ind[2*i+0]*D+j]-_data[tree_ind[2*i+1]*D+j]);
            tree_dist[i] = max3(tree_dist[i], _d_core[tree_ind[2*i+0]], _d_core[tree_ind[2*i+1]]);
        }
        for (Py_ssize_t i=0; i<n; ++i)
            d_core[i] = _d_core[perm[i]];
    }

    for (Py_ssize_t i=0; i<n-1; ++i) {
        GENIECLUST_ASSERT(tree_ind[2*i+0] != tree_ind[2*i+1]);
        GENIECLUST_ASSERT(tree_ind[2*i+0] < n);
        GENIECLUST_ASSERT(tree_ind[2*i+1] < n);
        tree_ind[2*i+0] = perm[tree_ind[2*i+0]];
        tree_ind[2*i+1] = perm[tree_ind[2*i+1]];
    }


    if (order) {
        std::vector< CMstTriple<FLOAT> > mst(n-1);

        for (Py_ssize_t i=0; i<n-1; ++i) {
            mst[i] = CMstTriple<FLOAT>(tree_ind[2*i+0], tree_ind[2*i+1], tree_dist[i]);
        }

        std::sort(mst.begin(), mst.end());

        for (Py_ssize_t i=0; i<n-1; ++i) {
            tree_dist[i]    = mst[i].d;
            tree_ind[2*i+0] = mst[i].i1;  // i1 < i2
            tree_ind[2*i+1] = mst[i].i2;
        }
    }
}



};  // namespace

#endif
