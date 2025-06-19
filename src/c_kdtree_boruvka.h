/*  A Boruvka-type algorithm for finding minimum spanning trees
 *  wrt the Euclidean metric or the thereon-based mutual reachability distance.
 *
 *  It features many performance enhancements, has good locality of reference,
 *  supports multicore processing via OpenMP, etc.
 *
 *  The dual-tree Boruvka version (not used by default for it turned out slower
 *  than a more straightforward version) is based on "Fast Euclidean Minimum
 *  Spanning Tree: Algorithm, Analysis, and Applications"
 *  by W.B. March, P. Ram, A.G. Gray in ACM SIGKDD 2010.
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


#ifndef __c_kdtree_boruvka_h
#define __c_kdtree_boruvka_h

#include "c_common.h"
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
    FLOAT cluster_max_dist;   // for DTB, redundant otherwise (TODO: template them out)
    FLOAT min_dcore;          // for M>2, redundant otherwise (TODO: template them out)

    kdtree_node_clusterable() {
        left = nullptr;
        // right = nullptr;
        //cluster_repr = -1;
        //cluster_max_dist = INFINITY;
        //min_dcore = 0.0;
    }

    inline bool is_leaf() const {
        return left == nullptr /*&& right == nullptr*/; // either both null or none
    }
};




/** A class enabling searching for the nearest neighbour
 *  outside of the current point's cluster;
 *  it is thread-safe
 */
template <
    typename FLOAT,
    Py_ssize_t D,
    typename DISTANCE=kdtree_distance_sqeuclid<FLOAT,D>,
    typename NODE=kdtree_node_clusterable<FLOAT, D>
>
class kdtree_nearest_outsider
{
private:
    Py_ssize_t which;   ///< for which point are we getting the nns
    Py_ssize_t cluster; ///< the point's cluster
    const FLOAT* x;     ///< the point itself (shortcut)
    const FLOAT* data;  ///< the dataset
    const FLOAT* dcore; ///< the "core" distances
    FLOAT nn_dist;      ///< shortest distance
    Py_ssize_t nn_ind;  ///< index of the nn

    const Py_ssize_t* ds_par;  ///< points' cluster IDs (par[i]==ds.find(i)!)

    template <bool USE_DCORE>
    inline void point_vs_points(Py_ssize_t idx_from, Py_ssize_t idx_to)
    {
        const FLOAT* y = data+D*idx_from;
        for (Py_ssize_t j=idx_from; j<idx_to; ++j, y+=D)
        {
            if (cluster == ds_par[j]) continue;
            if (USE_DCORE && dcore[j] >= nn_dist) continue;

            FLOAT dij = DISTANCE::point_point(x, y);

            if (USE_DCORE) {
                // pulled-away from each other, but ordered w.r.t. the original pairwise distances (increasingly)
                FLOAT dcore_max = std::max(dcore[which], dcore[j]);
                if (dij <= dcore_max)
                    dij = dcore_max+dij/DCORE_DIST_ADJ;
            }

            if (dij < nn_dist) {
                nn_dist = dij;
                nn_ind = j;
            }
        }
    }


    template <bool USE_DCORE>
    void find_nn(const NODE* root)
    {
        kdtree_nearest_outsider_find_nn_start:  /* tail recursion elimination */

        if (root->cluster_repr == cluster) {
            // nothing to do - all are members of the x's cluster
            return;
        }

        if (USE_DCORE && nn_dist <= root->min_dcore) {
            // works for M<=2 as then min_dcore==0.0;
            // we have a better candidate already
            return;
        }

        if (root->is_leaf()/* || root->idx_to-root->idx_from <= max_brute_size*/) {
            if (which < root->idx_from || which >= root->idx_to)
                point_vs_points<USE_DCORE>(root->idx_from, root->idx_to);
            else {
                point_vs_points<USE_DCORE>(root->idx_from, which);
                point_vs_points<USE_DCORE>(which+1, root->idx_to);
            }
            return;
        }

        FLOAT dist_left  = DISTANCE::point_node(
            x, root->left->bbox_min.data(),  root->left->bbox_max.data()
        );
        FLOAT dist_right = DISTANCE::point_node(
            x, root->right->bbox_min.data(), root->right->bbox_max.data()
        );

        if (USE_DCORE) {
            if (dist_left < root->left->min_dcore)
                dist_left = root->left->min_dcore;
            if (dist_right < root->right->min_dcore)
                dist_right = root->right->min_dcore;
        }

        // nearer node first (significant speedup)
        if (dist_left < dist_right) {
            if (dist_left < nn_dist) {
                find_nn<USE_DCORE>(root->left);
                if (dist_right < nn_dist) {
                    //find_nn<USE_DCORE>(root->right);
                    root = root->right;
                    goto kdtree_nearest_outsider_find_nn_start;  // tail recursion elimination
                }
            }
        }
        else {
            if (dist_right < nn_dist) {
                find_nn<USE_DCORE>(root->right);
                if (dist_left < nn_dist) {
                    //find_nn<USE_DCORE>(root->left);
                    root = root->left;
                    goto kdtree_nearest_outsider_find_nn_start;  // tail recursion elimination
                }
            }
        }
    }


public:
    kdtree_nearest_outsider(
        const FLOAT* data,
        FLOAT* dcore,
        const Py_ssize_t which,
        const Py_ssize_t* ds_par
    ) :
        which(which), cluster(ds_par[which]), x(data+D*which), data(data), dcore(dcore), ds_par(ds_par)
    {

    }


    /**
     *  @param root
     *  @param nn_dist best nn_dist found so far for the current cluster
     */
    void find(const NODE* root, FLOAT nn_dist=INFINITY)
    {
        this->nn_dist = nn_dist;
        this->nn_ind  = which;

        if (dcore) find_nn<true>(root);
        else find_nn<false>(root);
    }


    inline FLOAT get_nn_dist() { return nn_dist; }
    inline Py_ssize_t get_nn_ind() { return nn_ind; }
};



template <
    typename FLOAT,
    Py_ssize_t D,
    typename DISTANCE=kdtree_distance_sqeuclid<FLOAT,D>,
    typename NODE=kdtree_node_clusterable<FLOAT, D>
>
class kdtree_boruvka : public kdtree<FLOAT, D, DISTANCE, NODE>
{
protected:
    FLOAT*  tree_dist;     ///< size n-1
    Py_ssize_t* tree_ind;  ///< size 2*(n-1)
    Py_ssize_t  tree_num;  /// number of MST edges already found
    CDisjointSets ds;

    std::vector<FLOAT>      nn_dist;
    std::vector<Py_ssize_t> nn_ind;
    std::vector<Py_ssize_t> nn_from;

    const Py_ssize_t first_pass_max_brute_size;  // used in the first iter (finding 1-nns)

    const bool use_dtb;

    const Py_ssize_t M;  // mutual reachability distance - "smoothing factor"
    std::vector<FLOAT> dcore;  // distances to the (M-1)-th nns of each point if M>1


    struct kdtree_node_orderer {
        NODE* nearer_node;
        NODE* farther_node;
        FLOAT nearer_dist;
        FLOAT farther_dist;

        kdtree_node_orderer(NODE* from, NODE* to1, NODE* to2, bool use_min_dcore=false)
        {
            nearer_dist  = DISTANCE::node_node(
                from->bbox_min.data(), from->bbox_max.data(),
                 to1->bbox_min.data(),  to1->bbox_max.data()
            );

            farther_dist = DISTANCE::node_node(
                from->bbox_min.data(), from->bbox_max.data(),
                 to2->bbox_min.data(),  to2->bbox_max.data()
            );

            if (use_min_dcore) {
                nearer_dist  = max3(nearer_dist, from->min_dcore, to1->min_dcore);
                farther_dist = max3(farther_dist, from->min_dcore, to2->min_dcore);
            }

            if (nearer_dist <= farther_dist) {
                nearer_node  = to1;
                farther_node = to2;
            }
            else {
                std::swap(nearer_dist, farther_dist);
                nearer_node  = to2;
                farther_node = to1;
            }
        }
    };


    template <bool USE_DCORE>
    inline void leaf_vs_leaf(NODE* roota, NODE* rootb)
    {
        // assumes ds.find(i) == ds.get_parent(i) for all i!
        const FLOAT* _x = this->data + roota->idx_from*D;
        for (Py_ssize_t i=roota->idx_from; i<roota->idx_to; ++i, _x += D)
        {
            Py_ssize_t ds_find_i = ds.get_parent(i);
            if (USE_DCORE && dcore[i] >= nn_dist[ds_find_i]) continue;

            for (Py_ssize_t j=rootb->idx_from; j<rootb->idx_to; ++j)
            {
                Py_ssize_t ds_find_j = ds.get_parent(j);
                if (ds_find_i == ds_find_j) continue;
                if (USE_DCORE && dcore[j] >= nn_dist[ds_find_i]) continue;

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


    void update_min_dcore()
    {
        GENIECLUST_ASSERT(M>2);

        // if (M <= 2) {
        //     for (Py_ssize_t i=(Py_ssize_t)this->nodes.size()-1; i>=0; --i)
        //     {
        //         NODE* curnode = &(this->nodes[i]);
        //         curnode->min_dcore = 0.0;
        //     }
        // }
        // else {
        // nodes is a deque...
        for (auto curnode = this->nodes.rbegin(); curnode != this->nodes.rend(); ++curnode)
        {
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
        // }
    }


    void update_cluster_data()
    {
        for (Py_ssize_t i=0; i<this->n; ++i)
            this->ds.find(i);
        // now ds.find(i) == ds.get_parent(i) for all i

        // nodes is a deque...
        for (auto curnode = this->nodes.rbegin(); curnode != this->nodes.rend(); ++curnode)
        {
            curnode->cluster_max_dist = INFINITY;

            if (curnode->cluster_repr >= 0) {
                curnode->cluster_repr = ds.get_parent(curnode->cluster_repr);
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
                if (curnode->left->cluster_repr >= 0) {
                    // if both children only feature members of the same cluster, update the cluster repr for the current node;
                    if (curnode->left->cluster_repr == curnode->right->cluster_repr)
                        curnode->cluster_repr = curnode->left->cluster_repr;
                }
                // else curnode->cluster_repr = -1;  // it already is
            }
        }
    }


    void find_mst_first_1()
    {
        GENIECLUST_ASSERT(M <= 2);
        const Py_ssize_t k = 1;


        #if OPENMP_IS_ENABLED
        int omp_nthreads = Comp_get_max_threads();
        #else
        int omp_nthreads = 1;
        #endif

        for (Py_ssize_t i=0; i<this->n; ++i) nn_dist[i] = INFINITY;
        for (Py_ssize_t i=0; i<this->n; ++i) nn_ind[i] = -1;

        // find 1-nns of each point using max_brute_size,
        // preferably with max_brute_size>max_leaf_size
        #if OPENMP_IS_ENABLED
        #pragma omp parallel for schedule(static)
        #endif
        for (Py_ssize_t i=0; i<this->n; ++i) {
            kdtree_kneighbours<FLOAT, D, DISTANCE, NODE> nn(
                this->data, nullptr, i, &nn_dist[i], &nn_ind[i], k,
                first_pass_max_brute_size
            );
            nn.find(&this->nodes[0], false);

            if (omp_nthreads == 1 && nn_dist[i] < nn_dist[nn_ind[i]]) {
                nn_dist[nn_ind[i]] = nn_dist[i];
                nn_ind[nn_ind[i]] = i;
            }

            if (M == 2) dcore[i] = nn_dist[i];
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
        std::vector<FLOAT> knn_dist(this->n*k, INFINITY);
        std::vector<Py_ssize_t> knn_ind(this->n*k, -1);

        #if OPENMP_IS_ENABLED
        #pragma omp parallel for schedule(static)
        #endif
        for (Py_ssize_t i=0; i<this->n; ++i) {
            kdtree_kneighbours<FLOAT, D, DISTANCE, NODE> nn(
                this->data, nullptr, i, knn_dist.data()+k*i, knn_ind.data()+k*i, k,
                first_pass_max_brute_size
            );
            nn.find(&this->nodes[0], false);
            dcore[i] = knn_dist[i*k+(k-1)];
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


    void find_mst_next_dtb(NODE* roota, NODE* rootb)
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
                // nearer node first -> faster!
                kdtree_node_orderer sel(roota, rootb->left, rootb->right, (M>2));

                // prune nodes too far away if we have better candidates
                if (roota->cluster_max_dist >= sel.nearer_dist) {
                    find_mst_next_dtb(roota, sel.nearer_node);
                    if (roota->cluster_max_dist >= sel.farther_dist)
                        find_mst_next_dtb(roota, sel.farther_node);
                }


                // roota->cluster_max_dist updated above
            }
        }
        else {  // roota is not a leaf
            if (rootb->is_leaf()) {
                kdtree_node_orderer sel(rootb, roota->left, roota->right, (M>2));
                if (sel.nearer_node->cluster_max_dist >= sel.nearer_dist)
                    find_mst_next_dtb(sel.nearer_node, rootb);
                if (sel.farther_node->cluster_max_dist >= sel.farther_dist)  // separate if!
                    find_mst_next_dtb(sel.farther_node, rootb);
            }
            else {
                kdtree_node_orderer sel(roota->left, rootb->left, rootb->right, (M>2));
                if (roota->left->cluster_max_dist >= sel.nearer_dist) {
                    find_mst_next_dtb(roota->left, sel.nearer_node);
                    if (roota->left->cluster_max_dist >= sel.farther_dist)
                        find_mst_next_dtb(roota->left, sel.farther_node);
                }

                sel = kdtree_node_orderer(roota->right, rootb->left, rootb->right, (M>2));
                if (roota->right->cluster_max_dist >= sel.nearer_dist) {
                    find_mst_next_dtb(roota->right, sel.nearer_node);
                    if (roota->right->cluster_max_dist >= sel.farther_dist)
                        find_mst_next_dtb(roota->right, sel.farther_node);
                }
            }

            roota->cluster_max_dist = std::max(
                roota->left->cluster_max_dist,
                roota->right->cluster_max_dist
            );
        }
    }



    void find_mst_next_nn()
    {
        // find the point from another cluster that is closest to the i-th point
        #if OPENMP_IS_ENABLED
        omp_lock_t nn_dist_lock;
        int omp_nthreads = Comp_get_max_threads();
        if (omp_nthreads > 1) omp_init_lock(&nn_dist_lock);
        #else
        int omp_nthreads = 1;
        #endif

        #if OPENMP_IS_ENABLED
        #pragma omp parallel for schedule(static)
        #endif
        for (Py_ssize_t i=0; i<this->n; ++i) {
            Py_ssize_t ds_find_i = ds.get_parent(i);

            // NOTE: assumption: no race condition/atomic read...
            FLOAT nn_dist_best = nn_dist[ds_find_i];

            if (M > 2 && nn_dist_best < dcore[i]) continue;

            kdtree_nearest_outsider<FLOAT, D, DISTANCE, NODE> nn(
                this->data, (M>2)?this->dcore.data():nullptr,
                i, ds.get_parents()
            );
            nn.find(&this->nodes[0], nn_dist_best);

            FLOAT nn_dist_cur = nn.get_nn_dist();

            #if OPENMP_IS_ENABLED
            if (nn_dist_cur < nn_dist_best)
            #endif
            {
            #if OPENMP_IS_ENABLED
                if (omp_nthreads > 1) omp_set_lock(&nn_dist_lock);
            #endif
                if (nn_dist_cur < nn_dist[ds_find_i]) {
                    Py_ssize_t nn_ind_cur = nn.get_nn_ind();
                    GENIECLUST_ASSERT(nn_ind_cur != i);
                    nn_dist[ds_find_i] = nn_dist_cur;
                    nn_ind[ds_find_i]  = nn_ind_cur;
                    nn_from[ds_find_i] = i;
                }
                #if OPENMP_IS_ENABLED
                if (omp_nthreads > 1) omp_unset_lock(&nn_dist_lock);
                #endif
            }

            if (omp_nthreads == 1) {
                Py_ssize_t nn_ind_cur = nn.get_nn_ind();
                Py_ssize_t ds_find_j = ds.get_parent(nn_ind_cur);
                if (nn_dist_cur < nn_dist[ds_find_j]) {
                    nn_dist[ds_find_j] = nn_dist_cur;
                    nn_ind[ds_find_j]  = i;
                    nn_from[ds_find_j] = nn_ind_cur;
                }
            }
        }

        #if OPENMP_IS_ENABLED
        if (omp_nthreads > 1) omp_destroy_lock(&nn_dist_lock);
        #endif
    }


    void find_mst()
    {
        GENIECLUST_PROFILER_USE

        GENIECLUST_PROFILER_START
        // the 1st iteration: connect nearest neighbours with each other
        find_mst_first();
        GENIECLUST_PROFILER_STOP("find_mst_first")

        GENIECLUST_PROFILER_START
        if (M>2) update_min_dcore();
        GENIECLUST_PROFILER_STOP("update_min_dcore")

        std::vector<Py_ssize_t> ds_parents(this->n);
        Py_ssize_t ds_k;

        Py_ssize_t _iter = 0;
        while (tree_num < this->n-1) {
            #if GENIECLUST_R
            Rcpp::checkUserInterrupt();  // throws an exception, not a longjmp
            #elif GENIECLUST_PYTHON
            if (PyErr_CheckSignals() != 0) throw std::runtime_error("signal caught");
            #endif

            _iter++;
            GENIECLUST_PROFILER_START

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

            if (this->use_dtb)
                find_mst_next_dtb(&this->nodes[0], &this->nodes[0]);
            else
                find_mst_next_nn();

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

            GENIECLUST_PROFILER_STOP("find_mst iter #%d", (int)_iter)
        }
    }


public:
    kdtree_boruvka()
        : kdtree<FLOAT, D, DISTANCE, NODE>()
    {

    }


    kdtree_boruvka(
        FLOAT* data, const Py_ssize_t n, const Py_ssize_t M=1,
        const Py_ssize_t max_leaf_size=4,
        const Py_ssize_t first_pass_max_brute_size=16,
        const bool use_dtb=false
    ) :
        kdtree<FLOAT, D, DISTANCE, NODE>(data, n, max_leaf_size), tree_num(0),
        ds(n), nn_dist(n), nn_ind(n), nn_from(n),
        first_pass_max_brute_size(first_pass_max_brute_size), use_dtb(use_dtb), M(M)
    {
        if (M >= 2) dcore.resize(n);  // M==2 too!
    }



    void mst(FLOAT* tree_dist, Py_ssize_t* tree_ind)
    {
        this->tree_dist = tree_dist;
        this->tree_ind = tree_ind;

        if (ds.get_k() != (Py_ssize_t)this->n) ds.reset();
        tree_num = 0;

        for (Py_ssize_t i=0; i<this->n-1; ++i)     tree_dist[i] = INFINITY;
        for (Py_ssize_t i=0; i<2*(this->n-1); ++i) tree_ind[i]  = this->n;

        // nodes is a deque...
        for (auto curnode = this->nodes.rbegin(); curnode != this->nodes.rend(); ++curnode)
            curnode->cluster_repr = -1;

        find_mst();
    }


    inline const FLOAT* get_dcore() const { return this->dcore.data(); }
};




template <typename FLOAT, Py_ssize_t D, typename TREE>
void mst(
    TREE& tree,
    FLOAT* tree_dist,       // size n-1
    Py_ssize_t* tree_ind,   // size 2*(n-1),
    FLOAT* d_core=nullptr   // size n
) {
    Py_ssize_t n = tree.get_n();
    const Py_ssize_t* perm = tree.get_perm().data();

    tree.mst(tree_dist, tree_ind);

    if (d_core) {
        const FLOAT* _d_core = tree.get_dcore();
        const FLOAT* _data = tree.get_data();
        // we need to recompute the distances as we applied a correction for ambiguity
        for (Py_ssize_t i=0; i<n-1; ++i) {
            // TODO: DISTANCE::point_point....
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

    // the edges are not ordered, use Cmst_order
}



};  // namespace

#endif
