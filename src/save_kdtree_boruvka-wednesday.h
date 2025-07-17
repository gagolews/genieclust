/*  Borůvka-type algorithms for finding minimum spanning trees
 *  wrt the Euclidean metric or the thereon-based mutual reachability distance.
 *
 *  The dual-tree Borůvka version is, in principle, based on
 *  "Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis,
 *  and Applications" by W.B. March, P. Ram, A.G. Gray published
 *  in ACM SIGKDD 2010.  As far as our implementation
 *  is concerned, the dual-tree approach is only faster in 2- and
 *  3-dimensional spaces, for M <= 2, and in a single-threaded setting.
 *
 *  The single-tree version (iteratively find each point's nearest neighbour
 *  outside its own cluster, i.e., nearest alien) is naively parallelisable.
 *
 *  For more details on our implementation of K-d trees, see
 *  the source file defining the base class.
 *
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
#include <omp.h>




namespace quitefastkdtree {

template <typename FLOAT, Py_ssize_t D>
struct kdtree_node_clusterable : public kdtree_node_base<FLOAT, D>
{
    kdtree_node_clusterable* left;
    kdtree_node_clusterable* right;

    struct t_dtb_data { FLOAT max_ncl_dist; };
    struct t_qtb_data { FLOAT lastnn_dist; Py_ssize_t lastnn_ind; Py_ssize_t lastnn_from; };

    Py_ssize_t cluster_repr;  //< representative point index if all descendants are in the same cluster, -1 otherwise
    FLOAT min_dcore;          // for M>2, redundant otherwise (TODO: template them out)

    union {
        t_dtb_data dtb_data;
        t_qtb_data qtb_data;
    };

    kdtree_node_clusterable() {
        left = nullptr;
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
struct kdtree_node_orderer {
    NODE* nearer_node;
    NODE* farther_node;
    FLOAT nearer_dist;
    FLOAT farther_dist;

    kdtree_node_orderer(const FLOAT* x, NODE* to1, NODE* to2, bool use_min_dcore=false)
    {
        nearer_dist  = DISTANCE::point_node(
            x, to1->bbox_min.data(),  to1->bbox_max.data()
        );

        farther_dist = DISTANCE::point_node(
            x, to2->bbox_min.data(),  to2->bbox_max.data()
        );

        if (use_min_dcore) {
            if (nearer_dist < to1->min_dcore)
                nearer_dist = to1->min_dcore;
            if (farther_dist < to2->min_dcore)
                farther_dist = to2->min_dcore;
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




/** A class enabling searching for the nearest neighbour
 *  outside of the current point's cluster;
 *  (for the "single-tree" Borůvka algo); it is thread-safe
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
    Py_ssize_t M;

    const Py_ssize_t* ds_par;  ///< points' cluster IDs (par[i]==ds.find(i)!)

    template <bool USE_DCORE>
    inline void point_vs_points(Py_ssize_t idx_from, Py_ssize_t idx_to)
    {
        const FLOAT* y = data+D*idx_from;
        for (Py_ssize_t j=idx_from; j<idx_to; ++j, y+=D) {
            if (cluster == ds_par[j]) continue;

            if (USE_DCORE && dcore[j] >= nn_dist) continue;
            FLOAT dd = DISTANCE::point_point(x, y);
            if (USE_DCORE) dd = max3(dd, dcore[which], dcore[j]);
            if (dd < nn_dist) {
                nn_dist = dd;
                nn_ind = j;
            }
        }
    }


    template <bool USE_DCORE>
    void find_nn(const NODE* root)
    {
        if (root->cluster_repr == cluster) {
            // nothing to do - all are members of the x's cluster
            return;
        }

        if (USE_DCORE && nn_dist <= root->min_dcore) {
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


        kdtree_node_orderer<FLOAT, D, DISTANCE, NODE> sel(
            x, root->left, root->right, USE_DCORE
        );

        if (sel.nearer_dist < nn_dist) {
            find_nn<USE_DCORE>(sel.nearer_node);

            if (sel.farther_dist < nn_dist)
                find_nn<USE_DCORE>(sel.farther_node);
        }
    }


public:
    kdtree_nearest_outsider(
        const FLOAT* data,
        FLOAT* dcore,
        const Py_ssize_t which,
        const Py_ssize_t* ds_par,
        Py_ssize_t M
    ) :
        which(which), cluster(ds_par[which]), x(data+D*which), data(data),
        dcore(dcore), M(M), ds_par(ds_par)
    {

    }


    /**
     *  @param root
     *  @param nn_dist best nn_dist found so far for the current cluster
     */
    void find(const NODE* root, FLOAT nn_dist=INFINITY)
    {
        this->nn_dist = nn_dist;
        this->nn_ind  = -1;

        if (M>2) find_nn<true>(root);
        else find_nn<false>(root);
    }


    inline FLOAT get_nn_dist() { return nn_dist; }
    inline Py_ssize_t get_nn_ind() { return nn_ind; }
};






/** A class enabling searching for the nearest neighbour
 *  outside of the current point's cluster;
 *  (for the "sesqui-tree" Borůvka algo); it is thread-safe
 */
template <
    typename FLOAT,
    Py_ssize_t D,
    typename DISTANCE=kdtree_distance_sqeuclid<FLOAT,D>,
    typename NODE=kdtree_node_clusterable<FLOAT, D>
>
class kdtree_nearest_outsider_multi
{
private:
    NODE* curleaf;
    const FLOAT* data;      ///< the dataset
    const FLOAT* dcore;     ///< the "core" distances
    Py_ssize_t M;

    const Py_ssize_t* ds_par;  ///< points' cluster IDs (par[i]==ds.find(i)!)

    FLOAT nn_dist;          ///< shortest distance
    Py_ssize_t nn_ind;      ///< index of the nn
    Py_ssize_t nn_from;


    template <bool USE_DCORE>
    inline void points_vs_points(Py_ssize_t idx_from, Py_ssize_t idx_to)
    {
        const FLOAT* x = data+D*curleaf->idx_from;
        for (Py_ssize_t which=curleaf->idx_from; which<curleaf->idx_to; ++which, x+=D) {
            if (USE_DCORE && dcore[which] >= nn_dist) continue;

            const FLOAT* y = data+D*idx_from;
            for (Py_ssize_t j=idx_from; j<idx_to; ++j, y+=D) {
                if (curleaf->cluster_repr == ds_par[j]) continue;
                if (USE_DCORE && dcore[j] >= nn_dist) continue;

                FLOAT dd = DISTANCE::point_point(x, y);
                if (USE_DCORE) dd = max3(dd, dcore[which], dcore[j]);
                if (dd < nn_dist) {
                    nn_dist = dd;
                    nn_ind = j;
                    nn_from = which;
                }
            }
        }
    }


    template <bool USE_DCORE>
    void find_nn(const NODE* root)
    {
        if (root->cluster_repr == curleaf->cluster_repr) {
            // nothing to do - all are members of the x's cluster
            return;
        }

        if (USE_DCORE && nn_dist <= root->min_dcore) {
            // we have a better candidate already
            return;
        }

        if (root->is_leaf()) {
            points_vs_points<USE_DCORE>(root->idx_from, root->idx_to);
            return;
        }


        kdtree_node_orderer<FLOAT, D, DISTANCE, NODE> sel(
            curleaf, root->left, root->right, USE_DCORE
        );

        if (sel.nearer_dist < nn_dist) {
            find_nn<USE_DCORE>(sel.nearer_node);

            if (sel.farther_dist < nn_dist)
                find_nn<USE_DCORE>(sel.farther_node);
        }
    }


public:
    kdtree_nearest_outsider_multi(
        const FLOAT* data,
        FLOAT* dcore,
        NODE* curleaf,
        const Py_ssize_t* ds_par,
        Py_ssize_t M
    ) :
        curleaf(curleaf), data(data),
        dcore(dcore), M(M), ds_par(ds_par)
    {
        ;
    }


    /**
     *  @param root
     *  @param nn_dist best nn_dist found so far for the current cluster
     */
    void find(const NODE* root, FLOAT nn_dist=INFINITY)
    {
        this->nn_dist = nn_dist;
        this->nn_ind  = -1;
        this->nn_from = -1;

        if (M>2) find_nn<true>(root);
        else find_nn<false>(root);
    }


    inline FLOAT get_nn_dist()      { return nn_dist; }
    inline Py_ssize_t get_nn_ind()  { return nn_ind; }
    inline Py_ssize_t get_nn_from() { return nn_from; }
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
    FLOAT*  tree_dist;       ///< size n-1
    Py_ssize_t* tree_ind;    ///< size 2*(n-1)
    Py_ssize_t  tree_edges;  /// number of MST edges already found
    Py_ssize_t  tree_iter;   /// iteration of the algorithm
    CDisjointSets ds;

    std::vector<FLOAT>      ncl_dist;  // ncl_dist[find(i)] - distance to the i-th cluster's nearest cluster
    std::vector<Py_ssize_t> ncl_ind;   // ncl_ind[find(i)]  - index of i's nn
    std::vector<Py_ssize_t> ncl_from;  // ncl_from[find(i)] - the relevant member of i

    const Py_ssize_t first_pass_max_brute_size;  // used in the first iter (finding 1-nns)

    enum BORUVKA_TYPE { BORUVKA_STB, BORUVKA_QTB, BORUVKA_DTB };
    BORUVKA_TYPE boruvka_variant;
    bool reset_nns;

    const FLOAT mutreach_adj;  // M>2 only

    std::vector<FLOAT> lastnn_dist;   // !use_dtb only
    std::vector<Py_ssize_t> lastnn_ind;   // !use_dtb only

    std::vector<Py_ssize_t> ds_parents;

    const Py_ssize_t M;              // mutual reachability distance - "smoothing factor"
    std::vector<FLOAT> dcore;        // distances to the (M-1)-th nns of each point if M>1 or 1-NN for M==1
    std::vector<FLOAT> Mnn_dist;     // M-1 nearest neighbours of each point if M>1
    std::vector<Py_ssize_t> Mnn_ind;

    omp_lock_t omp_lock;
    int omp_nthreads;


    std::vector<NODE*> leaves; // TODO: sesquitree only




    void setup_leaves()  // TODO: sesquitree only
    {
        // NOTE: nleaves can be determined whilst building the tree
        Py_ssize_t nleaves = 0;
        for (auto curnode = this->nodes.begin(); curnode != this->nodes.end(); ++curnode)
            if (curnode->is_leaf()) nleaves++;

        leaves.resize(nleaves);

        Py_ssize_t _leafnum = 0;

        for (auto curnode = this->nodes.begin(); curnode != this->nodes.end(); ++curnode) {
            if (!curnode->is_leaf()) continue;
                leaves[_leafnum++] = &(*curnode);

            curnode->qtb_data.lastnn_dist = INFINITY;
            curnode->qtb_data.lastnn_ind  = -1;
            curnode->qtb_data.lastnn_from = -1;
        }

        GENIECLUST_ASSERT(_leafnum == nleaves);
    }


    void update_min_dcore()
    {
        GENIECLUST_ASSERT(M>=2);

        for (auto curnode = this->nodes.rbegin(); curnode != this->nodes.rend(); ++curnode)
        {
            if (curnode->is_leaf()) {
                curnode->min_dcore = dcore[curnode->idx_from];
                for (Py_ssize_t i=curnode->idx_from+1; i<curnode->idx_to; ++i) {
                    if (dcore[i] < curnode->min_dcore)
                        curnode->min_dcore = dcore[i];
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
        // 1. ensure that ds.find(i) == ds.get_parent(i) for all i
        // 2. reset ncl_dist
        // 3. setup ds_parents
        Py_ssize_t ds_k = 0;
        for (Py_ssize_t i=0; i<this->n; ++i) {
            // ds.find(i) - not thread safe
            if (i == this->ds.find(i)) {  // the parent is always first!
                ncl_dist[i] = INFINITY;
                ncl_ind[i]  = -1;
                ncl_from[i] = -1;
                ds_parents[ds_k++] = i;
            }
        }
        GENIECLUST_ASSERT(ds_k == ds.get_k());

        if (boruvka_variant != BORUVKA_DTB && tree_iter > 1) {
            // BORUVKA_DTB cannot update lastnn data

            // tree_iter == 1 -> all lastnn_ind == -1;

            // 1. invalidate lastnn_ind if they now point to the same cluster
            // 2. try to get a better upper bound for ncl_dist based on past NN data
            for (Py_ssize_t i=0; i<this->n; ++i) {
                if (lastnn_ind[i] < 0) continue;

                Py_ssize_t ds_find_i = ds.get_parent(i);
                Py_ssize_t ds_find_j = ds.get_parent(lastnn_ind[i]);

                if (ds_find_i == ds_find_j) {
                    lastnn_ind[i] = -1;
                    continue;
                }

                if (ncl_dist[ds_find_i] > lastnn_dist[i]) {
                    ncl_dist[ds_find_i] = lastnn_dist[i];
                    ncl_ind[ds_find_i]  = lastnn_ind[i];
                    ncl_from[ds_find_i] = i;
                }

                if (ncl_dist[ds_find_j] > lastnn_dist[i]) {
                    ncl_dist[ds_find_j] = lastnn_dist[i];
                    ncl_ind[ds_find_j]  = i;
                    ncl_from[ds_find_j] = lastnn_ind[i];
                }
            }
        }

        if (boruvka_variant == BORUVKA_DTB && M > 2) {
            // reuse M-1 NNs if d==dcore[i] as an initialiser to nn_ind/dist/from;
            // good speed-up sometimes (we'll be happy with any match; leaves
            // are formed in the 1st iteration of the algorithm)
            for (Py_ssize_t i=0; i<this->n; ++i) {
                if (lastnn_ind[i] <= -M) continue;

                Py_ssize_t ds_find_i = ds.get_parent(i);
                if (ncl_dist[ds_find_i] <= dcore[i]) continue;

                Py_ssize_t v = -lastnn_ind[i]-1; // -1→0, -2→1, ..., -M→M-1
                GENIECLUST_ASSERT(v >= 0);
                for (; v<M-1; ++v)
                {
                    Py_ssize_t j = Mnn_ind[i*(M-1)+((mutreach_adj<0.0)?((M-1)-1-v):(v))];
                    if (ds_find_i == ds.get_parent(j) || dcore[i] < dcore[j]) continue;

                    ncl_dist[ds_find_i] = dcore[i];
                    ncl_ind[ds_find_i]  = j;
                    ncl_from[ds_find_i] = i;

                    Py_ssize_t ds_find_j = ds.get_parent(j);
                    if (ncl_dist[ds_find_j] > dcore[i]) {
                        ncl_dist[ds_find_j] = dcore[i];
                        ncl_ind[ds_find_j]  = i;
                        ncl_from[ds_find_j] = j;
                    }

                    break;  // other candidates have d_M >= dcore[i] anyway
                }

                lastnn_ind[i] = -(v+1);
            }
        }
    }


    void update_node_data()
    {
        // We have ds.find(i) == ds.get_parent(i) for all i

        // nodes is a deque...
        for (auto curnode = this->nodes.rbegin(); curnode != this->nodes.rend(); ++curnode)
        {
            if (boruvka_variant == BORUVKA_DTB) {
                curnode->dtb_data.max_ncl_dist = INFINITY;
            }

            if (curnode->cluster_repr >= 0) {
                curnode->cluster_repr = ds.get_parent(curnode->cluster_repr);
                continue;
            }

            if (curnode->is_leaf()) {
                curnode->cluster_repr = ds.get_parent(curnode->idx_from);
                for (Py_ssize_t i=curnode->idx_from+1; i<curnode->idx_to; ++i) {
                    if (curnode->cluster_repr != ds.get_parent(i)) {
                        curnode->cluster_repr = -1;  // not all are members of the same cluster
                        break;
                    }
                }

                if (curnode->cluster_repr >= 0 && boruvka_variant == BORUVKA_QTB)
                {
                    curnode->qtb_data.lastnn_dist = lastnn_dist[curnode->idx_from];
                    curnode->qtb_data.lastnn_ind  = lastnn_ind[curnode->idx_from];  // can be < 0
                    curnode->qtb_data.lastnn_from = curnode->idx_from;
                    for (Py_ssize_t i=curnode->idx_from+1; i<curnode->idx_to; ++i) {
                        if (curnode->qtb_data.lastnn_dist > lastnn_dist[i]) {
                            curnode->qtb_data.lastnn_dist = lastnn_dist[i];
                            curnode->qtb_data.lastnn_ind  = lastnn_ind[i];
                            curnode->qtb_data.lastnn_from = i;
                        }
                    }
                    // now curleaf->qtb_data.lastnn_dist is max of lastnn_dist
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

        for (Py_ssize_t i=0; i<this->n; ++i) ncl_dist[i] = INFINITY;
        for (Py_ssize_t i=0; i<this->n; ++i) ncl_ind[i] = -1;

        // find 1-nns of each point using max_brute_size,
        // preferably with max_brute_size>max_leaf_size
        #if OPENMP_IS_ENABLED
        #pragma omp parallel for schedule(static)
        #endif
        for (Py_ssize_t i=0; i<this->n; ++i) {
            kdtree_kneighbours<FLOAT, D, DISTANCE, NODE> nn(
                this->data, nullptr, i, &ncl_dist[i], &ncl_ind[i], k,
                first_pass_max_brute_size
            );
            nn.find(&this->nodes[0], /*reset=*/false);

            if (omp_nthreads == 1 && ncl_dist[i] < ncl_dist[ncl_ind[i]]) {
                // the speed up is rather small...
                ncl_dist[ncl_ind[i]] = ncl_dist[i];  // merely an upper bound...
                ncl_ind[ncl_ind[i]]  = i;
            }

            lastnn_dist[i] = ncl_dist[i];
            lastnn_ind[i] = -1;  // important: all lastnns invalidated

            if (M > 1) {
                dcore[i]    = ncl_dist[i];
                Mnn_dist[i] = ncl_dist[i];
                Mnn_ind[i]  = ncl_ind[i];
            }

        }

        // connect nearest neighbours with each other
        for (Py_ssize_t i=0; i<this->n; ++i) {
            if (ds.find(i) != ds.find(ncl_ind[i])) {
                tree_ind[tree_edges*2+0] = i;
                tree_ind[tree_edges*2+1] = ncl_ind[i];
                tree_dist[tree_edges] = ncl_dist[i];
                ds.merge(i, ncl_ind[i]);
                tree_edges++;
            }
        }
    }


    void find_mst_first_M()
    {
        GENIECLUST_ASSERT(M>1);
        const Py_ssize_t k = M-1;
        // find (M-1)-nns of each point

        for (size_t i=0; i<Mnn_dist.size(); ++i) Mnn_dist[i] = INFINITY;
        for (size_t i=0; i<Mnn_ind.size(); ++i)  Mnn_ind[i] = -1;

        #if OPENMP_IS_ENABLED
        #pragma omp parallel for schedule(static)
        #endif
        for (Py_ssize_t i=0; i<this->n; ++i) {
            kdtree_kneighbours<FLOAT, D, DISTANCE, NODE> nn(
                this->data, nullptr, i, Mnn_dist.data()+k*i, Mnn_ind.data()+k*i, k,
                first_pass_max_brute_size
            );
            nn.find(&this->nodes[0], /*reset=*/false);
            dcore[i] = Mnn_dist[i*k+(k-1)];  // distance to the (M-1)-th NN

            lastnn_dist[i] = dcore[i];  // merely a lower bound
            lastnn_ind[i] = -1;  // important: all lastnns invalidated

        }


        // k-nns wrt Euclidean distances are not necessarily k-nns wrt M-mutreach
        // k-nns have d_M >= d_core

        // dcore[i] is definitely the smallest possible d_M(i, *); i!=*
        // we can only be sure that j is a NN if d_M(i, j) == dcore[i]

        // but NNs wrt d_m might be ambiguous - we might want to pick,
        // e.g., the farthest or the closest one wrt the original dist

        // the correction for ambiguity is only applied at this stage!

        #define MERGE_I_J(_i, _j) { \
            tree_ind[tree_edges*2+0] = _i; \
            tree_ind[tree_edges*2+1] = _j; \
            tree_dist[tree_edges] = dcore[_i]; \
            ds.merge(_i, _j); \
            tree_edges++; \
        }

        if (mutreach_adj <= -1 || mutreach_adj >= 1) {
            for (Py_ssize_t i=0; i<this->n; ++i) {
                // mutreach_adj <= -1 - connect with j whose dcore[j] is the smallest
                // mutreach_adj >=  1 - connect with j whose dcore[j] is the largest

                Py_ssize_t bestj = -1;
                FLOAT bestdcorej = (mutreach_adj <= -1)?INFINITY:(-INFINITY);
                for (Py_ssize_t v=0; v<k; ++v)
                {
                    Py_ssize_t j = Mnn_ind[i*k+v];
                    if (dcore[i] >= dcore[j] && ds.find(i) != ds.find(j)) {
                        if (
                            (mutreach_adj <= -1 && bestdcorej >= dcore[j]) ||
                            (mutreach_adj >=  1 && bestdcorej <  dcore[j])
                        ) {
                            bestj = j;
                            bestdcorej = dcore[j];
                        }
                    }
                }
                if (bestj >= 0) MERGE_I_J(i, bestj);
            }
        }
        else {
            for (Py_ssize_t i=0; i<this->n; ++i) {
                // connect with j whose d(i,j) is the smallest (1>mutreach_adj>0) or largest (-1<mutreach_adj<0)
                // stops searching early, because the original distances are sorted

                for (Py_ssize_t v=0; v<k; ++v)
                {
                    Py_ssize_t j = Mnn_ind[i*k+((mutreach_adj<0.0)?(k-1-v):(v))];
                    if (dcore[i] >= dcore[j] && ds.find(i) != ds.find(j)) {
                        // j is the nearest neighbour of i wrt mutreach dist.
                        MERGE_I_J(i, j);
                        break;  // other candidates have d_M >= dcore[i] anyway
                    }
                }
            }
        }
    }


    void find_mst_first()
    {
        // the 1st iteration: connect nearest neighbours with each other
        if (M <= 2) find_mst_first_1();
        else        find_mst_first_M();
    }





    template <bool USE_DCORE>
    inline void leaf_vs_leaf_dtb(NODE* roota, NODE* rootb)
    {
        // assumes ds.find(i) == ds.get_parent(i) for all i!
        const FLOAT* _x = this->data + roota->idx_from*D;
        for (Py_ssize_t i=roota->idx_from; i<roota->idx_to; ++i, _x += D)
        {
            Py_ssize_t ds_find_i = ds.get_parent(i);
            if (USE_DCORE && dcore[i] >= ncl_dist[ds_find_i]) continue;

            for (Py_ssize_t j=rootb->idx_from; j<rootb->idx_to; ++j)
            {
                Py_ssize_t ds_find_j = ds.get_parent(j);
                if (ds_find_i == ds_find_j) continue;
                if (USE_DCORE && dcore[j] >= ncl_dist[ds_find_i]) continue;

                FLOAT dij = DISTANCE::point_point(_x, this->data+j*D);

                if (USE_DCORE) dij = max3(dij, dcore[i], dcore[j]);

                if (dij < ncl_dist[ds_find_i]) {
                    ncl_dist[ds_find_i] = dij;
                    ncl_ind[ds_find_i]  = j;
                    ncl_from[ds_find_i] = i;
                }
            }
        }
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
        //if (roota->dtb_data.max_ncl_dist < dist) {
        //    // we've a better candidate already - nothing to do
        //    return;
        //}

        if (roota->is_leaf()) {
            if (rootb->is_leaf()) {

                if (M>2) leaf_vs_leaf_dtb<true>(roota, rootb);
                else     leaf_vs_leaf_dtb<false>(roota, rootb);

                if (roota->cluster_repr >= 0) {  // all points are in the same cluster
                    roota->dtb_data.max_ncl_dist = ncl_dist[roota->cluster_repr];
                }
                else {
                    // max_ncl_dist = max(nn_dist[ds_parent(points in roota)])
                    roota->dtb_data.max_ncl_dist = ncl_dist[ds.get_parent(roota->idx_from)];
                    for (Py_ssize_t i=roota->idx_from+1; i<roota->idx_to; ++i) {
                        FLOAT dist_cur = ncl_dist[ds.get_parent(i)];
                        if (dist_cur > roota->dtb_data.max_ncl_dist)
                            roota->dtb_data.max_ncl_dist = dist_cur;
                    }
                }
            }
            else {
                // nearer node first -> faster!
                kdtree_node_orderer<FLOAT, D, DISTANCE, NODE> sel(roota, rootb->left, rootb->right, (M>2));

                // prune nodes too far away if we have better candidates
                if (roota->dtb_data.max_ncl_dist > sel.nearer_dist) {
                    find_mst_next_dtb(roota, sel.nearer_node);
                    if (roota->dtb_data.max_ncl_dist > sel.farther_dist)
                        find_mst_next_dtb(roota, sel.farther_node);
                }


                // roota->dtb_data.max_ncl_dist updated above
            }
        }
        else {  // roota is not a leaf
            if (rootb->is_leaf()) {
                kdtree_node_orderer<FLOAT, D, DISTANCE, NODE> sel(rootb, roota->left, roota->right, (M>2));
                if (sel.nearer_node->dtb_data.max_ncl_dist > sel.nearer_dist)
                    find_mst_next_dtb(sel.nearer_node, rootb);
                if (sel.farther_node->dtb_data.max_ncl_dist > sel.farther_dist)  // separate if!
                    find_mst_next_dtb(sel.farther_node, rootb);
            }
            else {
                kdtree_node_orderer<FLOAT, D, DISTANCE, NODE> sel(roota->left, rootb->left, rootb->right, (M>2));
                if (roota->left->dtb_data.max_ncl_dist > sel.nearer_dist) {
                    find_mst_next_dtb(roota->left, sel.nearer_node);
                    if (roota->left->dtb_data.max_ncl_dist > sel.farther_dist)
                        find_mst_next_dtb(roota->left, sel.farther_node);
                }

                sel = kdtree_node_orderer<FLOAT, D, DISTANCE, NODE>(roota->right, rootb->left, rootb->right, (M>2));
                if (roota->right->dtb_data.max_ncl_dist > sel.nearer_dist) {
                    find_mst_next_dtb(roota->right, sel.nearer_node);
                    if (roota->right->dtb_data.max_ncl_dist > sel.farther_dist)
                        find_mst_next_dtb(roota->right, sel.farther_node);
                }
            }

            roota->dtb_data.max_ncl_dist = std::max(
                roota->left->dtb_data.max_ncl_dist,
                roota->right->dtb_data.max_ncl_dist
            );
        }
    }


    void find_mst_next_dtb()
    {
        find_mst_next_dtb(&this->nodes[0], &this->nodes[0]);
    }


    void find_mst_next_qtb()
    {
        // find the point from another cluster that is closest to the i-th point
        // i.e., the nearest "alien"; iterate leaf-by-leaf
        #if OPENMP_IS_ENABLED
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t l=0; l<leaves.size(); ++l) {
            NODE* curleaf = leaves[l];

            if (curleaf->cluster_repr >= 0 && curleaf->idx_to - curleaf->idx_from > 1)  // all elems in the same cluster
            {
                GENIECLUST_ASSERT(curleaf->cluster_repr == ds.get_parent(curleaf->idx_from));
                Py_ssize_t ds_find_i = curleaf->cluster_repr;

                // NOTE: assumption: no race condition/atomic read...
                FLOAT ncl_dist_cur = ncl_dist[ds_find_i];

                if (ncl_dist_cur <= curleaf->qtb_data.lastnn_dist) continue;

                if (curleaf->qtb_data.lastnn_ind >= 0) {
                    Py_ssize_t ds_find_j = ds.get_parent(curleaf->qtb_data.lastnn_ind);
                    if (ds_find_i == ds_find_j)
                        curleaf->qtb_data.lastnn_ind = -1;
                }

                if (curleaf->qtb_data.lastnn_ind < 0) {
                    kdtree_nearest_outsider_multi<FLOAT, D, DISTANCE, NODE> nn(
                        this->data, (M>2)?(this->dcore.data()):NULL,
                        curleaf, ds.get_parents(), M
                    );
                    nn.find(&this->nodes[0], reset_nns?INFINITY:ncl_dist_cur);
                    curleaf->qtb_data.lastnn_ind = nn.get_nn_ind();
                    if (curleaf->qtb_data.lastnn_ind >= 0) {
                        curleaf->qtb_data.lastnn_dist = nn.get_nn_dist();
                        curleaf->qtb_data.lastnn_from = nn.get_nn_from();
                    }
                }

                if (curleaf->qtb_data.lastnn_ind < 0) continue;

                Py_ssize_t ds_find_j = ds.get_parent(curleaf->qtb_data.lastnn_ind);
                GENIECLUST_ASSERT(ds_find_i != ds_find_j);

                #if OPENMP_IS_ENABLED
                if (omp_nthreads > 1) omp_set_lock(&omp_lock);
                #endif
                if (curleaf->qtb_data.lastnn_dist < ncl_dist[ds_find_i]) {
                    ncl_dist[ds_find_i] = curleaf->qtb_data.lastnn_dist;
                    ncl_ind[ds_find_i]  = curleaf->qtb_data.lastnn_ind;
                    ncl_from[ds_find_i] = curleaf->qtb_data.lastnn_from;
                }

                if (curleaf->qtb_data.lastnn_dist < ncl_dist[ds_find_j]) {
                    ncl_dist[ds_find_j] = curleaf->qtb_data.lastnn_dist;
                    ncl_ind[ds_find_j]  = curleaf->qtb_data.lastnn_from;
                    ncl_from[ds_find_j] = curleaf->qtb_data.lastnn_ind;
                }
                #if OPENMP_IS_ENABLED
                if (omp_nthreads > 1) omp_unset_lock(&omp_lock);
                #endif
            }
            else {
                for (Py_ssize_t i=curleaf->idx_from; i<curleaf->idx_to; ++i) {
                    Py_ssize_t ds_find_i = ds.get_parent(i);

                    // NOTE: assumption: no race condition/atomic read...
                    FLOAT ncl_dist_cur = ncl_dist[ds_find_i];

                    if (ncl_dist_cur <= lastnn_dist[i]) continue;  // speeds up even for M==1

                    if (lastnn_ind[i] < 0 && M > 2 && lastnn_dist[i] <= dcore[i]) {
                        // try reusing (M-1) NN data
                        for (Py_ssize_t v=0; v<M-1; ++v)
                        {
                            Py_ssize_t j = Mnn_ind[i*(M-1)+((mutreach_adj<0.0)?((M-1)-1-v):(v))];
                            if (ds_find_i == ds.get_parent(j) || dcore[i] < dcore[j]) continue;

                            lastnn_dist[i] = dcore[i];
                            lastnn_ind[i] = j;

                            break;  // other candidates have d_M >= dcore[i] anyway
                        }
                    }

                    if (lastnn_ind[i] < 0) {
                        kdtree_nearest_outsider<FLOAT, D, DISTANCE, NODE> nn(
                            this->data, (M>2)?(this->dcore.data()):NULL,
                            i, ds.get_parents(), M
                        );
                        nn.find(&this->nodes[0], reset_nns?INFINITY:ncl_dist_cur);
                        if (nn.get_nn_ind() >= 0) {
                            // ret index can be negative if best found >= ncl_dist_cur
                            lastnn_ind[i]  = nn.get_nn_ind();
                            lastnn_dist[i] = nn.get_nn_dist();
                        }
                    }

                    if (lastnn_ind[i] < 0) continue;

                    Py_ssize_t ds_find_j = ds.get_parent(lastnn_ind[i]);
                    GENIECLUST_ASSERT(ds_find_i != ds_find_j);

                    #if OPENMP_IS_ENABLED
                    if (omp_nthreads > 1) omp_set_lock(&omp_lock);
                    #endif

                    if (lastnn_dist[i] < ncl_dist[ds_find_i]) {
                        ncl_dist[ds_find_i] = lastnn_dist[i];
                        ncl_ind[ds_find_i]  = lastnn_ind[i];
                        ncl_from[ds_find_i] = i;
                    }

                    if (lastnn_dist[i] < ncl_dist[ds_find_j]) {
                        ncl_dist[ds_find_j] = lastnn_dist[i];
                        ncl_ind[ds_find_j]  = i;
                        ncl_from[ds_find_j] = lastnn_ind[i];
                    }

                    #if OPENMP_IS_ENABLED
                    if (omp_nthreads > 1) omp_unset_lock(&omp_lock);
                    #endif
                }
            }
        }
    }


    void find_mst_next_stb()
    {
        // find the point from another cluster that is closest to the i-th point
        // i.e., the nearest "alien"
        #if OPENMP_IS_ENABLED
        #pragma omp parallel for schedule(static)
        #endif
        for (Py_ssize_t i=0; i<this->n; ++i) {
            Py_ssize_t ds_find_i = ds.get_parent(i);

            // NOTE: assumption: no race condition/atomic read...
            FLOAT ncl_dist_cur = ncl_dist[ds_find_i];

            if (ncl_dist_cur <= lastnn_dist[i]) continue;  // speeds up even for M==1

            if (lastnn_ind[i] < 0 && M > 2 && lastnn_dist[i] <= dcore[i]) {
                // try reusing (M-1) NN data
                for (Py_ssize_t v=0; v<M-1; ++v)
                {
                    Py_ssize_t j = Mnn_ind[i*(M-1)+((mutreach_adj<0.0)?((M-1)-1-v):(v))];
                    if (ds_find_i == ds.get_parent(j) || dcore[i] < dcore[j]) continue;

                    lastnn_dist[i] = dcore[i];
                    lastnn_ind[i] = j;

                    break;  // other candidates have d_M >= dcore[i] anyway
                }
            }

            if (lastnn_ind[i] < 0) {
                kdtree_nearest_outsider<FLOAT, D, DISTANCE, NODE> nn(
                    this->data, (M>2)?(this->dcore.data()):NULL,
                    i, ds.get_parents(), M
                );
                nn.find(&this->nodes[0], reset_nns?INFINITY:ncl_dist_cur);
                if (nn.get_nn_ind() >= 0) {
                    // ret index can be negative if best found >= ncl_dist_cur
                    lastnn_ind[i]  = nn.get_nn_ind();
                    lastnn_dist[i] = nn.get_nn_dist();
                }
            }

            if (lastnn_ind[i] < 0) continue;

            Py_ssize_t ds_find_j = ds.get_parent(lastnn_ind[i]);
            GENIECLUST_ASSERT(ds_find_i != ds_find_j);

            #if OPENMP_IS_ENABLED
            if (omp_nthreads > 1) omp_set_lock(&omp_lock);
            #endif

            if (lastnn_dist[i] < ncl_dist[ds_find_i]) {
                ncl_dist[ds_find_i] = lastnn_dist[i];
                ncl_ind[ds_find_i]  = lastnn_ind[i];
                ncl_from[ds_find_i] = i;
            }

            if (lastnn_dist[i] < ncl_dist[ds_find_j]) {
                ncl_dist[ds_find_j] = lastnn_dist[i];
                ncl_ind[ds_find_j]  = i;
                ncl_from[ds_find_j] = lastnn_ind[i];
            }

            #if OPENMP_IS_ENABLED
            if (omp_nthreads > 1) omp_unset_lock(&omp_lock);
            #endif
        }
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

        if (boruvka_variant == BORUVKA_QTB) {
            GENIECLUST_PROFILER_START
            setup_leaves();  // TODO: sesquitree only
            GENIECLUST_PROFILER_STOP("setup_leaves")
        }

        while (ds.get_k() != 1) {
            tree_iter++;

            #if GENIECLUST_R
            Rcpp::checkUserInterrupt();  // throws an exception, not a longjmp
            #elif GENIECLUST_PYTHON
            if (PyErr_CheckSignals() != 0) throw std::runtime_error("signal caught");
            #endif


            // set up ds_parents, ncl_dist, etc.
            // also ensures that ds.find(i) == ds.get_parent(i) for all i
            GENIECLUST_PROFILER_START
            update_cluster_data();
            GENIECLUST_PROFILER_STOP("update_cluster_data")

            // set up cluster_repr and reset max_ncl_dist
            GENIECLUST_PROFILER_START
            update_node_data();
            GENIECLUST_PROFILER_STOP("update_node_data")

            GENIECLUST_PROFILER_START
            if (boruvka_variant == BORUVKA_DTB)
                find_mst_next_dtb();
            else if (boruvka_variant == BORUVKA_QTB)
                find_mst_next_qtb();
            else
                find_mst_next_stb();

            // for each cluster, merge it with its nearest counterpart
            for (Py_ssize_t j=0; j<ds.get_k(); ++j) {
                Py_ssize_t i = ds_parents[j];
                GENIECLUST_ASSERT(ncl_ind[i] >= 0 && ncl_ind[i] < this->n);
                if (ds.find(i) != ds.find(ncl_ind[i])) {  // find, not get_parent!
                    GENIECLUST_ASSERT(ds.find(i) == ds.find(ncl_from[i]));
                    tree_ind[tree_edges*2+0] = ncl_from[i];
                    tree_ind[tree_edges*2+1] = ncl_ind[i];
                    tree_dist[tree_edges]    = ncl_dist[i];
                    ds.merge(i, ncl_ind[i]);
                    tree_edges++;
                }
            }

            GENIECLUST_PROFILER_STOP("find_mst iter #%d (tree_edges=%d)", (int)tree_iter, tree_edges)
        }
    }


public:
    kdtree_boruvka()
        : kdtree<FLOAT, D, DISTANCE, NODE>()
    {
        omp_nthreads = -1;
    }


    /**!
     * see fastmst.h for the description of the parameters,
     * no need to repeat that here
     */
    kdtree_boruvka(
        FLOAT* data, const Py_ssize_t n, const Py_ssize_t M=1,
        const Py_ssize_t max_leaf_size=16,
        const Py_ssize_t first_pass_max_brute_size=16,
        const bool use_dtb=false,
        const FLOAT mutreach_adj=-INFINITY
    ) :
        kdtree<FLOAT, D, DISTANCE, NODE>(data, n, max_leaf_size), tree_edges(0),
        ds(n), ncl_dist(n), ncl_ind(n), ncl_from(n),
        first_pass_max_brute_size(first_pass_max_brute_size),
        mutreach_adj(mutreach_adj), ds_parents(n), M(M)
    {
        GENIECLUST_ASSERT(M>0);

        if (M >= 2) {
            dcore.resize(n);
            Mnn_dist.resize(n*(M-1));
            Mnn_ind.resize(n*(M-1));
        }

        lastnn_dist.resize(n);
        lastnn_ind.resize(n);

        if (use_dtb)
            boruvka_variant = BORUVKA_DTB;
        else {
            if (max_leaf_size != first_pass_max_brute_size)  // TODO
                boruvka_variant = BORUVKA_QTB;
            else
                boruvka_variant = BORUVKA_STB;
        }

        reset_nns = (M<=2);  // plain Euclidean MST sometimes benefits from this


        #if OPENMP_IS_ENABLED
        omp_nthreads = Comp_get_max_threads();
        if (omp_nthreads > 1) omp_init_lock(&omp_lock);
        #else
        omp_nthreads = 1;
        #endif
    }


    ~kdtree_boruvka()
    {
        #if OPENMP_IS_ENABLED
        if (omp_nthreads > 1) omp_destroy_lock(&omp_lock);
        #endif
    }


    void mst(FLOAT* tree_dist, Py_ssize_t* tree_ind)
    {
        this->tree_dist = tree_dist;
        this->tree_ind  = tree_ind;

        if (ds.get_k() != (Py_ssize_t)this->n) ds.reset();
        tree_edges = 0;
        tree_iter = 0;

        for (Py_ssize_t i=0; i<this->n-1; ++i)     tree_dist[i] = INFINITY;
        for (Py_ssize_t i=0; i<2*(this->n-1); ++i) tree_ind[i]  = -1;

        // nodes is a deque...
        for (auto curnode = this->nodes.rbegin(); curnode != this->nodes.rend(); ++curnode)
            curnode->cluster_repr = -1;

        find_mst();
    }


    inline const FLOAT* get_Mnn_dist() const
    {
        GENIECLUST_ASSERT(M>1);
        return this->Mnn_dist.data();
    }

    inline const Py_ssize_t* get_Mnn_ind() const {
        GENIECLUST_ASSERT(M>1);
        return this->Mnn_ind.data();
    }

    inline const FLOAT* get_dcore() const {
        GENIECLUST_ASSERT(M>1);
        return this->dcore.data();
    }

    inline Py_ssize_t get_M() const { return this->M; }
};



/*!
 * Find a minimum spanning tree of X (in the tree)
 *
 * see _mst_euclid_kdtree
 *
 * @param tree a pre-built K-d tree containing n points
 * @param tree_dist [out] size n*k
 * @param tree_ind [out] size n*k
 * @param nn_dist [out] distances to M-1 nns of each point
 * @param nn_ind  [out] indexes of M-1 nns of each point
 */
template <typename FLOAT, Py_ssize_t D, typename DISTANCE, typename TREE>
void mst(
    TREE& tree,
    FLOAT* tree_dist,           // size n-1
    Py_ssize_t* tree_ind,       // size 2*(n-1),
    FLOAT* nn_dist=nullptr,     // size n*(M-1)
    Py_ssize_t* nn_ind=nullptr  // size n*(M-1)
) {
    tree.mst(tree_dist, tree_ind);

    Py_ssize_t n = tree.get_n();
    Py_ssize_t M = tree.get_M();
    const Py_ssize_t* perm = tree.get_perm();

    if (M > 1) {
        GENIECLUST_ASSERT(nn_dist);
        GENIECLUST_ASSERT(nn_ind);
        const FLOAT*      _nn_dist = tree.get_Mnn_dist();
        const Py_ssize_t* _nn_ind  = tree.get_Mnn_ind();

        for (Py_ssize_t i=0; i<n; ++i) {
            for (Py_ssize_t j=0; j<M-1; ++j) {
                nn_dist[perm[i]*(M-1)+j] = _nn_dist[i*(M-1)+j];
                nn_ind[perm[i]*(M-1)+j]  = perm[_nn_ind[i*(M-1)+j]];
            }
        }

        // if (M > 2) {
        //     // we need to recompute the distances as we applied a correction for ambiguity
        //     const FLOAT* _data     = tree.get_data();
        //     const FLOAT* _d_core   = tree.get_dcore();
        //     for (Py_ssize_t i=0; i<n-1; ++i) {
        //         Py_ssize_t i1 = tree_ind[2*i+0];
        //         Py_ssize_t i2 = tree_ind[2*i+1];
        //         tree_dist[i] = max3(
        //             DISTANCE::point_point(_data+i1*D, _data+i2*D),
        //             _d_core[i1],
        //             _d_core[i2]
        //         );
        //     }
        // }
    }

    for (Py_ssize_t i=0; i<n-1; ++i) {
        Py_ssize_t i1 = tree_ind[2*i+0];
        Py_ssize_t i2 = tree_ind[2*i+1];
        GENIECLUST_ASSERT(i1 != i2);
        GENIECLUST_ASSERT(i1 >= 0 && i1 < n);
        GENIECLUST_ASSERT(i2 >= 0 && i2 < n);
        tree_ind[2*i+0] = perm[i1];
        tree_ind[2*i+1] = perm[i2];
    }

    // the edges are not ordered, use Cmst_order
}



};  // namespace

#endif
