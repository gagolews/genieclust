/*  An implementation of K-d trees wrt the squared Euclidean distance
 *
 *  Supports finding k nearest neighbours of points within the same dataset;
 *  fast for small k and dimensionality d.
 *
 *  Features the sliding midpoint (midrange) rule suggested in "It's okay to be
 *  skinny, if your friends are fat" by S. Maneewongvatana and D.M. Mount, 1999
 *  and some further enhancements (minding locality of reference, etc.).
 *  This split criterion was the most efficient amongst those tested
 *  (different quantiles, adjusted midrange, etc.), at least for the purpose
 *  of building minimum spanning trees.
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


#ifndef __c_kdtree_h
#define __c_kdtree_h

#include "c_common.h"

#include <cmath>
#include <algorithm>
#include <vector>
#include <deque>
#include <array>


namespace quitefastkdtree {

template <typename FLOAT, Py_ssize_t D>
struct kdtree_node_base
{
    // some implementations store split_dim and split_val, but exact bounding
    // boxes (smallest) have better pruning capabilities
    std::array<FLOAT,D> bbox_min;  //< points' bounding box (min dims)
    std::array<FLOAT,D> bbox_max;  //< points' bounding box (max dims)

    // std::array<FLOAT,D> centroid;

    Py_ssize_t idx_from;
    Py_ssize_t idx_to;
};



template <typename FLOAT, Py_ssize_t D>
struct kdtree_node_knn : public kdtree_node_base<FLOAT, D>
{
    kdtree_node_knn* left;
    kdtree_node_knn* right;

    kdtree_node_knn() {
        left = nullptr;
        // right = nullptr;
    }

    inline bool is_leaf() const {
        return left == nullptr /*&& right == nullptr*/; // either both null or none
    }
};


template <typename FLOAT, Py_ssize_t D>
class kdtree_distance_sqeuclid
{
public:
    static inline FLOAT point_point(const FLOAT* x, const FLOAT* y)
    {
        FLOAT dist = 0.0;
        for (Py_ssize_t u=0; u<D; ++u)
            dist += square(x[u]-y[u]);
        return dist;
    }

    static inline FLOAT point_node(
        const FLOAT* x, const FLOAT* bbox_min, const FLOAT* bbox_max
    ) {
        FLOAT dist = 0.0;
        for (Py_ssize_t u=0; u<D; ++u) {
            if (bbox_min[u] > x[u])  // compare first, as FP subtract is slower
                dist += square(bbox_min[u] - x[u]);
            else if (x[u] > bbox_max[u])
                dist += square(x[u] - bbox_max[u]);
            // else dist += 0.0;
        }
        return dist;
    }

    static inline FLOAT node_node(
        const FLOAT* bbox_min_a, const FLOAT* bbox_max_a,
        const FLOAT* bbox_min_b, const FLOAT* bbox_max_b
    ) {
        FLOAT dist = 0.0;
        for (Py_ssize_t u=0; u<D; ++u) {
            if (bbox_min_b[u] > bbox_max_a[u])
                dist += square(bbox_min_b[u] - bbox_max_a[u]);
            else if (bbox_min_a[u] > bbox_max_b[u])
                dist += square(bbox_min_a[u] - bbox_max_b[u]);
            // else dist += 0.0;
        }
        return dist;
    }
};


/** A class enabling searching for k nearest neighbours of a given point
 *  (excluding self) within the same dataset;
 *  it is thread-safe
 */
template <
    typename FLOAT,
    Py_ssize_t D,
    typename DISTANCE=kdtree_distance_sqeuclid<FLOAT,D>,
    typename NODE=kdtree_node_knn<FLOAT,D>
>
class kdtree_kneighbours
{
private:
    const Py_ssize_t which;    ///< for which point are we getting the k-nns?
    const Py_ssize_t k;        ///< how many nns?
    const FLOAT* x;      ///< the point itself (shortcut)
    const FLOAT* data;   ///< the dataset
    FLOAT* knn_dist;
    Py_ssize_t* knn_ind;

    const Py_ssize_t max_brute_size;  // when to switch to the brute-force mode? 0 to honour the tree's max_leaf_size


    inline void point_vs_points(Py_ssize_t idx_from, Py_ssize_t idx_to)
    {
        const FLOAT* y = data+D*idx_from;
        for (Py_ssize_t i=idx_from; i<idx_to; ++i, y+=D) {
            FLOAT dist = DISTANCE::point_point(x, y);

            if (dist >= knn_dist[k-1])
                continue;

            // insertion-sort like scheme (fast for small k)
            Py_ssize_t j = k-1;
            while (j > 0 && dist < knn_dist[j-1]) {
                knn_ind[j]  = knn_ind[j-1];
                knn_dist[j] = knn_dist[j-1];
                j--;
            }
            knn_ind[j] = i;
            knn_dist[j] = dist;
        }
    }


    void find_knn(const NODE* root)
    {
        if (root->is_leaf() || root->idx_to-root->idx_from <= max_brute_size) {
            if (which < root->idx_from || which >= root->idx_to)
                point_vs_points(root->idx_from, root->idx_to);
            else {
                point_vs_points(root->idx_from, which);
                point_vs_points(which+1, root->idx_to);
            }
            return;
        }


        // closer node first (significant speedup)
        FLOAT left_dist  = DISTANCE::point_node(
            x, root->left->bbox_min.data(),  root->left->bbox_max.data()
        );
        FLOAT right_dist = DISTANCE::point_node(
            x, root->right->bbox_min.data(), root->right->bbox_max.data()
        );

        #define FIND_KNN_PROCESS(nearer_dist, farther_dist, nearer_node, farther_node) \
        if (nearer_dist < knn_dist[k-1]) {    \
            find_knn(nearer_node);            \
            if (farther_dist < knn_dist[k-1]) \
                find_knn(farther_node);       \
        }                                     \

        if (left_dist <= right_dist) {
            FIND_KNN_PROCESS(left_dist, right_dist, root->left, root->right);
        }
        else {
            FIND_KNN_PROCESS(right_dist, left_dist, root->right, root->left);
        }

        // slower:
        // if (DISTANCE::point_point(x, root->left->centroid.data()) <= DISTANCE::point_point(x, root->right->centroid.data()))
        // {
        //     if (left_dist < knn_dist[k-1])
        //         find_knn(root->left);
        //     if (right_dist < knn_dist[k-1])
        //         find_knn(root->right);
        // }
        // else {
        //     if (right_dist < knn_dist[k-1])
        //         find_knn(root->right);
        //     if (left_dist < knn_dist[k-1])
        //         find_knn(root->left);
        // }


    }

public:
    kdtree_kneighbours(
        const FLOAT* data,
        const FLOAT* x,
        const Py_ssize_t which,
        FLOAT* knn_dist,
        Py_ssize_t* knn_ind,
        const Py_ssize_t k,
        const Py_ssize_t max_brute_size=0
    ) :
        which(which), k(k), x(x), data(data),
        knn_dist(knn_dist), knn_ind(knn_ind),
        max_brute_size(max_brute_size)
    {
        if (x == nullptr) {
            GENIECLUST_ASSERT(which >= 0);
            this->x = data+D*which;
        }
        // // Pre-flight (no benefit)
        // for (Py_ssize_t i=0; i<=2*k; ++i) {
        //     Py_ssize_t j = (Py_ssize_t)which-i-(Py_ssize_t)k;
        //     if (j == (Py_ssize_t)which) continue;
        //     else if (j < 0) j = (Py_ssize_t)n+j;
        //     else if (j >= (Py_ssize_t)n) j = j - (Py_ssize_t)n;
        //
        //     const FLOAT* y = data+j*D;
        //     FLOAT dist = 0.0;
        //     for (size_t u=0; u<D; ++u)
        //         dist += square(x[u]-*(y++));
        //
        //     if (dist >= knn_dist[k-1])
        //         continue;
        //
        //     // insertion-sort like scheme (fast for small k)
        //
        //     j = (Py_ssize_t)k-1;
        //     while (j > 0 && dist < knn_dist[j-1]) {
        //         knn_dist[j] = knn_dist[j-1];
        //         j--;
        //     }
        //     knn_dist[j] = dist;
        // }
        //
        // knn_dist[k-1] = std::nexttoward(knn_dist[k-1], INFINITY);
        // for (size_t i=0; i<k-1; ++i) knn_dist[i] = knn_dist[k-1];
    }


    void find(const NODE* root, bool reset=true)
    {
        if (reset) {
            for (Py_ssize_t i=0; i<k; ++i) knn_dist[i] = INFINITY;
            for (Py_ssize_t i=0; i<k; ++i) knn_ind[i]  = which;
        }

        find_knn(root);
    }
};




template <
    typename FLOAT,
    Py_ssize_t D,
    typename DISTANCE=kdtree_distance_sqeuclid<FLOAT,D>,
    typename NODE=kdtree_node_knn<FLOAT, D>
>
class kdtree
{
protected:
    std::deque< NODE > nodes;  // stores all nodes

    FLOAT* data;  //< destroyable; a row-major n*D matrix (points are permuted, see perm[] - that's for better locality of reference)
    const Py_ssize_t n;  //< number of points
    std::vector<Py_ssize_t> perm;  //< original point indexes

    const Py_ssize_t max_leaf_size;  //< unless in pathological cases

    inline void compute_bounding_box(NODE*& root)
    {
        const FLOAT* y = data+root->idx_from*D;
        for (Py_ssize_t u=0; u<D; ++u) {
            root->bbox_min[u] = *y;
            root->bbox_max[u] = *y;
            // root->centroid[u] = *y;
            ++y;
        }
        for (Py_ssize_t i=root->idx_from+1; i<root->idx_to; ++i) {
            for (Py_ssize_t u=0; u<D; ++u) {
                if      (*y < root->bbox_min[u]) root->bbox_min[u] = *y;
                else if (*y > root->bbox_max[u]) root->bbox_max[u] = *y;
                // root->centroid[u] += *y;
                ++y;
            }
        }
        // for (Py_ssize_t u=0; u<D; ++u) {
        //     root->centroid[u] /= (root->idx_to-root->idx_from);
        // }
    }


    void build_tree(
        NODE* root, Py_ssize_t idx_from, Py_ssize_t idx_to
    )
    {
        GENIECLUST_ASSERT(idx_to - idx_from > 0);

        root->idx_from = idx_from;
        root->idx_to   = idx_to;

        compute_bounding_box(root);

        if (idx_to - idx_from <= max_leaf_size) {
            // this will be a leaf node; nothing more to do
            return;
        }


        // cut by the dim of the greatest range
        Py_ssize_t split_dim = 0;
        FLOAT dim_width = root->bbox_max[0] - root->bbox_min[0];
        for (Py_ssize_t u=1; u<D; ++u) {
            FLOAT cur_width = root->bbox_max[u] - root->bbox_min[u];
            if (cur_width > dim_width) {
                dim_width = cur_width;
                split_dim = u;
            }
        }
        // The sliding midpoint rule:
        FLOAT split_val = 0.5*(root->bbox_min[split_dim] + root->bbox_max[split_dim]);  // midrange

        // this doesn't improve:
        // size_t cnt = 0;
        // for (size_t i=idx_from; i<idx_to; ++i) {
        //     if (data[i*D+split_dim] <= split_val) cnt++;
        // }
        // if (cnt <= max_leaf_size/4)
        //     split_val = root->bbox_min[split_dim]+0.75*(root->bbox_max[split_dim] - root->bbox_min[split_dim]);
        // else if ((idx_to-idx_from)-cnt <= max_leaf_size/4)
        //     split_val = root->bbox_min[split_dim]+0.25*(root->bbox_max[split_dim] - root->bbox_min[split_dim]);


        if (dim_width == 0) {
            // a pathological case: this will be a "large" leaf (all points with the same coords)
            return;
        }


        GENIECLUST_ASSERT(root->bbox_min[split_dim] < split_val);
        GENIECLUST_ASSERT(split_val < root->bbox_max[split_dim]);

        // FLOAT split_left_max  = root->bbox_min[split_dim];
        // FLOAT split_right_min = root->bbox_max[split_dim];

        // partition data[idx_from:idx_left, split_dim] <= split_val, data[idx_left:idx_to, split_dim] > split_val
        Py_ssize_t idx_left = idx_from;
        Py_ssize_t idx_right = idx_to-1;
        while (true) {
            while (data[idx_left*D+split_dim] <= split_val) {  // split_val < curbox_max[split_dim]
                // if (data[idx_left*D+split_dim] > split_left_max)
                    // split_left_max = data[idx_left*D+split_dim];
                idx_left++;
            }

            while (data[idx_right*D+split_dim] > split_val) {  // split_val > curbox_min[split_dim]
                // if (data[idx_right*D+split_dim] < split_right_min)
                    // split_right_min = data[idx_right*D+split_dim];
                idx_right--;
            }

            if (idx_left >= idx_right)
                break;

            std::swap(perm[idx_left], perm[idx_right]);
            for (Py_ssize_t u=0; u<D; ++u)
                std::swap(data[idx_left*D+u], data[idx_right*D+u]);
        }

        GENIECLUST_ASSERT(idx_left > idx_from);
        GENIECLUST_ASSERT(idx_left < idx_to);

        GENIECLUST_ASSERT(data[idx_left*D+split_dim] > split_val);
        GENIECLUST_ASSERT(data[(idx_left-1)*D+split_dim] <= split_val);

        // GENIECLUST_ASSERT(split_left_max <= split_val);
        // GENIECLUST_ASSERT(split_right_min > split_val);

        // root->intnode_data.split_dim = split_dim;
        // root->intnode_data.split_left_max = split_left_max;
        // root->intnode_data.split_right_min = split_right_min;

        nodes.push_back(NODE());
        root->left = &nodes[nodes.size()-1];

        nodes.push_back(NODE());
        root->right = &nodes[nodes.size()-1];

        build_tree(root->left, idx_from, idx_left);
        build_tree(root->right, idx_left, idx_to);
    }


public:
    kdtree()
        : data(nullptr), n(0), perm(0), max_leaf_size(1)
    {

    }

    kdtree(FLOAT* data, const Py_ssize_t n, const Py_ssize_t max_leaf_size=16)
        : data(data), n(n), perm(n), max_leaf_size(max_leaf_size)
    {
        GENIECLUST_ASSERT(max_leaf_size > 0);

        for (Py_ssize_t i=0; i<n; ++i) perm[i] = i;

        GENIECLUST_PROFILER_USE

        GENIECLUST_PROFILER_START
        GENIECLUST_ASSERT(nodes.size()==0);
        nodes.push_back(NODE());
        build_tree(&nodes[0], 0, n);
        GENIECLUST_PROFILER_STOP("build_tree")
    }


    ~kdtree()
    {
        nodes.clear();
    }


    inline Py_ssize_t get_n() const { return n; }
    inline const Py_ssize_t* get_perm() const { return perm.data(); }
    inline const FLOAT* get_data() const { return data; }


    void kneighbours(Py_ssize_t which, FLOAT* knn_dist, Py_ssize_t* knn_ind, Py_ssize_t k)
    {
        kdtree_kneighbours<FLOAT, D, DISTANCE, NODE> knn(data, nullptr, which, knn_dist, knn_ind, k);
        knn.find(&nodes[0]);
    }

    void kneighbours(const FLOAT* x, FLOAT* knn_dist, Py_ssize_t* knn_ind, Py_ssize_t k)
    {
        kdtree_kneighbours<FLOAT, D, DISTANCE, NODE> knn(data, x, -1, knn_dist, knn_ind, k);
        knn.find(&nodes[0]);
    }
};


/*!
 * k nearest neighbours of each point in X (in the tree);
 * each point is not its own neighbour
 *
 * see _knn_sqeuclid_kdtree
 *
 * @param tree a pre-built K-d tree containing n points
 * @param knn_dist [out] size n*k
 * @param knn_ind [out] size n*k
 * @param k number of neighbours
 */
template <typename FLOAT, Py_ssize_t D, typename TREE>
void kneighbours(
    TREE& tree,
    FLOAT* knn_dist,     // size n*k
    Py_ssize_t* knn_ind, // size n*k
    Py_ssize_t k
) {
    Py_ssize_t n = tree.get_n();
    const Py_ssize_t* perm = tree.get_perm();

    #if OPENMP_IS_ENABLED
    #pragma omp parallel for schedule(static)
    #endif
    for (Py_ssize_t i=0; i<n; ++i) {
        Py_ssize_t i_orig = perm[i];
        tree.kneighbours(i, knn_dist+k*i_orig, knn_ind+k*i_orig, k);
    }

    for (Py_ssize_t i=0; i<n*k; ++i) {
        knn_ind[i] = perm[knn_ind[i]];
    }
}


/*!
 * k nearest neighbours of each point in Y from X (in the tree)
 *
 * see _knn_sqeuclid_kdtree
 *
 * @param tree a pre-built K-d tree containing n points
 * @param Y size m*D
 * @param m number of points in Y
 * @param knn_dist [out] size n*k
 * @param knn_ind [out] size n*k
 * @param k number of neighbours
 */
template <typename FLOAT, Py_ssize_t D, typename TREE>
void kneighbours(
    TREE& tree,
    const FLOAT* Y,
    Py_ssize_t m,
    FLOAT* knn_dist,      // size m*k
    Py_ssize_t* knn_ind,  // size m*k
    Py_ssize_t k
) {
    #if OPENMP_IS_ENABLED
    #pragma omp parallel for schedule(static)
    #endif
    for (Py_ssize_t i=0; i<m; ++i) {
        tree.kneighbours(Y+i*D, knn_dist+k*i, knn_ind+k*i, k);
    }

    const Py_ssize_t* perm = tree.get_perm();
    for (Py_ssize_t i=0; i<m*k; ++i) {
        knn_ind[i] = perm[knn_ind[i]];
    }
}



};  // namespace

#endif
