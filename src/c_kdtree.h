/*  An implementation of Kd-trees wrt the squared Euclidean distance
 *
 *  Supports finding k nearest neighbours of points within the same dataset;
 *  fast for small k and dimensionality d.
 *
 *  TODO: support of querying points outside of a dataset can be added if needed.
 *
 *
 *  Features the sliding midpoint rule suggested in "It's okay to be skinny,
 *  if your friends are fat" by S. Maneewongvatana and D.M. Mount, 1999
 *  and some further enhancements (minding locality of reference,
 *  multithreading, etc.).
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
#include <cstddef>
#include <algorithm>
#include <vector>
#include <deque>
#include <array>


namespace mgtree {

template <typename FLOAT>
inline FLOAT square(FLOAT v) { return v*v; }



template <typename FLOAT, size_t D>
struct kdtree_node_base
{

    std::array<FLOAT,D> bbox_min;  //< points' bounding box (min dims)
    std::array<FLOAT,D> bbox_max;  //< points' bounding box (max dims)

    size_t idx_from;
    size_t idx_to;

    // union {
    //     struct {
    //         size_t idx_from;
    //         size_t idx_to;
    //     } leaf_data;
    //     struct {
    //         //size_t split_dim;
    //         //FLOAT split_left_max;
    //         //FLOAT split_right_min;
    //     } intnode_data;
    // };

};



template <typename FLOAT, size_t D>
struct kdtree_node_knn : public kdtree_node_base<FLOAT, D>
{
    kdtree_node_knn* left;
    kdtree_node_knn* right;

    kdtree_node_knn() {
        left = nullptr;
        right = nullptr;
    }

    inline bool is_leaf() const { return left == nullptr /*&& right == nullptr*/; }  // either both null or none
};





/** A class enabling searching for k nearest neighbours of a given point
 *  (excluding self) within the same dataset;
 *  it is thread-safe
 */
template <typename FLOAT, size_t D, typename NODE=kdtree_node_knn<FLOAT,D> >
class kdtree_kneighbours
{
private:
    const FLOAT* data;
    const size_t n;
    const size_t which;
    FLOAT* knn_dist;
    size_t* knn_ind;
    const size_t k;

    const FLOAT* x;

    inline FLOAT dist_to_node(const NODE* root) const
    {
        FLOAT dist = 0.0;
        for (size_t u=0; u<D; ++u) {
            if (root->bbox_min[u] > x[u])  // it's better to compare first, as FP subtract is slower
                dist += square(root->bbox_min[u] - x[u]);
            else if (x[u] > root->bbox_max[u])
                dist += square(x[u] - root->bbox_max[u]);
            // else dist += 0.0;
        }
        return dist;
    }



public:
    kdtree_kneighbours(
        const FLOAT* data,
        const size_t n,
        const size_t which,
        FLOAT* knn_dist,
        size_t* knn_ind,
        const size_t k
    ) :
        data(data), n(n), which(which), knn_dist(knn_dist), knn_ind(knn_ind), k(k),
        x(data+D*which)
    {
        for (size_t i=0; i<k; ++i) knn_dist[i] = INFINITY;
        for (size_t i=0; i<k; ++i) knn_ind[i]  = which;

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


    void find(const NODE* root)
    {
        find_start:  /* tail recursion elimination */

        if (root->is_leaf()) {
            const FLOAT* y = data+D*root->idx_from;
            for (size_t i=root->idx_from; i<root->idx_to; ++i) {
                if (i == which) {
                    y += D;
                    continue;
                }

                FLOAT dist = 0.0;
                for (size_t u=0; u<D; ++u)
                    dist += square(x[u]-*(y++));

                if (dist >= knn_dist[k-1])
                    continue;

                // insertion-sort like scheme (fast for small k)
                size_t j = k-1;
                while (j > 0 && dist < knn_dist[j-1]) {
                    knn_ind[j]  = knn_ind[j-1];
                    knn_dist[j] = knn_dist[j-1];
                    j--;
                }
                knn_ind[j] = i;
                knn_dist[j] = dist;
            }

            return;
        }

        FLOAT dist_left  = dist_to_node(root->left);
        FLOAT dist_right = dist_to_node(root->right);

        // closer node first (significant speedup
        if (dist_left < dist_right) {
            if (dist_left < knn_dist[k-1]) {
                find(root->left);
                if (dist_right < knn_dist[k-1]) {
                    //find(root->right);
                    root = root->right;
                    goto find_start;  // tail recursion elimination
                }
            }
        }
        else {
            if (dist_right < knn_dist[k-1]) {
                find(root->right);
                if (dist_left < knn_dist[k-1]) {
                    //find(root->left);
                    root = root->left;
                    goto find_start;  // tail recursion elimination
                }
            }
        }
    }
};




template <typename FLOAT, size_t D, typename NODE=kdtree_node_knn<FLOAT, D> >
class kdtree
{
protected:
    std::deque< NODE > nodes;  // stores all nodes

    NODE* root;  // nodes[0] or nullptr

    FLOAT* data;  //< destroyable; a row-major n*D matrix (points are permuted, see perm)
    const size_t n;  //< number of points
    std::vector<size_t> perm;  //< original point indexes

    const size_t max_leaf_size;  //< unless in pathological cases


    // void delete_tree(NODE*& root)
    // {
    //     if (!root) return;
    //
    //     delete_tree(root->left);
    //     delete_tree(root->right);
    //
    //     delete root;
    //     root = nullptr;
    // }


    inline void compute_bounding_box(NODE*& root)
    {
        const FLOAT* _x = data+root->idx_from*D;
        for (size_t u=0; u<D; ++u) {
            root->bbox_min[u] = *_x;
            root->bbox_max[u] = *_x;
            ++_x;
        }
        for (size_t i=root->idx_from+1; i<root->idx_to; ++i) {
            for (size_t u=0; u<D; ++u) {
                if      (*_x < root->bbox_min[u]) root->bbox_min[u] = *_x;
                else if (*_x > root->bbox_max[u]) root->bbox_max[u] = *_x;
                ++_x;
            }
        }
    }


    void build_tree(
        NODE*& root, size_t idx_from, size_t idx_to
    )
    {
        GENIECLUST_ASSERT(idx_to - idx_from > 0);
        //root = new NODE();
        nodes.push_back(NODE());
        root = &nodes[nodes.size()-1];

        root->idx_from = idx_from;
        root->idx_to   = idx_to;

        compute_bounding_box(root);

        if (idx_to - idx_from <= max_leaf_size) {
            // this will be a leaf node; nothing more to do
            return;
        }

        // cut by the dim of the greatest range
        size_t split_dim = 0;
        FLOAT dim_width = root->bbox_max[0]-root->bbox_min[0];
        for (size_t u=1; u<D; ++u) {
            FLOAT cur_width = root->bbox_max[u]-root->bbox_min[u];
            if (cur_width > dim_width) {
                dim_width = cur_width;
                split_dim = u;
            }
        }

        if (dim_width == 0) {
            // a pathological case: this will be a "large" leaf (all points with the same coords)
            return;
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


        GENIECLUST_ASSERT(root->bbox_min[split_dim] < split_val);
        GENIECLUST_ASSERT(split_val < root->bbox_max[split_dim]);

        FLOAT split_left_max  = root->bbox_min[split_dim];
        FLOAT split_right_min = root->bbox_max[split_dim];

        // partition data[idx_from:idx_left, split_dim] <= split_val, data[idx_left:idx_to, split_dim] > split_val
        size_t idx_left = idx_from;
        size_t idx_right = idx_to-1;
        while (true) {
            while (data[idx_left*D+split_dim] <= split_val) {  // split_val < curbox_max[split_dim]
                if (data[idx_left*D+split_dim] > split_left_max)
                    split_left_max = data[idx_left*D+split_dim];
                idx_left++;
            }

            while (data[idx_right*D+split_dim] > split_val) {  // split_val > curbox_min[split_dim]
                if (data[idx_right*D+split_dim] < split_right_min)
                    split_right_min = data[idx_right*D+split_dim];
                idx_right--;
            }

            if (idx_left >= idx_right)
                break;

            std::swap(perm[idx_left], perm[idx_right]);
            for (size_t u=0; u<D; ++u)
                std::swap(data[idx_left*D+u], data[idx_right*D+u]);
        }

        GENIECLUST_ASSERT(idx_left > idx_from);
        GENIECLUST_ASSERT(idx_left < idx_to);

        // for (size_t i=idx_from; i<idx_left; ++i) {  // TODO: delme
        //     GENIECLUST_ASSERT(data[i*D+split_dim] <= split_val);
        // }
        // for (size_t i=idx_left; i<idx_to; ++i) {  // TODO: delme
        //     GENIECLUST_ASSERT(data[i*D+split_dim] > split_val);
        // }

        GENIECLUST_ASSERT(data[idx_left*D+split_dim] > split_val);
        GENIECLUST_ASSERT(data[(idx_left-1)*D+split_dim] <= split_val);

        GENIECLUST_ASSERT(split_left_max <= split_val);
        GENIECLUST_ASSERT(split_right_min > split_val);

        // root->intnode_data.split_dim = split_dim;
        // root->intnode_data.split_left_max = split_left_max;
        // root->intnode_data.split_right_min = split_right_min;

        build_tree(root->left, idx_from, idx_left);
        build_tree(root->right, idx_left, idx_to);
    }


public:
    kdtree()
        : root(nullptr), data(nullptr), n(0), perm(0), max_leaf_size(1)
    {

    }

    kdtree(FLOAT* data, const size_t n, const size_t max_leaf_size=32)
        : root(nullptr), data(data), n(n), perm(n), max_leaf_size(max_leaf_size)
    {
        GENIECLUST_ASSERT(max_leaf_size > 0);
        for (size_t i=0; i<n; ++i) perm[i] = i;
        build_tree(root, 0, n);
    }


    ~kdtree()
    {
        //delete_tree(root);
        root = nullptr;
        nodes.clear();
    }


    inline size_t get_n() const { return n; }
    inline std::vector<size_t>& get_perm() { return perm; }
    inline FLOAT* get_data() { return data; }


    void kneighbours(size_t which, FLOAT* knn_dist, size_t* knn_ind, size_t k)
    {
        kdtree_kneighbours<FLOAT, D, NODE> knn(data, n, which, knn_dist, knn_ind, k);
        knn.find(root);
    }
};



template <typename FLOAT, size_t D, typename TREE>
void kneighbours(
    TREE& tree,
    FLOAT* knn_dist,
    size_t* knn_ind,
    size_t k
) {
    size_t n = tree.get_n();
    const size_t* perm = tree.get_perm().data();

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i=0; i<n; ++i) {
        size_t i_orig = perm[i];
        tree.kneighbours(i, knn_dist+k*i_orig, knn_ind+k*i_orig, k);
    }

    for (size_t i=0; i<n*k; ++i) {
        knn_ind[i] = perm[knn_ind[i]];
    }
}



};  // namespace

#endif
