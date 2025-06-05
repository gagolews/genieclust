/*  Kd-trees w.r.t. the squared Euclidean distance (optimised)
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

#include <cstddef>
#include <stdexcept>
#include <limits>
#include <vector>
#include <array>

template <typename FLOAT, size_t D>
struct kdtree_node
{
    kdtree_node* left;
    kdtree_node* right;

    std::array<FLOAT,D> bbox_min;  //< points' bounding box (min dims)
    std::array<FLOAT,D> bbox_max;  //< points' bounding box (max dims)

    union {
        struct {
            size_t idx_from;
            size_t idx_to;
        } leaf_data;
        struct {
            size_t split_dim;
            //FLOAT split_left_max;
            //FLOAT split_right_min;
        } intnode_data;
    };

    kdtree_node()
        : left(nullptr), right(nullptr)
    {
        leaf_data.idx_from = 0;
        leaf_data.idx_to = 0;
    }

    bool is_leaf() const { return left == nullptr /*&& right == nullptr*/; }  // either both null or none
};


template <typename FLOAT, size_t D>
class kdtree_kneighbours
{
private:
    const FLOAT* data;

    const size_t which;
    FLOAT* knn_dist;
    size_t* knn_ind;
    const size_t k;

    const FLOAT* x;


    static constexpr FLOAT infty = std::numeric_limits<FLOAT>::infinity();

    inline FLOAT square(FLOAT v) const { return v*v; }


    FLOAT get_dist(const kdtree_node<FLOAT, D>* root)
    {
        FLOAT dist = 0.0;
        for (size_t u=0; u<D; ++u) {
            if (x[u] < root->bbox_min[u])
                dist += square(x[u] - root->bbox_min[u]);
            else if (x[u] > root->bbox_max[u])
                dist += square(x[u] - root->bbox_max[u]);
            // else dist += 0.0;
        }
        return dist;
    }


public:
    kdtree_kneighbours(
        const FLOAT* data,
        const size_t which,
        FLOAT* knn_dist,
        size_t* knn_ind,
        const size_t k
    ) :
        data(data), which(which), knn_dist(knn_dist), knn_ind(knn_ind), k(k),
        x(data+D*which)
    {
        for (size_t i=0; i<k; ++i) knn_dist[i] = infty;
        for (size_t i=0; i<k; ++i) knn_ind[i]  = which;
    }


    void find(const kdtree_node<FLOAT, D>* root)
    {
        if (root->is_leaf()) {
            const FLOAT* y = data+D*root->leaf_data.idx_from;
            for (size_t i=root->leaf_data.idx_from; i<root->leaf_data.idx_to; ++i) {
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

        FLOAT dist_left  = get_dist(root->left);
        FLOAT dist_right = get_dist(root->right);

        if (dist_left < dist_right) {
            if (dist_left < knn_dist[k-1]) {
                find(root->left);
                if (dist_right < knn_dist[k-1])
                    find(root->right);
            }
        }
        else {
            if (dist_right < knn_dist[k-1]) {
                find(root->right);
                if (dist_left < knn_dist[k-1])
                    find(root->left);
            }
        }
    }
};


template <typename FLOAT, size_t D>
class kdtree
{
protected:
    kdtree_node<FLOAT, D>* root;

    FLOAT* data;  //< destroyable; a row-major n*D matrix (points are permuted, see perm)
    const size_t n;  //< number of points
    std::vector<size_t> perm;  //< original point indexes

    const size_t max_leaf_size;
    static constexpr FLOAT infty = std::numeric_limits<FLOAT>::infinity();


    void delete_tree(kdtree_node<FLOAT, D>*& root)
    {
        if (!root) return;

        delete_tree(root->left);
        delete_tree(root->right);

        delete root;
        root = nullptr;
    }


    void build_tree(
        kdtree_node<FLOAT, D>*& root, size_t idx_from, size_t idx_to
    )
    {
        GENIECLUST_ASSERT(idx_to - idx_from > 0);
        root = new kdtree_node<FLOAT, D>();

        // get the node's bounding box
        for (size_t u=0; u<D; ++u) root->bbox_min[u] = data[idx_from*D+u];
        for (size_t u=0; u<D; ++u) root->bbox_max[u] = data[idx_from*D+u];

        for (size_t i=idx_from+1; i<idx_to; ++i) {
            for (size_t u=0; u<D; ++u) {
                if      (data[i*D+u]<root->bbox_min[u]) root->bbox_min[u] = data[i*D+u];
                else if (data[i*D+u]>root->bbox_max[u]) root->bbox_max[u] = data[i*D+u];
            }
        }

        if (idx_to - idx_from <= max_leaf_size) {
            // a leaf node; nothing more to do
            root->leaf_data.idx_from = idx_from;
            root->leaf_data.idx_to   = idx_to;
            // printf("%5d %5d %5d [% 8f,% 8f]-[% 8f,% 8f]\n",
            //         (int)(idx_to-idx_from), (int)idx_from, (int)idx_to,
            //         curbox_min[0], curbox_max[0], curbox_min[1], curbox_max[1]);
            return;
        }

        // cut by the dim of the greatest range
        size_t split_dim = 0;
        FLOAT dim_width = -infty;
        for (size_t u=0; u<D; ++u) {
            FLOAT cur_width = root->bbox_max[u]-root->bbox_min[u];
            if (cur_width > dim_width) {
                dim_width = cur_width;
                split_dim = u;
            }
        }

        if (dim_width == 0) {
            // a pathological case: this will be a "large" leaf (all points with the same coords)
            root->leaf_data.idx_from = idx_from;
            root->leaf_data.idx_to   = idx_to;
            return;
        }

        FLOAT split_val = 0.5*(root->bbox_min[split_dim] + root->bbox_max[split_dim]);  // midrange

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

        for (size_t i=idx_from; i<idx_left; ++i) {  // TODO: delme
            GENIECLUST_ASSERT(data[i*D+split_dim] <= split_val);
            // if (data[i*D+split_dim] > split_left_max)
                // split_left_max = data[i*D+split_dim];
        }
        for (size_t i=idx_left; i<idx_to; ++i) {  // TODO: delme
            GENIECLUST_ASSERT(data[i*D+split_dim] > split_val);
            // if (data[i*D+split_dim] < split_right_min)
                // split_right_min = data[i*D+split_dim];
        }

        GENIECLUST_ASSERT(data[idx_left*D+split_dim] > split_val);
        GENIECLUST_ASSERT(data[(idx_left-1)*D+split_dim] <= split_val);

        GENIECLUST_ASSERT(split_left_max <= split_val);
        GENIECLUST_ASSERT(split_right_min > split_val);

        root->intnode_data.split_dim = split_dim;
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

    kdtree(FLOAT* data, const size_t n, const size_t max_leaf_size=12)
        : root(nullptr), data(data), n(n), perm(n), max_leaf_size(max_leaf_size)
    {
        for (size_t i=0; i<n; ++i) perm[i] = i;
        build_tree(root, 0, n);
    }


    ~kdtree()
    {
        delete_tree(root);
    }


    inline size_t get_n() const { return n; }
    inline std::vector<size_t>& get_perm() { return perm; }
    inline FLOAT* get_data() { return data; }


    void kneighbours(size_t which, FLOAT* knn_dist, size_t* knn_ind, size_t k)
    {
        kdtree_kneighbours<FLOAT, D> knn(data, which, knn_dist, knn_ind, k);
        knn.find(root);
    }
};


template <typename FLOAT, size_t D>
void kneighbours(
    kdtree<FLOAT, D>& tree,
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




#endif
