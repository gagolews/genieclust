/*
 *  An implementation of kd-trees
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

#include <vector>
#include <array>


#ifndef GENIECLUST_ASSERT
#define __GENIECLUST_STR(x) #x
#define GENIECLUST_STR(x) __GENIECLUST_STR(x)

#define GENIECLUST_ASSERT(EXPR) { if (!(EXPR)) \
    throw std::runtime_error( "genieclust: Assertion " #EXPR " failed in "\
        __FILE__ ":" GENIECLUST_STR(__LINE__) ); }
#endif


template <typename FLOAT, size_t D>
struct kdtree_node
{
    kdtree_node* left;
    kdtree_node* right;

    union {
        struct {
            size_t idx_from;
            size_t idx_to;
        } leaf_data;
        struct {
            size_t split_dim;
            FLOAT split_left_max;
            FLOAT split_right_min;
        } intnode_data;
    };

    //std::array<FLOAT,D> bbox_min;
    //std::array<FLOAT,D> bbox_max;

    kdtree_node()
        : left(nullptr), right(nullptr)
    {
        leaf_data.idx_from = 0;
        leaf_data.idx_to = 0;
    }

    bool is_leaf() const { return left == nullptr /*&& right == nullptr*/; }  // either both null or none
};


template <typename FLOAT, size_t D>
class kdtree
{
protected:
    kdtree_node<FLOAT, D>* root;

    FLOAT* data;  // destroyable; a row-major n*D matrix
    const size_t n;
    std::vector<size_t> perm;

    std::array<FLOAT, D> bbox_min;
    std::array<FLOAT, D> bbox_max;

    const size_t max_leaf_size = 12;
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
        kdtree_node<FLOAT, D>*& root, size_t idx_from, size_t idx_to,
        std::array<FLOAT, D>& curbox_min, std::array<FLOAT, D>& curbox_max
    )
    {
        root = new kdtree_node<FLOAT, D>();
        if (idx_to - idx_from <= max_leaf_size) {
            // a leaf node; nothing more to do
            root->leaf_data.idx_from = idx_from;
            root->leaf_data.idx_to   = idx_to;
            Rprintf("%5d %5d %5d [%8f,%8f]-[%8f,%8f]\n",
                    (int)(idx_to-idx_from), (int)idx_from, (int)idx_to,
                    curbox_min[0], curbox_max[0], curbox_min[1], curbox_max[1]);
            return;
        }

        // cut by the dim of the greatest range
        size_t split_dim = 0;
        FLOAT dim_width = -infty;
        for (size_t u=0; u<D; ++u) {
            FLOAT cur_width = curbox_max[u]-curbox_min[u];
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

        FLOAT split_val = 0.5*(curbox_min[split_dim] + curbox_max[split_dim]);

        // partition data[:, split_dim] <= split_val, data[:, split_dim] > split_val
        FLOAT split_left_max = curbox_min[split_dim];
        FLOAT split_right_min = curbox_max[split_dim];
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

        GENIECLUST_ASSERT(data[idx_right*D+split_dim] <= split_val);
        GENIECLUST_ASSERT(data[idx_left*D+split_dim] > split_val);
        GENIECLUST_ASSERT(idx_left == idx_right+1);
        GENIECLUST_ASSERT(split_left_max <= split_val);
        GENIECLUST_ASSERT(split_right_min > split_val);

        root->intnode_data.split_dim = split_dim;
        root->intnode_data.split_left_max = split_left_max;
        root->intnode_data.split_right_min = split_right_min;

        FLOAT save;

        save = curbox_max[split_dim];
        curbox_max[split_dim] = split_left_max;
        build_tree(root->left, idx_from, idx_left, curbox_min, curbox_max);
        curbox_max[split_dim] = save;

        save = curbox_min[split_dim];
        curbox_min[split_dim] = split_right_min;
        build_tree(root->right, idx_left, idx_to, curbox_min, curbox_max);
        curbox_min[split_dim] = save;
    }


public:
    kdtree()
        : root(nullptr), data(nullptr), n(0), perm(0)
    {

    }

    kdtree(FLOAT* data, const size_t n)
        : root(nullptr), data(data), n(n), perm(n)
    {
        for (size_t i=0; i<n; ++i) perm[i] = i;

        for (size_t u=0; u<D; ++u) bbox_min[u] =  infty;
        for (size_t u=0; u<D; ++u) bbox_max[u] = -infty;

        for (size_t i=0; i<n; ++i) {
            for (size_t u=0; u<D; ++u) {
                if (data[i*D+u] > bbox_max[u])
                    bbox_max[u] = data[i*D+u];
                else if  (data[i*D+u] < bbox_min[u])
                    bbox_min[u] = data[i*D+u];
            }
        }

        std::array<FLOAT,D> curbox_min(bbox_min);
        std::array<FLOAT,D> curbox_max(bbox_max);
        build_tree(root, 0, n, curbox_min, curbox_max);
    }


    ~kdtree()
    {
        delete_tree(root);
    }


    inline size_t get_n() const { return n; }
    inline std::vector<size_t>& get_perm() { return perm; }
    inline FLOAT* get_data() { return data; }


    void kneighbours(size_t which, size_t* knn_ind, FLOAT* knn_dist, size_t k)
    {
        for (size_t i=0; i<k; ++i) knn_ind[i]  = which;
        for (size_t i=0; i<k; ++i) knn_dist[i] = infty;

    }
};


template <typename FLOAT, size_t D>
void kneighbours(
    kdtree<FLOAT, D>& tree,
    size_t* knn_ind,
    FLOAT* knn_dist,
    size_t k
) {
    size_t n = tree.get_n();
    const size_t* perm = tree.get_perm().data();

    for (size_t i=0; i<n; ++i) {
        size_t i_orig = perm[i];
        tree.kneighbours(i, knn_ind+k*i_orig, knn_dist+k*i_orig, k);
    }

    for (size_t i=0; i<n*k; ++i) {
        knn_ind[i] = perm[knn_ind[i]];
    }
}



#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::RObject test_kdtree(Rcpp::NumericMatrix X, int k)
{
    size_t n = X.nrow();
    size_t d = X.ncol();
    if (k < 1 || (size_t)k >= n) return R_NilValue;
    if (n < 1) return R_NilValue;

    std::vector<float> XC(n*d);
    size_t j = 0;
    for (size_t i=0; i<n; ++i)
        for (size_t u=0; u<d; ++u)
            XC[j++] = X(i, u);  // row-major

    std::vector<size_t> knn_ind(n*k);
    std::vector<float> knn_dist(n*k);

    if (d == 2) {
        kdtree<float, 2> tree(XC.data(), n);
        kneighbours<float, 2>(tree, knn_ind.data(), knn_dist.data(), k);
    }
    else
        return R_NilValue;  // TODO


    Rcpp::IntegerMatrix out_ind(n, k);
    Rcpp::NumericMatrix out_dist(n, k);
    size_t u = 0;
    for (size_t i=0; i<n; ++i) {
        for (int j=0; j<k; ++j) {
            out_ind(i, j) = knn_ind[u];
            out_dist(i, j) = knn_dist[u];
            u++;
        }
    }

    return Rcpp::List::create(out_ind, out_dist);
}


//CXX_DEFS="-O3 -march=native" Rscript -e 'Rcpp::sourceCpp("/home/gagolews/Python/genieclust/.devel/kdtree_test_rcpp.cpp")'


/*** R

set.seed(1234)
X <- matrix(rnorm(100*2), ncol=2)
test_kdtree(X, 5)

*/
