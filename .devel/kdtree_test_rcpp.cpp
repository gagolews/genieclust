#include <Rcpp.h>
#include <vector>
#include <array>



#ifndef GENIECLUST_ASSERT
#define __GENIECLUST_STR(x) #x
#define GENIECLUST_STR(x) __GENIECLUST_STR(x)

#define GENIECLUST_ASSERT(EXPR) { if (!(EXPR)) \
    throw std::runtime_error( "genieclust: Assertion " #EXPR " failed in "\
        __FILE__ ":" GENIECLUST_STR(__LINE__) ); }
#endif


template <typename FLOAT, size_t K>
struct kdtree_node {
    kdtree_node* left;
    kdtree_node* right;

    size_t idx_from;
    size_t idx_to;

    std::array<FLOAT,K> bbox_min;
    std::array<FLOAT,K> bbox_max;

    kdtree_node()
        : left(nullptr), right(nullptr), idx_from(0), idx_to(0)
    {

    }

    kdtree_node(
        size_t idx_from, size_t idx_to,
        const std::array<FLOAT,K>& bbox_min, const std::array<FLOAT,K>& bbox_max
    )
    :
        left(nullptr), right(nullptr),
        idx_from(idx_from), idx_to(idx_to),
        bbox_min(bbox_min), bbox_max(bbox_max)
    {

    }

    bool is_leaf() const { return left == nullptr && right == nullptr; }
};


template <typename FLOAT, size_t K>
class kdtree {
protected:
    kdtree_node<FLOAT, K>* root;
    FLOAT* data;  // ownership will be taken; a row-major n*K matrix
    const size_t n;
    std::vector<size_t> perm;

    const size_t max_leaf_size = 12;
    static constexpr FLOAT infty = std::numeric_limits<FLOAT>::infinity();


    void free_node(kdtree_node<FLOAT, K>*& root)
    {
        if (!root) return;

        free_node(root->left);
        free_node(root->right);
        delete root;
        root = nullptr;
    }


    void build(kdtree_node<FLOAT, K>*& root, size_t idx_from, size_t idx_to,
        std::array<FLOAT,K>& bbox_min, std::array<FLOAT,K>& bbox_max)
    {
        root = new kdtree_node<FLOAT, K>(idx_from, idx_to, bbox_min, bbox_max);
        if (idx_to - idx_from <= max_leaf_size) {
            Rprintf("%5d %5d %5d [%f,%f]-[%f,%f]\n", idx_to-idx_from, idx_from, idx_to, bbox_min[0], bbox_max[0], bbox_min[1], bbox_max[1]);
            return;  // a leaf node; nothing more to do
        }

        // cut by the dim of the greatest range
        size_t dim_index = 0;
        FLOAT dim_width = -infty;
        for (size_t i=0; i<K; ++i) {
            FLOAT cur_width = bbox_max[i]-bbox_min[i];
            if (cur_width > dim_width) {
                dim_width = cur_width;
                dim_index = i;
            }
        }

        if (dim_width == 0) return;  // this will be a "large" leaf (all points with the same coords)

        FLOAT dim_cut = 0.5*(bbox_min[dim_index] + bbox_max[dim_index]);

        // partition data[:, dim_index] <= dim_cut, data[:, dim_index] > dim_cut
        FLOAT dim_left_max = bbox_min[dim_index];
        FLOAT dim_right_min = bbox_max[dim_index];
        size_t idx_left = idx_from;
        size_t idx_right = idx_to-1;
        while (true) {
            while (data[idx_left*K+dim_index] <= dim_cut) {  // dim_cut < bbox_max[dim_index]
                if (data[idx_left*K+dim_index] > dim_left_max)
                    dim_left_max = data[idx_left*K+dim_index];
                idx_left++;
            }

            while (data[idx_right*K+dim_index] > dim_cut) {  // dim_cut > bbox_min[dim_index]
                if (data[idx_right*K+dim_index] < dim_right_min)
                    dim_right_min = data[idx_right*K+dim_index];
                idx_right--;
            }

            if (idx_left >= idx_right)
                break;

            std::swap(perm[idx_left], perm[idx_right]);
            for (size_t j=0; j<K; ++j)
                std::swap(data[idx_left*K+j], data[idx_right*K+j]);
        }

        GENIECLUST_ASSERT(data[idx_right*K+dim_index] <= dim_cut);
        GENIECLUST_ASSERT(data[idx_left*K+dim_index] > dim_cut);
        GENIECLUST_ASSERT(idx_left == idx_right+1);
        GENIECLUST_ASSERT(dim_left_max <= dim_cut);
        GENIECLUST_ASSERT(dim_right_min > dim_cut);

        FLOAT save;

        save = bbox_max[dim_index];
        bbox_max[dim_index] = dim_left_max;
        build(root->left, idx_from, idx_left, bbox_min, bbox_max);
        bbox_max[dim_index] = save;

        save = bbox_min[dim_index];
        bbox_min[dim_index] = dim_right_min;
        build(root->right, idx_left, idx_to, bbox_min, bbox_max);
        bbox_min[dim_index] = save;
    }


public:
    kdtree()
        : root(nullptr), data(nullptr), n(0), perm(0)
    {

    }

    kdtree(FLOAT* data, const size_t n)
        : root(nullptr), data(data), n(n), perm(n)
    {
        std::array<FLOAT,K> bbox_min;
        std::array<FLOAT,K> bbox_max;

        for (size_t i=0; i<n; ++i) perm[i] = i;

        for (size_t j=0; j<K; ++j) bbox_min[j] =  infty;
        for (size_t j=0; j<K; ++j) bbox_max[j] = -infty;

        for (size_t i=0; i<n; ++i) {
            for (size_t j=0; j<K; ++j) {
                if (data[i*K+j] > bbox_max[j])
                    bbox_max[j] = data[i*K+j];
                else if  (data[i*K+j] < bbox_min[j])
                    bbox_min[j] = data[i*K+j];
            }
        }


        build(root, 0, n, bbox_min, bbox_max);
    }

    ~kdtree()
    {
        free_node(root);
    }

};


// [[Rcpp::export]]
void test_kdtree(Rcpp::NumericMatrix X)
{
    size_t n = X.nrow();
    size_t k = X.ncol();
    if (X.ncol() != 2) return;

    std::vector<float> x(n*k);
    size_t u = 0;
    for (size_t i=0; i<n; ++i)
        for (size_t j=0; j<k; ++j)
            x[u++] = X(i, j);  // row-major

    kdtree<float, 2> t(x.data(), n);
}


//Rscript -e 'Rcpp::sourceCpp("kdtree_test_rcpp.cpp")'


/*** R

set.seed(123)
X <- matrix(rnorm(100*2), ncol=2)
test_kdtree(X)

*/
