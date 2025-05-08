/*  Lumbermark: A Robust Divisive Clustering Algorithm based on Spanning Trees
 *
 *  Detects a specific number of clusters
 *  as well as classifies points as noise/outliers.
 *
 *  Copyleft (C) 2025-2025, Marek Gagolewski <https://www.gagolewski.com>
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


#ifndef __c_lumbermark_h
#define __c_lumbermark_h

#include "c_common.h"
#include <algorithm>
#include <vector>
#include <deque>
#include <cmath>

#include "c_disjoint_sets.h"
#include "c_preprocess.h"



/*!  Lumbermark: A Robust Divisive Clustering Algorithm based on Spanning Trees
 *
 *
 *   References
 *   ===========
 *
 *   Gagolewski M., TODO, 2025
 */
template <class T>
class CLumbermark {
protected:

    const T* mst_d;            //<! n-1 edge weights, sorted increasingly

    const Py_ssize_t* mst_i;   /*!< n-1 edges of the spanning tree,
                          * given by c_contiguous (n-1)*2 indices;
                          * (-1, -1) denotes a no-edge and will be ignored
                          */


    Py_ssize_t n;        //<! number of points

    bool skip_leaves;    //<! should leaves be treated as noise/boundary points?

    std::vector<Py_ssize_t> deg; //<! deg[i] denotes the degree of the i-th vertex




public:
    CLumbermark(const T* mst_d, const Py_ssize_t* mst_i, Py_ssize_t n, bool skip_leaves=false)
        : mst_d(mst_d), mst_i(mst_i), n(n), skip_leaves(skip_leaves), deg(n)
    {
        for (Py_ssize_t i=0; i<n-1; ++i) {
            if (mst_i[i] < 0 || mst_i[i] < 0) {
                continue;
            }
            else {
                GENIECLUST_ASSERT(i == 0 || mst_d[i-1] <= mst_d[i])
            }
        }

        // set up this->deg:
        Cget_graph_node_degrees(mst_i, n-1, n, this->deg.data());

        // TODO
    }


    /*! Run the Lumbermark algorithm
     *
     * @param n_clusters number of clusters to find
     */
    void apply_lumbermark(Py_ssize_t n_clusters)
    {
        if (n_clusters < 1)
            throw std::domain_error("n_clusters must be >= 1");

        // this->results = typename CGenieBase<T>::CGenieResult(this->n,
        //     this->noise_count, n_clusters);

        // ... TODO ....
    }


    /*! Set res[i] to true if the i-th point is a noise one.
     *
     *  Makes sense only if noise_leaves==true
     *
     *  @param res [out] array of length n
     */
    void get_noise_status(bool* res) const
    {
        // ... TODO ....
        // for (Py_ssize_t i=0; i<n; ++i) {
        //     res[i] = (this->noise_leaves && this->deg[i] <= 1);
        // }
    }



    /*! Propagate res with clustering results.
     *
     * Noise points get cluster id of -1. TODO ...
     *
     *
     * @return number of clusters detected (not including the noise cluster;
     * can be less than n_clusters)
     */
    Py_ssize_t get_labels(Py_ssize_t* res)
    {
        // TO DO ...
        return 0;
    }
};


#endif
