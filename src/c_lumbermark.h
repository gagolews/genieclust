/*  Lumbermark: A Robust Divisive Clustering Algorithm based on Spanning Trees
 *
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


#define LUMBERMARK_UNSET (-1) // TODO
#define LUMBERMARK_NOISE  // TODO
#define LUMBERMARK_SKIP   // TODO


/*!  Lumbermark: A Robust Divisive Clustering Algorithm based on Spanning Trees
 *
 *   Detects a specific number of clusters
 *   as well as classifies points as noise/outliers.
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
                                * TODO [not implemented]: (-1, -1) denotes a no-edge and will be ignored
                                */


    Py_ssize_t n;        //<! number of points

    bool skip_leaves;    //<! should leaves be treated as noise/boundary points?

    std::vector<Py_ssize_t> deg;  //<! deg[i] denotes the degree of the i-th vertex

    std::vector<Py_ssize_t*> adj;  //<! adj[i] is a length-deg[i] array of nodes adjacent to the i-th vertex; adj's length is n+1 (adj[n] is a sentinel element)
    std::vector<Py_ssize_t> _adjdata;  //<! the underlying data buffer for adj


public:
    CLumbermark() : CLumbermark(NULL, NULL, 0) { }

    CLumbermark(const T* mst_d, const Py_ssize_t* mst_i, Py_ssize_t n)
        : mst_d(mst_d), mst_i(mst_i), n(n)
    {
        if (n == 0) return;

        for (Py_ssize_t i=0; i<n-1; ++i) {
            GENIECLUST_ASSERT(mst_i[2*i+0] >= 0);  // TODO: add to cutlist
            GENIECLUST_ASSERT(mst_i[2*i+1] >= 0);  // TODO: add to cutlist

            // check if edge weights are sorted increasingly
            GENIECLUST_ASSERT(i == 0 || mst_d[i-1] <= mst_d[i])
        }

        deg.resize(n);
        _adjdata.resize(2*(n-1));
        adj.resize(n);

        // set up this->deg:
        Cget_graph_node_degrees(mst_i, n-1, n, /*out:*/this->deg.data());

        Cget_graph_node_adjlists(
            mst_i, n-1, n, this->deg.data(),
            /*out:*/this->_adjdata.data(), /*out:*/this->adj.data()
        );

        // TODO....
    }


    /*! Run the Lumbermark algorithm
     *
     * @param n_clusters number of clusters to find
     * @param min_cluster_size Minimal cluster size
     * @param min_cluster_factor Output cluster sizes won't be smaller than
     *    min_cluster_factor/n_clusters*n_points (noise points excluding)
     * @param skip_leaves Marks leaves and degree-2 nodes incident
     *    to cut edges as noise and does not take them into account whilst
     *    determining the partition sizes.  Noise markers can be fetched
     *    separately. Noise points will still be assigned to nearest clusters.
     *
     * @return number of clusters detected (can be smaller than the requested one)
     */
    Py_ssize_t compute(
        Py_ssize_t n_clusters, Py_ssize_t min_cluster_size,
        T min_cluster_factor, bool skip_leaves
    ) {
        if (n_clusters < 1)
            throw std::domain_error("n_clusters must be >= 1");

        // this->results = typename CGenieBase<T>::CGenieResult(this->n,
        //     this->noise_count, n_clusters);

        // ... TODO ....

        return 0;
    }


    /*! Set res[i] to true iff skip_leaves is true and the i-th point
     *  is a noise/boundary node, i.e., is a leaf in the spanning tree
     *  or is of degree two but is adjacent to one of the cut edges.
     *
     *  @param res [out] array of length n
     */
    void get_is_noise(int* res) const
    {
        // TODO

        for (Py_ssize_t i=0; i<n; ++i) {
            res[i] = (this->skip_leaves && this->deg[i] <= 1);
        }

        // ... TODO .... // or is of degree two but is adjacent to one of the cut edges.
    }



    /*! Propagate res with clustering results.
     *
     *  All points, even noise ones, are assigned a cluster.
     *
     *  @param res [out] array of length n
     */
    void get_labels(Py_ssize_t* res)
    {
        // TODO ...
    }


    /*! Get the cut edges of the spanning tree that lead to n_clusters
     *   connected components.
     *
     *  @param res [out] array of length n_clusters-1
     */
    void get_links(Py_ssize_t* res)
    {
        // TODO ...
    }
};


#endif
