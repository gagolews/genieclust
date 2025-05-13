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

#include <limits>
#define LUMBERMARK_UNSET     (std::numeric_limits<Py_ssize_t>::min())
#define LUMBERMARK_CUTEDGE   (LUMBERMARK_UNSET+1)
#define LUMBERMARK_NOISEEDGE (LUMBERMARK_UNSET+2)


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
template <class floatT>
class CLumbermark {
protected:

    const floatT* mst_d;  //<! n-1 edge weights, sorted increasingly

    /*! n-1 edges of the spanning tree given by c_contiguous (n-1)*2 indices;
     ** TODO [unimplemented]: (-1, -1) denotes a no-edge and will be ignored */
    const Py_ssize_t* mst_i;


    Py_ssize_t n;        //<! number of points

    bool skip_leaves;    //<! should leaves be treated as noise/boundary points?

    std::vector<Py_ssize_t> deg;  //<! deg[i] denotes the degree of the i-th vertex

    std::vector<Py_ssize_t*> inc;  //<! inc[i] is a length-deg[i] array of edges incident on the i-th vertex; inc's length is n+1 (inc[n] is a sentinel element)
    std::vector<Py_ssize_t> _incdata;  //<! the underlying data buffer for inc


    // auxiliary data for generating clustering results:
    std::vector<Py_ssize_t> mst_labels;  //<! edge labels, size n-1
    std::vector<Py_ssize_t> labels;  //<! node labels, size n, in 1..n_clusters and -1..-n_clusters (noise/boundary points)
    std::vector<Py_ssize_t> mst_cutsizes;  //<!  size (n-1)*2, each pair gives the sizes of the clusters that are formed when we cut out the corresponding edge
    std::vector<Py_ssize_t> cluster_sizes; //<!  size n_clusters
    std::vector<Py_ssize_t> cut_edges; //<!  size n_clusters-1


    /*! vertex visitor (0th pass):
     *  going from v, visits w and then all its neighbours, mst_i[e,:] = {v,w};
     *  marks edges incident on leaves as noise edges,
     *  checks if the graph is acyclic
     */
    void visit0(Py_ssize_t v, Py_ssize_t e)
    {
        Py_ssize_t iv = (Py_ssize_t)(mst_i[2*e+1]==v);
        Py_ssize_t w = mst_i[2*e+(1-iv)];
        //     so v == mst_i[2*e+iv]

        GENIECLUST_ASSERT(e >= 0 && e < n-1);
        GENIECLUST_ASSERT(v >= 0 && v < n);
        GENIECLUST_ASSERT(w >= 0 && w < n);
        GENIECLUST_ASSERT(labels[v] == 0);
        GENIECLUST_ASSERT(mst_labels[e] == LUMBERMARK_UNSET);
        GENIECLUST_ASSERT(labels[w] == LUMBERMARK_UNSET);

        labels[w] = 0;

        if (skip_leaves && deg[w] == 1)
            mst_labels[e] = LUMBERMARK_NOISEEDGE;
        else
            mst_labels[e] = 0;

        for (Py_ssize_t* e2 = inc[w]; e2 != inc[w+1]; e2++) {
            if (*e2 != e) visit0(w, *e2);
        }

    }

    /*! vertex visitor (kth pass):
     *  going from v, visits w and then all its neighbours, mst_i[e,:] = {v,w};
     *  marks them as members of the c-th cluster. */
    Py_ssize_t visit(Py_ssize_t v, Py_ssize_t e, Py_ssize_t c)
    {
        if (mst_labels[e] == LUMBERMARK_CUTEDGE)
            return 0;

        Py_ssize_t iv = (Py_ssize_t)(mst_i[2*e+1]==v);
        Py_ssize_t w = mst_i[2*e+(1-iv)];
        //     so v == mst_i[2*e+iv]

        // don't count vertices incident to noise edges
        Py_ssize_t tot = (mst_labels[e] != LUMBERMARK_NOISEEDGE);

        GENIECLUST_ASSERT(c > 0);
        if (mst_labels[e] != LUMBERMARK_NOISEEDGE) {
            labels[w] = c;
            mst_labels[e] = c;
        }
        else {
            labels[w] = -c;
            // mst_labels[e] = LUMBERMARK_NOISEEDGE;  // already is
        }

        for (Py_ssize_t* e2 = inc[w]; e2 != inc[w+1]; e2++) {
            if (*e2 != e) tot += visit(w, *e2, c);
        }

        mst_cutsizes[2*e+(1-iv)] = tot;
        mst_cutsizes[2*e+iv] = LUMBERMARK_UNSET;  // t.b.d. later

        return tot;
    }




public:
    CLumbermark() : CLumbermark(NULL, NULL, 0, false) { }

    CLumbermark(const floatT* mst_d, const Py_ssize_t* mst_i, Py_ssize_t n, bool skip_leaves) :
        mst_d(mst_d), mst_i(mst_i), n(n), skip_leaves(skip_leaves)
    {
        if (n == 0) return;

        for (Py_ssize_t i=0; i<n-1; ++i) {
            GENIECLUST_ASSERT(mst_i[2*i+0] >= 0);  // TODO: add to cutlist
            GENIECLUST_ASSERT(mst_i[2*i+1] >= 0);  // TODO: add to cutlist

            // check if edge weights are sorted increasingly
            GENIECLUST_ASSERT(i == 0 || mst_d[i-1] <= mst_d[i]);
        }

        deg.resize(n);
        _incdata.resize(2*(n-1));
        inc.resize(n);

        // set up this->deg:
        Cget_graph_node_degrees(mst_i, n-1, n, /*out:*/this->deg.data());

        for (Py_ssize_t i=0; i<n-1; ++i) {
            GENIECLUST_ASSERT(deg[i] > 0);  // TODO: won't hold if the graph is not connected
        }

        Cget_graph_node_inclists(
            mst_i, n-1, n, this->deg.data(),
            /*out:*/this->_incdata.data(), /*out:*/this->inc.data()
        );
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
        floatT min_cluster_factor
    ) {
        GENIECLUST_ASSERT(n > 2);

        if (n_clusters < 1)
            throw std::domain_error("n_clusters must be >= 1");

        Py_ssize_t n_skip = 0;
        if (skip_leaves) {
            // assume that nodes incident to cut edges will be marked as leaves:
            n_skip += (n_clusters-1)*2;

            // n_skip += n_leaves;
            for (Py_ssize_t i=0; i<n; ++i)
                if (deg[i] <= 1) n_skip++;
        }
        // else {  // TODO: not true if the graph is not connected
        //     for (Py_ssize_t i=0; i<n; ++i)
        //         if (deg[i] <= 0) n_skip++;
        // }


        min_cluster_size = std::max(
            min_cluster_size,
            (Py_ssize_t)(min_cluster_factor*(n-n_skip)/n_clusters)
        );

        labels.resize(n);
        for (Py_ssize_t i=0; i<n; ++i)
            labels[i] = LUMBERMARK_UNSET;

        mst_labels.resize(n-1);
        for (Py_ssize_t i=0; i<n-1; ++i)
            mst_labels[i] = LUMBERMARK_UNSET;

        // ensure that the graph is acyclic:
        // visit all nodes starting from a non-leaf and incident to edge 0,
        // and mark edges as noise/boundary is skip_leaves is true
        Py_ssize_t v = mst_i[2*0+0];
        if (deg[v] <= 1) {
            v = mst_i[2*0+1];
            GENIECLUST_ASSERT(deg[v] > 1);
        }

        labels[v] = 0;
        for (Py_ssize_t* e2 = inc[v]; e2 != inc[v+1]; e2++) {
            visit0(v, *e2);
        }

        // ensure all vertices and edges are reachable:
        for (Py_ssize_t i=0; i<n;   ++i)
            GENIECLUST_ASSERT(labels[i] == 0);
        for (Py_ssize_t i=0; i<n-1; ++i)
            GENIECLUST_ASSERT(mst_labels[i] != LUMBERMARK_UNSET);


        mst_cutsizes.resize((n-1)*2);
        cluster_sizes.resize(n_clusters);
        cut_edges.resize(n_clusters-1);

        // todo: mark as 1 in visit0?

        Py_ssize_t iter = 0;
        // while (iter < n_clusters)
        // {
        //
        //     iter++;
        // }

        return iter;
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
