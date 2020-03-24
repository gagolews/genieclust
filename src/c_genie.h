/*  The Genie+ Clustering Algorithm
 *
 *  Copyright (C) 2018-2020 Marek Gagolewski (https://www.gagolewski.com)
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *  contributors may be used to endorse or promote products derived from this
 *  software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 *  OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef __c_genie_h
#define __c_genie_h

#include <stdexcept>
#include <algorithm>
#include <vector>
#include <deque>
#include <cassert>
#include <cmath>

#include "c_gini_disjoint_sets.h"
#include "c_int_dict.h"
#include "c_preprocess.h"






/*!  The Genie++ Hierarchical Clustering Algorithm
 *
 *   The Genie algorithm (Gagolewski et al., 2016) links two clusters
 *   in such a way that a chosen economic inequity measure
 *   (here, the Gini index) of the cluster sizes does not increase drastically
 *   above a given threshold. The method most often outperforms
 *   the Ward or average linkage, k-means, spectral clustering,
 *   DBSCAN, Birch and others in terms of the clustering
 *   quality on benchmark data while retaining the speed of the single
 *   linkage algorithm.
 *
 *   This is a re-implementation of the original (Gagolewski et al., 2016)
 *   algorithm. First of all, given a pre-computed minimum spanning tree (MST),
 *   it only requires amortised O(n sqrt(n))-time.
 *   Additionally, MST leaves can be
 *   marked as noise points (if `noise_leaves==True`). This is useful,
 *   if the Genie algorithm is applied on the MST with respect to
 *   the HDBSCAN-like mutual reachability distance.
 *
 *   Note that the input graph might be disconnected (spanning forest,
 *   but here we will call it MST anyway) - it must be acyclic though.
 *
 *
 *   References
 *   ===========
 *
 *   Gagolewski M., Bartoszuk M., Cena A.,
 *   Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
 *   Information Sciences 363, 2016, pp. 8-23. doi:10.1016/j.ins.2016.05.003
 */
template <class T>
class CGenie {
protected:
    ssize_t* mst_i;   //<! n-1 edges of the MST (given by c_contiguous (n-1)*2 indices)
    T* mst_d;         //<! n-1 edge weights
    ssize_t n;        //<! number of points
    bool noise_leaves;//<! mark leaves as noise points?

    std::vector<ssize_t> deg; //<! deg[i] denotes the degree of the i-th vertex

    ssize_t noise_count; //<! now many noise points are there (leaves)
    std::vector<ssize_t> denoise_index; //<! which noise point is it?
    std::vector<ssize_t> denoise_index_rev; //!< reverse look-up for denoise_index


// TODO: mst_i[] < 0 !!!!!!!!!!!!!!!!!!!!!
// TODO:    separate GIc subclass ???


    /*! When the Genie correction is on, some MST edges will be chosen
     * in non-consecutive order. An array-based skiplist will speed up
     * searching within the to-be-consumed edges. Also, if there are
     * noise points, then the skiplist allows the algorithm
     * to naturally ignore edges that connect the leaves. */
    void mst_skiplist_init(CIntDict<ssize_t>* mst_skiplist) {
        // start with a list that skips all edges that lead to noise points
        mst_skiplist->clear();
        for (ssize_t i=0; i<n-1; ++i) {
            if (!noise_leaves || (deg[mst_i[i*2+0]]>1 && deg[mst_i[i*2+1]]>1))
                    (*mst_skiplist)[i] = i;
        }
    }


    void do_genie(CGiniDisjointSets* ds, CIntDict<ssize_t>* mst_skiplist,
        ssize_t n_clusters, double gini_threshold)
    {
        if (n-noise_count-n_clusters <= 0) {
            throw std::runtime_error("The requested number of clusters \
                is too large with this many detected noise points");
        }

        ssize_t lastidx = mst_skiplist->get_key_min();
        ssize_t lastm = 0; // last minimal cluster size
        for (ssize_t i=0; i<n-noise_count-n_clusters; ++i) {

            // determine the pair of vertices to merge
            ssize_t i1;
            ssize_t i2;

            if (ds->get_gini() > gini_threshold) {
                // the Genie correction for inequity of cluster sizes
                ssize_t m = ds->get_smallest_count();
                if (m != lastm || lastidx < mst_skiplist->get_key_min()) {
                    // need to start from the beginning of the MST skiplist
                    lastidx = mst_skiplist->get_key_min();
                }
                // else reuse lastidx

                // find the MST edge connecting a cluster of the smallest size
                // with another one
                while (ds->get_count(denoise_index_rev[mst_i[2*lastidx+0]]) != m
                    && ds->get_count(denoise_index_rev[mst_i[2*lastidx+1]]) != m)
                {
                    lastidx = mst_skiplist->get_key_next(lastidx);
                    assert(lastidx < n-1);
                }

                i1 = mst_i[2*lastidx+0];
                i2 = mst_i[2*lastidx+1];
                ssize_t delme = lastidx;
                lastidx = mst_skiplist->get_key_next(lastidx);
                mst_skiplist->erase(delme); // O(1)
                lastm = m;
            }
            else { // single linkage-like
                ssize_t curidx = mst_skiplist->pop_key_min();
                i1 = mst_i[2*curidx+0];
                i2 = mst_i[2*curidx+1];
            }

            ds->merge(denoise_index_rev[i1], denoise_index_rev[i2]);
        }
    }

    /*! Propagate res with clustering results
     *
     * @param ds disjoint sets representing the partition
     * @param res [out] array of length n
     */
    void get_labels(CDisjointSets* ds, ssize_t* res) {
        std::vector<ssize_t> res_cluster_id(n, -1);
        ssize_t c = 0;
        for (ssize_t i=0; i<n; ++i) {
            if (denoise_index_rev[i] >= 0) {
                // a non-noise point
                ssize_t j = denoise_index[ds->find(denoise_index_rev[i])];
                // assert 0 <= j < n
                if (res_cluster_id[j] < 0) {
                    res_cluster_id[j] = c;
                    ++c;
                }
                res[i] = res_cluster_id[j];
            }
            else {
                // a noise point
                res[i] = -1;
            }
        }
    }

    /*! Run the Genie+ algorithm with different Gini index
     *  thresholds and determine the intersection of all the resulting
     *  partitions; for this, we need the union of the set of MST edges that
     *  were left "unmerged"
     *
     * @param n_clusters number of clusters to look for in Genie run
     * @param gini_thresholds array of floats in [0,1]
     * @param n_thresholds size of gini_thresholds
     *
     * @return indexes of MST edges that were left unused by at least
     * one Genie algorithm run; this gives the intersection of partitions.
     *
     * If n_thresholds is 0 or requested n_clusters is too large,
     * all non-noise edges are set as unused.
     */
    std::vector<ssize_t> get_intersection_of_genies(ssize_t n_clusters,
        double* gini_thresholds, ssize_t n_thresholds)
    {
        std::vector<ssize_t> unused_edges;
        if (n_thresholds <= 0 || n_clusters >= n-noise_count) {
            // all edges unused -> will start from n singletons
            if (!noise_leaves) {
                unused_edges.resize(n-1);
                for (ssize_t i=0; i<n-1; ++i)
                    unused_edges[i] = i;
            }
            else {
                for (ssize_t i=0; i<n-1; ++i) {
                    ssize_t i1 = mst_i[2*i+0];
                    ssize_t i2 = mst_i[2*i+1];
                    if (deg[i1] > 1 && deg[i2] > 1)
                        unused_edges.push_back(i);
                }
            }
            unused_edges.push_back(n-1);  // sentinel
            return unused_edges;
            // EOF.
        }

        for (ssize_t i=0; i<n_thresholds; ++i) {
            double gini_threshold = gini_thresholds[i];
            CGiniDisjointSets ds(n-noise_count);
            CIntDict<ssize_t> mst_skiplist(n-1);
            mst_skiplist_init(&mst_skiplist);
            do_genie(&ds, &mst_skiplist, n_clusters, gini_threshold);

            // start where do_genie concluded
            while (!mst_skiplist.empty())
                unused_edges.push_back(mst_skiplist.pop_key_min());
        }

        // let unused_edges = sort(unique(unused_edges))
        unused_edges.push_back(n-1); // sentinel
        std::sort(unused_edges.begin(), unused_edges.end());
        // sorted, but some might not be unique, so let's remove dups
        ssize_t k = 0;
        for (ssize_t i=1; i<(ssize_t)unused_edges.size(); ++i) {
            if (unused_edges[i] != unused_edges[k]) {
                k++;
                unused_edges[k] = unused_edges[i];
            }
        }
        unused_edges.resize(k+1);
        return unused_edges;
    }


public:
    CGenie(T* mst_d, ssize_t* mst_i, ssize_t n, bool noise_leaves)
        : deg(n), denoise_index(n), denoise_index_rev(n)
    {
        this->mst_d = mst_d;
        this->mst_i = mst_i;
        this->n = n;
        this->noise_leaves = noise_leaves;

        for (ssize_t i=1; i<n-1; ++i)
            if (mst_d[i-1] > mst_d[i])
                throw std::domain_error("mst_d unsorted");

        // set up this->deg:
        Cget_graph_node_degrees(mst_i, n-1, n, this->deg.data());

        // Create the non-noise points' translation table (for GiniDisjointSets)
        // and count the number of noise points
        if (noise_leaves) {
            this->noise_count = 0;
            ssize_t j = 0;
            for (ssize_t i=0; i<n; ++i) {
                if (deg[i] == 1) { // a leaf
                    ++noise_count;
                    denoise_index_rev[i] = -1;
                }
                else {             // a non-leaf
                    denoise_index[j] = i;
                    denoise_index_rev[i] = j;
                    ++j;
                }
            }
            if (!(noise_count >= 2))
                throw std::runtime_error("ASSERT FAIL (noise_count >= 2)");
            if (!(j + noise_count == n))
                throw std::runtime_error("ASSERT FAIL (j + noise_count == n)");
        }
        else { // there are no noise points
            this->noise_count = 0;
            for (ssize_t i=0; i<n; ++i) {
                denoise_index[i]     = i; // identity
                denoise_index_rev[i] = i;
            }
        }
    }

    CGenie() : CGenie(NULL, NULL, 0, false) { }


    /*! Run the Genie+ algorithm
     *
     * @param n_clusters number of clusters to find
     * @param gini_threshold the Gini index threshold
     * @param res [out] array of length n, will give cluster labels
     */
    void apply_genie(ssize_t n_clusters, double gini_threshold, ssize_t* res)
    {
        CGiniDisjointSets ds(n-noise_count);
        CIntDict<ssize_t> mst_skiplist(n-1);
        mst_skiplist_init(&mst_skiplist);
        do_genie(&ds, &mst_skiplist, n_clusters, gini_threshold);
        get_labels(&ds, res);
    }



    /*! Run the GIC (Genie+Information Criterion) algorithm
     *  by Anna Cena
     *
     * References
     * ==========
     *
     * [1] Cena A., Adaptive hierarchical clustering algorithms based on
     * data aggregation methods, PhD Thesis, Systems Research Institute,
     * Polish Academy of Sciences 2018.
     *
     * [2] Mueller A., Nowozin S., Lampert C.H., Information Theoretic
     * Clustering using Minimum Spanning Trees, DAGM-OAGM 2012.
     *
     *
     * @param n_clusters number of clusters to find
     * @param add_clusters number of additional clusters to work
     *     with internally
     * @param n_features number of features (can be fractional)
     * @param gini_thresholds array of size n_thresholds
     * @param n_thresholds size of gini_thresholds
     * @param res [out] array of length n, will give cluster labels
     */
    void apply_gic(ssize_t n_clusters,
                   ssize_t add_clusters,
                   double n_features,
                   double* gini_thresholds,
                   ssize_t n_thresholds,
                   ssize_t* res)
    {
        assert(add_clusters>=0);
        assert(n_clusters>=1);
        assert(n_thresholds>=0);

        std::vector<ssize_t> unused_edges = get_intersection_of_genies(
                n_clusters+add_clusters, gini_thresholds, n_thresholds
        );
        // note that unused_edges do not include noise edges


        // first generate ds representing the intersection of the partitions
        CDisjointSets ds(n-noise_count);
        ssize_t cur_unused_edges = 0;
        std::vector<ssize_t> cluster_sizes(n-noise_count, 1);
        std::vector<T> cluster_d_sums(n-noise_count, (T)0.0);
        for (ssize_t i=0; i<n-1; ++i) {
            ssize_t i1 = mst_i[2*i+0];
            ssize_t i2 = mst_i[2*i+1];
            if (noise_leaves && (deg[i1] == 1 || deg[i2] == 1)) {
                // noise edge, ignore
                continue;
            }

            assert(i<=unused_edges[cur_unused_edges]);

            if (unused_edges[cur_unused_edges] == i) {
                // ignore current edge and advance to the next unused edge
                cur_unused_edges++;
                continue;
            }

            i1 = ds.find(denoise_index_rev[i1]);
            i2 = ds.find(denoise_index_rev[i2]);
            if (i1 > i2) std::swap(i1, i2);
            ds.merge(i1, i2);
            // new parent node is i1
            cluster_sizes[i1]  += cluster_sizes[i2];
            cluster_d_sums[i1] += cluster_d_sums[i2]+mst_d[i];
            cluster_sizes[i2]   = 0;
            cluster_d_sums[i2]  = INFTY;
        }

        /*  The objective function to MINIMISE is
            sum_{i in ds.parents()} cluster_sizes[i] * (
                n_features     * log cluster_sizes[i]
              -(n_features-1)  * log cluster_d_sums[i]
            )
        */

        assert(unused_edges[unused_edges.size()-1] == n-1); // sentinel
        ssize_t num_unused_edges = unused_edges.size()-1; // ignore sentinel
        assert(num_unused_edges+1 == ds.get_k());

        while (ds.get_k() != n_clusters) {
            ssize_t min_which = -1;
            double  min_obj = INFTY;
            for (ssize_t j=0; j<num_unused_edges; ++j) {
                ssize_t i = unused_edges[j];
                ssize_t i1 = mst_i[2*i+0];
                ssize_t i2 = mst_i[2*i+1];
                i1 = ds.find(denoise_index_rev[i1]);
                i2 = ds.find(denoise_index_rev[i2]);
                if (i1 > i2) std::swap(i1, i2);

                assert(i1 != i2);

                // singletons will be merged first
                if (cluster_d_sums[i1] < 1e-12 || cluster_d_sums[i2] < 1e-12) {
                    min_which = j;
                    break;
                }

                // compute difference in obj
                double cur_obj = (cluster_sizes[i1]+cluster_sizes[i2])*(
                    n_features*log(cluster_d_sums[i1]+cluster_d_sums[i2]+mst_d[i])
                  -(n_features-1.0)*log(cluster_sizes[i1]+cluster_sizes[i2])
                );
                cur_obj -= cluster_sizes[i1]*(
                    n_features*log(cluster_d_sums[i1])
                  -(n_features-1.0)*log(cluster_sizes[i1])
                );
                cur_obj -= cluster_sizes[i2]*(
                    n_features*log(cluster_d_sums[i2])
                  -(n_features-1.0)*log(cluster_sizes[i2])
                );

                assert(std::isfinite(cur_obj));
                if (cur_obj < min_obj) {
                    min_obj = cur_obj;
                    min_which = j;
                }
            }

            assert(min_which >= 0 && min_which < num_unused_edges);
            ssize_t i = unused_edges[min_which];
            ssize_t i1 = mst_i[2*i+0];
            ssize_t i2 = mst_i[2*i+1];
            i1 = ds.find(denoise_index_rev[i1]);
            i2 = ds.find(denoise_index_rev[i2]);
            if (i1 > i2) std::swap(i1, i2);

            ds.merge(i1, i2);
            // new parent node is i1

            cluster_sizes[i1]  += cluster_sizes[i2];
            cluster_d_sums[i1] += cluster_d_sums[i2]+mst_d[i];
            cluster_sizes[i2] = 0;
            cluster_d_sums[i2] = INFTY;

            unused_edges[min_which] = unused_edges[num_unused_edges-1];
            num_unused_edges--;
        }

        get_labels(&ds, res);
    }
};


#endif
