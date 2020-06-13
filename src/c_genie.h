/*  The Genie++ Clustering Algorithm
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

#include "c_common.h"
#include <algorithm>
#include <vector>
#include <deque>
#include <cmath>

#include "c_gini_disjoint_sets.h"
#include "c_int_dict.h"
#include "c_preprocess.h"




/*!  Base class for CGenie and CGIc
 */
template <class T>
class CGenieBase {
protected:

    /*!  Stores the clustering result as obtained by
     *   CGenie::apply_genie() or CGIc::apply_gic()
     */
    struct CGenieResult {

        CGiniDisjointSets ds; /*!< ds at the last iteration, it;
                               * use denoise_index to obtain the final partition
                               */
        std::vector<ssize_t> links; //<! links[..] = index of merged mst_i
        ssize_t it;                 //<! number of merges performed
        ssize_t n_clusters;         //<! maximal number of clusters requested

        CGenieResult() { }

        CGenieResult(ssize_t n, ssize_t noise_count, ssize_t n_clusters):
            ds(n-noise_count), links(n-1, -1), it(0), n_clusters(n_clusters) { }

    };



    ssize_t* mst_i;   /*!< n-1 edges of the MST,
                       * given by c_contiguous (n-1)*2 indices;
                       * (-1, -1) denotes a no-edge and will be ignored
                       */
    T* mst_d;         //<! n-1 edge weights
    ssize_t n;        //<! number of points
    bool noise_leaves;//<! mark leaves as noise points?

    std::vector<ssize_t> deg; //<! deg[i] denotes the degree of the i-th vertex

    ssize_t noise_count; //<! now many noise points are there (leaves)
    std::vector<ssize_t> denoise_index; //<! which noise point is it?
    std::vector<ssize_t> denoise_index_rev; //!< reverse look-up for denoise_index

    CGenieResult results;



    /*! When the Genie correction is on, some MST edges will be chosen
     * in a non-consecutive order. An array-based skiplist will speed up
     * searching within the to-be-consumed edges. Also, if there are
     * noise points, then the skiplist allows the algorithm
     * to naturally ignore edges that connect the leaves. */
    void mst_skiplist_init(CIntDict<ssize_t>* mst_skiplist) {
        // start with a list that skips all edges that lead to noise points
        mst_skiplist->clear();
        for (ssize_t i=0; i<this->n-1; ++i) {
            GENIECLUST_ASSERT(this->mst_i[i*2+0] < this->n && this->mst_i[i*2+1] < this->n);
            if (this->mst_i[i*2+0] < 0 || this->mst_i[i*2+1] < 0)
                continue; // a no-edge -> ignore
            if (!this->noise_leaves ||
                    (this->deg[this->mst_i[i*2+0]]>1 && this->deg[this->mst_i[i*2+1]]>1)) {
                (*mst_skiplist)[i] = i; /*only the key is important, not the value*/
            }
        }
    }




    /** internal, used by get_labels(n_clusters, res) */
    ssize_t get_labels(CGiniDisjointSets* ds, ssize_t* res) {
        std::vector<ssize_t> res_cluster_id(n, -1);
        ssize_t c = 0;
        for (ssize_t i=0; i<n; ++i) {
            if (this->denoise_index_rev[i] >= 0) {
                // a non-noise point
                ssize_t j = this->denoise_index[
                                ds->find(this->denoise_index_rev[i])
                            ];
                if (res_cluster_id[j] < 0) {
                    // new cluster
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

        return c;
    }



public:
    CGenieBase(T* mst_d, ssize_t* mst_i, ssize_t n, bool noise_leaves)
        : deg(n), denoise_index(n), denoise_index_rev(n)
    {
        this->mst_d = mst_d;
        this->mst_i = mst_i;
        this->n = n;
        this->noise_leaves = noise_leaves;

        for (ssize_t i=1; i<n-1; ++i)
            if (mst_i[i] >= 0 && mst_i[i] >= 0 && mst_d[i-1] > mst_d[i])
                throw std::domain_error("mst_d unsorted");

        // set up this->deg:
        Cget_graph_node_degrees(mst_i, n-1, n, this->deg.data());

        // Create the non-noise points' translation table (for GiniDisjointSets)
        // and count the number of noise points
        if (noise_leaves) {
            noise_count = 0;
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
            GENIECLUST_ASSERT(noise_count >= 2);
            GENIECLUST_ASSERT(j + noise_count == n);
        }
        else { // there are no noise points
            this->noise_count = 0;
            for (ssize_t i=0; i<n; ++i) {
                denoise_index[i]     = i; // identity
                denoise_index_rev[i] = i;
            }
        }
    }


    /*! There can be at most n-noise_count singleton clusters
     *  in the hierarchy. */
    ssize_t get_max_n_clusters() const {
        return this->n - this->noise_count;
    }


    /*! Propagate res with clustering results.
     *
     * Noise points get cluster id of -1.
     *
     * @param n_clusters maximal number of clusters to find
     * @param res [out] c_contiguous array of length n
     *
     * @return number of clusters detected (not including the noise cluster;
     * can be less than n_clusters)
     */
    ssize_t get_labels(ssize_t n_clusters, ssize_t* res) {
        if (this->results.ds.get_n() <= 0)
            throw std::runtime_error("Apply the clustering procedure first.");

        if (n_clusters <= this->results.n_clusters) {
            // use this->results.ds -- from the final iteration
            return this->get_labels(&(this->results.ds), res);
        }
        else {
            CGiniDisjointSets ds(this->get_max_n_clusters());
            for (ssize_t it=0; it<this->get_max_n_clusters() - n_clusters; ++it) {
                ssize_t j = (this->results.links[it]);
                ssize_t i1 = this->mst_i[2*j+0];
                ssize_t i2 = this->mst_i[2*j+1];
                GENIECLUST_ASSERT(i1 >= 0 && i2 >= 0)
                ds.merge(this->denoise_index_rev[i1], this->denoise_index_rev[i2]);
            }
            return this->get_labels(&ds, res);
        }
    }


    /*! Propagate res with clustering results -
     *  all partitions from cardinality n_clusters to 1.
     *
     * Noise points get cluster id of -1.
     *
     * @param n_clusters maximal number of clusters to find
     * @param res [out] c_contiguous matrix of shape (n_clusters+1, n)
     */
    void get_labels_matrix(ssize_t n_clusters, ssize_t* res) {
        if (this->get_max_n_clusters() < n_clusters) {
            // there is nothing to do, no merge needed.
            throw std::runtime_error("The requested number of clusters \
                is too large with this many detected noise points");
        }

        if (this->results.ds.get_n() <= 0)
            throw std::runtime_error("Apply the clustering procedure first.");

        CGiniDisjointSets ds(this->get_max_n_clusters());
        // you can do up to this->get_max_n_clusters() - 1 merges
        ssize_t cur_cluster = n_clusters;
        if (this->get_max_n_clusters() == n_clusters) {
            cur_cluster--;
            GENIECLUST_ASSERT(cur_cluster >= 0)
            this->get_labels(&ds, &res[cur_cluster * this->n]);
        }
        for (ssize_t it=0; it<this->get_max_n_clusters() - 1; ++it) {
            ssize_t j = (this->results.links[it]);
            ssize_t i1 = this->mst_i[2*j+0];
            ssize_t i2 = this->mst_i[2*j+1];
            GENIECLUST_ASSERT(i1 >= 0 && i2 >= 0)
            ds.merge(this->denoise_index_rev[i1], this->denoise_index_rev[i2]);
            if (it >= this->get_max_n_clusters() - n_clusters - 1) {
                cur_cluster--;
                GENIECLUST_ASSERT(cur_cluster >= 0)
                this->get_labels(&ds, &res[cur_cluster * this->n]);
            }
        }
        GENIECLUST_ASSERT(cur_cluster == 0)
    }


    /*! Propagate res with clustering results -
     *  based on the current this->results.links.
     *
     * If there are noise points, not all elements in res will be set.
     *
     * @param res [out] c_contiguous array of length n-1,
     * res[i] gives the index of the MST edge merged at the i-th iteration.
     *
     * @return number of items in res set (the array is padded with -1s)
     */
    ssize_t get_links(ssize_t* res) {
        if (this->results.ds.get_n() <= 0)
            throw std::runtime_error("Apply the clustering procedure first.");

        for (ssize_t i=0; i<this->results.it; ++i) {
            res[i] = this->results.links[i];
        }

        for (ssize_t i=this->results.it; i<this->n-1; ++i) {
            res[i] = -1;
        }

        return this->results.it;
    }

    /*! Set res[i] to true if the i-th point is a noise one.
     *
     *  Makes sense only if noise_leaves==true
     *
     *  @param res [out] array of length n
     */
    void get_noise_status(bool* res) const {
        for (ssize_t i=0; i<n; ++i) {
            res[i] = (this->noise_leaves && this->deg[i] <= 1);
        }
    }

};



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
 *
 *   References
 *   ===========
 *
 *   Gagolewski M., Bartoszuk M., Cena A.,
 *   Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
 *   Information Sciences 363, 2016, pp. 8-23. doi:10.1016/j.ins.2016.05.003
 */
template <class T>
class CGenie : public CGenieBase<T> {
protected:


    /*! Run the Genie++ partitioning.
     *
     *  @param ds
     *  @param mst_skiplist
     *  @param n_clusters maximal number of clusters to detect
     *  @param gini_threshold
     *  @param links [out] c_contiguous array of size (n-1),
     *      links[iter] = index of merged mst_i (up to the number of performed
     *      merges, see retval).
     *
     *  @return The number of performed merges.
     */
    ssize_t do_genie(CGiniDisjointSets* ds, CIntDict<ssize_t>* mst_skiplist,
        ssize_t n_clusters, double gini_threshold, std::vector<ssize_t>* links)
    {
        if (this->get_max_n_clusters() < n_clusters) {
            // there is nothing to do, no merge needed.
            throw std::runtime_error("The requested number of clusters \
                is too large with this many detected noise points");
        }

        // mst_skiplist contains all mst_i edge indexes
        // that we need to consider, and nothing more.
        GENIECLUST_ASSERT(!mst_skiplist->empty());
        ssize_t lastidx = mst_skiplist->get_key_min();
        ssize_t lastm = 0; // last minimal cluster size
        ssize_t it = 0;
        while (!mst_skiplist->empty() && it<this->get_max_n_clusters() - n_clusters) {

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

                GENIECLUST_ASSERT(lastidx >= 0 && lastidx < this->n - 1);
                GENIECLUST_ASSERT(this->mst_i[2*lastidx+0] >= 0 && this->mst_i[2*lastidx+1] >= 0);

                // find the MST edge connecting a cluster of the smallest size
                // with another one
                while (ds->get_count(this->denoise_index_rev[this->mst_i[2*lastidx+0]]) != m
                    && ds->get_count(this->denoise_index_rev[this->mst_i[2*lastidx+1]]) != m)
                {
                    lastidx = mst_skiplist->get_key_next(lastidx);
                    GENIECLUST_ASSERT(lastidx >= 0 && lastidx < this->n - 1);
                    GENIECLUST_ASSERT(this->mst_i[2*lastidx+0] >= 0 && this->mst_i[2*lastidx+1] >= 0);
                }

                i1 = this->mst_i[2*lastidx+0];
                i2 = this->mst_i[2*lastidx+1];
                (*links)[it] = lastidx;
                ssize_t delme = lastidx;
                lastidx = mst_skiplist->get_key_next(lastidx);
                mst_skiplist->erase(delme); // O(1)
                lastm = m;
            }
            else { // single linkage-like
                // note that we consume the MST edges in an non-decreasing order w.r.t. weights
                ssize_t curidx = mst_skiplist->pop_key_min();
                GENIECLUST_ASSERT(curidx >= 0 && curidx < this->n - 1);
                i1 = this->mst_i[2*curidx+0];
                i2 = this->mst_i[2*curidx+1];
                (*links)[it] = curidx;
            }

            GENIECLUST_ASSERT(i1 >= 0 && i2 >= 0)
            ds->merge(this->denoise_index_rev[i1], this->denoise_index_rev[i2]);
            it++;
        }

        return it; // number of merges performed
    }




public:
    CGenie(T* mst_d, ssize_t* mst_i, ssize_t n, bool noise_leaves)
        : CGenieBase<T>(mst_d, mst_i, n, noise_leaves)
    {
        ;
    }

    CGenie() : CGenie(NULL, NULL, 0, false) { }


    /*! Run the Genie++ algorithm
     *
     * @param n_clusters number of clusters to find, 1 for the complete hierarchy
     *    (warning: the algorithm might stop early if there are many noise points
     *     or the number of clusters to detect is > 1).
     * @param gini_threshold the Gini index threshold
     */
    void apply_genie(ssize_t n_clusters, double gini_threshold)
    {
        if (n_clusters < 1)
            throw std::domain_error("n_clusters must be >= 1");

        this->results = typename CGenieBase<T>::CGenieResult(this->n,
            this->noise_count, n_clusters);

        CIntDict<ssize_t> mst_skiplist(this->n - 1);
        this->mst_skiplist_init(&mst_skiplist);

        this->results.it = this->do_genie(&(this->results.ds), &mst_skiplist,
            n_clusters, gini_threshold, &(this->results.links));
    }

};





/*! GIc (Genie+Information Criterion) Hierarchical Clustering Algorithm
 *
 *  GIc has been proposed by Anna Cena in [1] and was inspired
 *  by Mueller's (et al.) ITM [2] and Gagolewski's (et al.) Genie [3]
 *
 *  References
 *  ==========
 *
 *  [1] Cena A., Adaptive hierarchical clustering algorithms based on
 *  data aggregation methods, PhD Thesis, Systems Research Institute,
 *  Polish Academy of Sciences 2018.
 *
 *  [2] Mueller A., Nowozin S., Lampert C.H., Information Theoretic
 *  Clustering using Minimum Spanning Trees, DAGM-OAGM 2012.
 *
 *  [3] Gagolewski M., Bartoszuk M., Cena A.,
 *  Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
 *  Information Sciences 363, 2016, pp. 8-23. doi:10.1016/j.ins.2016.05.003
 */
template <class T>
class CGIc : public CGenie<T> {
protected:

    /*! Run the Genie++ algorithm with different thresholds for the Gini index
     *  and determine the intersection of all the resulting
     *  n_clusters-partitions; for this, we need the union of the
     *  set of MST edges that were left "unmerged".
     *
     * @param n_clusters number of clusters to look for in Genie run
     * @param gini_thresholds array of floats in [0,1]
     * @param n_thresholds size of gini_thresholds
     *
     * @return indexes of MST edges that were left unused by at least
     * one Genie algorithm run; this gives the intersection of partitions.
     * The resulting list will contain a sentinel, this->n - 1.
     *
     * If n_thresholds is 0 or the requested n_clusters is too large,
     * all non-noise edges are set as unused.
     */
    std::vector<ssize_t> get_intersection_of_genies(ssize_t n_clusters,
        double* gini_thresholds, ssize_t n_thresholds)
    {
        std::vector<ssize_t> unused_edges;
        if (n_thresholds == 0 || n_clusters >= this->get_max_n_clusters()) {
            // all edges unused -> will start from n singletons
            for (ssize_t i=0; i < this->n - 1; ++i) {
                ssize_t i1 = this->mst_i[2*i+0];
                ssize_t i2 = this->mst_i[2*i+1];
                if (i1 < 0 || i2 < 0)
                    continue; // a no-edge -> ignore
                if (!this->noise_leaves || (this->deg[i1] > 1 && this->deg[i2] > 1))
                    unused_edges.push_back(i);
            }
            unused_edges.push_back(this->n - 1);  // sentinel
            return unused_edges;
            // EOF.
        }
        else {
            // the same initial skiplist is used in each iter:
            CIntDict<ssize_t> mst_skiplist_template(this->n-1);
            this->mst_skiplist_init(&mst_skiplist_template);

            for (ssize_t i=0; i<n_thresholds; ++i) {
                double gini_threshold = gini_thresholds[i];
                CGiniDisjointSets ds(this->get_max_n_clusters());
                std::vector<ssize_t> links(this->n - 1, -1); // the history of edge merges
                CIntDict<ssize_t> mst_skiplist(mst_skiplist_template);
                this->do_genie(&ds, &mst_skiplist, n_clusters, gini_threshold,
                               &links);

                // start where do_genie() concluded; add all remaining MST edges
                // to the list of unused_edges
                while (!mst_skiplist.empty())
                    unused_edges.push_back(mst_skiplist.pop_key_min());
            }

            // let unused_edges = sort(unique(unused_edges))
            unused_edges.push_back(this->n - 1); // sentinel
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
            GENIECLUST_ASSERT(unused_edges[k] == this->n - 1);
            return unused_edges;
        }
    }

public:
    CGIc(T* mst_d, ssize_t* mst_i, ssize_t n, bool noise_leaves)
        : CGenie<T>(mst_d, mst_i, n, noise_leaves)
    {
        ;
    }

    CGIc() : CGIc(NULL, NULL, 0, false) { }



    /*! Run the GIc (Genie+Information Criterion) algorithm
     *
     * @param n_clusters maximal number of clusters to find,
     *    1 for the complete hierarchy (if possible)
     * @param add_clusters number of additional clusters to work
     *     with internally
     * @param n_features number of features (can be fractional)
     * @param gini_thresholds array of size n_thresholds
     * @param n_thresholds size of gini_thresholds
     */
    void apply_gic(ssize_t n_clusters,
                   ssize_t add_clusters, double n_features,
                   double* gini_thresholds, ssize_t n_thresholds)
    {
        if (n_clusters < 1)
            throw std::domain_error("n_clusters must be >= 1");

        GENIECLUST_ASSERT(add_clusters>=0);
        GENIECLUST_ASSERT(n_thresholds>=0);

        std::vector<ssize_t> unused_edges = get_intersection_of_genies(
                n_clusters+add_clusters, gini_thresholds, n_thresholds
        );

        // note that the unused_edges list:
        // 1. does not include noise edges;
        // 2. is sorted (strictly) increasingly
        // 3. contains a sentinel element at the end == n-1


        this->results = typename CGenieBase<T>::CGenieResult(this->n,
            this->noise_count, n_clusters);

        // Step 1. Merge all used edges (used by all the Genies)
        // There are of course many possible merge orders that we could consider
        // here. We will rely on the current ordering of the MST edges,
        // which is wrt non-decreasing mst_d.

        ssize_t cur_unused_edges = 0;
        ssize_t num_unused_edges = unused_edges.size()-1; // ignore sentinel
        std::vector<ssize_t> cluster_sizes(this->get_max_n_clusters(), 1);
        std::vector<T> cluster_d_sums(this->get_max_n_clusters(), (T)0.0);
        this->results.it = 0;
        for (ssize_t i=0; i<this->n - 1; ++i) {
            GENIECLUST_ASSERT(i<=unused_edges[cur_unused_edges]);
            if (unused_edges[cur_unused_edges] == i) {
                // ignore current edge and advance to the next unused edge
                cur_unused_edges++;
                continue;
            }

            ssize_t i1 = this->mst_i[2*i+0];
            ssize_t i2 = this->mst_i[2*i+1];

            if (i1 < 0 || i2 < 0)
                continue; // a no-edge -> ignore

            if (!this->noise_leaves || (this->deg[i1] > 1 && this->deg[i2] > 1)) {
                GENIECLUST_ASSERT(this->results.it < this->n-1);
                this->results.links[this->results.it++] = i;
                i1 = this->results.ds.find(this->denoise_index_rev[i1]);
                i2 = this->results.ds.find(this->denoise_index_rev[i2]);
                if (i1 > i2) std::swap(i1, i2);
                this->results.ds.merge(i1, i2);
                // new parent node is i1
                cluster_sizes[i1]  += cluster_sizes[i2];
                cluster_d_sums[i1] += cluster_d_sums[i2] + this->mst_d[i];
                cluster_sizes[i2]   = 0;
                cluster_d_sums[i2]  = INFTY;
            }
        }
        GENIECLUST_ASSERT(cur_unused_edges == num_unused_edges); // sentinel
        GENIECLUST_ASSERT(unused_edges[num_unused_edges] == this->n-1); // sentinel
        GENIECLUST_ASSERT(num_unused_edges+1 == this->results.ds.get_k());


        // Step 2. Merge all used edges

        /*  The objective function - Information Criterion - to MAXIMISE is
            sum_{i in ds.parents()} -cluster_sizes[i] * (
                n_features     * log cluster_sizes[i]
              -(n_features-1)  * log cluster_d_sums[i]
            )
        */

        while (num_unused_edges > 0 && this->results.it<this->get_max_n_clusters() - n_clusters) {
            ssize_t max_which = -1;
            double  max_obj = -INFTY;
            for (ssize_t j=0; j<num_unused_edges; ++j) {
                ssize_t i = unused_edges[j];
                ssize_t i1 = this->mst_i[2*i+0];
                ssize_t i2 = this->mst_i[2*i+1];
                GENIECLUST_ASSERT(i1 >= 0 && i2 >= 0);
                i1 = this->results.ds.find(this->denoise_index_rev[i1]);
                i2 = this->results.ds.find(this->denoise_index_rev[i2]);
                if (i1 > i2) std::swap(i1, i2);
                GENIECLUST_ASSERT(i1 != i2);

                // singletons should be merged first
                // (we assume that they have cluster_d_sums==Inf
                // (this was not addressed in Mueller's in his paper)
                if (cluster_d_sums[i1] < 1e-12 || cluster_d_sums[i2] < 1e-12) {
                    max_which = j;
                    break;
                }

                double cur_obj = -(cluster_sizes[i1]+cluster_sizes[i2])*(
                    n_features*log(cluster_d_sums[i1]+cluster_d_sums[i2]+this->mst_d[i])
                  -(n_features-1.0)*log(cluster_sizes[i1]+cluster_sizes[i2])
                );
                cur_obj += cluster_sizes[i1]*(
                    n_features*log(cluster_d_sums[i1])
                  -(n_features-1.0)*log(cluster_sizes[i1])
                );
                cur_obj += cluster_sizes[i2]*(
                    n_features*log(cluster_d_sums[i2])
                  -(n_features-1.0)*log(cluster_sizes[i2])
                );

                GENIECLUST_ASSERT(std::isfinite(cur_obj));
                if (cur_obj > max_obj) {
                    max_obj = cur_obj;
                    max_which = j;
                }
            }

            GENIECLUST_ASSERT(max_which >= 0 && max_which < num_unused_edges);
            ssize_t i = unused_edges[max_which];
            GENIECLUST_ASSERT(this->results.it < this->n - 1);
            this->results.links[this->results.it++] = i;
            ssize_t i1 = this->mst_i[2*i+0];
            ssize_t i2 = this->mst_i[2*i+1];
            GENIECLUST_ASSERT(i1 >= 0 && i2 >= 0);
            i1 = this->results.ds.find(this->denoise_index_rev[i1]);
            i2 = this->results.ds.find(this->denoise_index_rev[i2]);
            if (i1 > i2) std::swap(i1, i2);

            this->results.ds.merge(i1, i2);
            // new parent node is i1

            cluster_sizes[i1]  += cluster_sizes[i2];
            cluster_d_sums[i1] += cluster_d_sums[i2]+this->mst_d[i];
            cluster_sizes[i2] = 0;
            cluster_d_sums[i2] = INFTY;

            unused_edges[max_which] = unused_edges[num_unused_edges-1];
            num_unused_edges--;
        }
    }
};




#endif
