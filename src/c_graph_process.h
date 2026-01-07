/*  Graph pre/post-processing and other routines
 *
 *  Copyleft (C) 2018-2026, Marek Gagolewski <https://www.gagolewski.com>
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


#ifndef __c_graph_process_h
#define __c_graph_process_h

#include <stdexcept>
#include <algorithm>
#include "c_common.h"
#include "c_argfuns.h"


/*! Translate indexes based on a skip array.
 *
 * If skip=[False, True, False, False, True, False, False],
 * then the indexes in ind are mapped in such a way that:
 * 0 -> 0,
 * 1 -> 2,
 * 2 -> 3,
 * 3 -> 5,
 * 4 -> 6
 *
 * @param ind [in/out] Array of m indexes to translate (does not have to be sorted)
 * @param m size of ind
 * @param skip Boolean array of size n
 * @param n size of skip
 */
void Ctranslate_skipped_indexes(
    Py_ssize_t* ind, Py_ssize_t m,
    const bool* skip, Py_ssize_t n
) {
    if (m <= 0) return;

    std::vector<Py_ssize_t> o(m);
    Cargsort(o.data(), ind, m, false);

    Py_ssize_t j = 0;
    Py_ssize_t k = 0;
    for (Py_ssize_t i=0; i<n; ++i) {
        if (skip[i]) continue;

        if (ind[o[k]] == j) {
            ind[o[k]] = i;
            k++;

            if (k == m) return;
        }

        j++;
    }

    throw std::domain_error("index to translate out of range");
}


/** Count the number of non-zero elements in a Boolean array x of length n
 */
Py_ssize_t Csum_bool(const bool* x, Py_ssize_t n)
{
    Py_ssize_t s = 0;
    for (Py_ssize_t i=0; i<n; ++i)
        if (x[i]) s++;
    return s;
}


/*! Compute the degree of each vertex in an undirected graph
 *  over a vertex set {0,...,n-1}.
 *
 *
 * @param ind c_contiguous matrix of size m*2,
 *     where {ind[i,0], ind[i,1]} is the i-th edge with ind[i,j] < n
 * @param m number of edges (rows in ind)
 * @param n number of vertices
 * @param deg [out] array of size n, where
 *     deg[i] will give the degree of the i-th vertex.
 */
void Cgraph_get_node_degrees(
    const Py_ssize_t* ind,
    const Py_ssize_t m,
    const Py_ssize_t n,
    Py_ssize_t* deg /*out*/
) {
    for (Py_ssize_t i=0; i<n; ++i)
        deg[i] = 0;

    for (Py_ssize_t i=0; i<m; ++i) {
        Py_ssize_t u = ind[2*i+0];
        Py_ssize_t v = ind[2*i+1];

        if (u < 0 || v < 0)
            throw std::domain_error("All elements must be >= 0");
        else if (u >= n || v >= n)
            throw std::domain_error("All elements must be < n");
        else if (u == v)
            throw std::domain_error("Self-loops are not allowed");

        deg[u]++;
        deg[v]++;
    }
}


/*! Compute the incidence list of each vertex in an undirected graph
 *  over a vertex set {0,...,n-1}.
 *
 *  @param ind c_contiguous matrix of size m*2,
 *      where {ind[i,0], ind[i,1]} is the i-th edge with ind[i,j] < n
 *  @param m number of edges (rows in ind)
 *  @param n number of vertices
 *  @param cumdeg [out] array of size n+1, where cumdeg[i+1] the sum of the first i vertex degrees
 *  @param inc [out] array of size 2*m; inc[cumdeg[i]]..inc[cumdeg[i+1]-1] gives the edges incident on the i-th vertex
 */
void Cgraph_get_node_inclists(
    const Py_ssize_t* ind,
    const Py_ssize_t m,
    const Py_ssize_t n,
    Py_ssize_t* cumdeg,
    Py_ssize_t* inc
) {
    cumdeg[0] = 0;
    Cgraph_get_node_degrees(ind, m, n, cumdeg+1);

    Py_ssize_t cd = 0;
    for (Py_ssize_t i=1; i<n+1; ++i) {
        Py_ssize_t this_deg = cumdeg[i];
        cumdeg[i] = cd;
        cd += this_deg;
    }
    // that's not it yet; cumdeg is adjusted below


    for (Py_ssize_t e=0; e<m; ++e) {
        Py_ssize_t u = ind[2*e+0];
        Py_ssize_t v = ind[2*e+1];

        *(inc+cumdeg[u+1]) = e;
        ++(cumdeg[u+1]);

        *(inc+cumdeg[v+1]) = e;
        ++(cumdeg[v+1]);
    }

    GENIECLUST_ASSERT(cumdeg[0] == 0);
    GENIECLUST_ASSERT(cumdeg[n] == 2*m);


// #ifdef DEBUG
//     cumdeg = 0;
//     inc[0] = data;
//     for (Py_ssize_t i=0; i<n; ++i) {
//         GENIECLUST_ASSERT(inc[i] == data+cumdeg);
//         cumdeg += deg[i];
//     }
// #endif
}



/* ************************************************************************** */


class CMSTProcessorBase
{
protected:
    const Py_ssize_t* mst_i;  // size m*2, elements in [0,n)
    const Py_ssize_t m;  // preferably == n-1; number of edges in mst_i
    const Py_ssize_t n;  // number of vertices

    Py_ssize_t* c;  // nullable or length n
    const Py_ssize_t* cumdeg;  // nullable or length n+1
    const Py_ssize_t* inc;  // nullable or length 2*m
    const bool* skip_edges;  // nullable or length m

    std::vector<Py_ssize_t> _cumdeg;  // data buffer for cumdeg (optional)
    std::vector<Py_ssize_t> _inc;     // data buffer for inc (optional)


public:

    CMSTProcessorBase(
        const Py_ssize_t* mst_i,
        const Py_ssize_t m,
        const Py_ssize_t n,
        Py_ssize_t* c=nullptr,
        const Py_ssize_t* cumdeg=nullptr,
        const Py_ssize_t* inc=nullptr,
        const bool* skip_edges=nullptr
    ) :
        mst_i(mst_i), m(m), n(n), c(c),
        cumdeg(cumdeg), inc(inc), skip_edges(skip_edges)
    {
        if (!cumdeg) {
            GENIECLUST_ASSERT(!inc);
            _cumdeg.resize(n+1);
            _inc.resize(2*m);
            Cgraph_get_node_inclists(mst_i, m, n, _cumdeg.data(), _inc.data());
            this->cumdeg = _cumdeg.data();
            this->inc = _inc.data();
        }
        else {
            GENIECLUST_ASSERT(inc);
        }
    }

    virtual void process() = 0;
};




/* ************************************************************************** */



/* ************************************************************************** */



/** See Cmst_get_cluster_sizes below.
 */
class CMSTClusterSizeGetter : public CMSTProcessorBase
{
private:

    Py_ssize_t max_k;
    Py_ssize_t* s;  // NULL or of size max_k >= k, where k is the number of clusters
    Py_ssize_t k;   // the number of connected components identified


    Py_ssize_t visit(Py_ssize_t v, Py_ssize_t e)
    {
        Py_ssize_t w;

        if (e < 0) {
            w = v;
        }
        else if (skip_edges && skip_edges[e])
            return 0;
        else {
            Py_ssize_t iv = (Py_ssize_t)(mst_i[2*e+1]==v);
            w = mst_i[2*e+(1-iv)];
        }

        GENIECLUST_ASSERT(c[w] < 0);
        c[w] = k;

        Py_ssize_t curs = 1;

        for (const Py_ssize_t* pe = inc+cumdeg[w]; pe != inc+cumdeg[w+1]; pe++) {
            if (*pe != e) curs += visit(w, *pe);
        }

        return curs;
    }


public:
    CMSTClusterSizeGetter(
        const Py_ssize_t* mst_i,
        Py_ssize_t m,
        Py_ssize_t n,
        Py_ssize_t* c,
        Py_ssize_t max_k,
        Py_ssize_t* s=nullptr,
        const Py_ssize_t* cumdeg=nullptr,
        const Py_ssize_t* inc=nullptr,
        const bool* skip_edges=nullptr
    ) : CMSTProcessorBase(mst_i, m, n, c, cumdeg, inc, skip_edges), max_k(max_k), s(s), k(-1)
    {
        GENIECLUST_ASSERT(this->c);
        GENIECLUST_ASSERT(this->cumdeg);
        GENIECLUST_ASSERT(this->inc);
    }


    virtual void process()
    {
        for (Py_ssize_t v=0; v<n; ++v) c[v] = -1;
        //for (Py_ssize_t i=0; i<k; ++i) s[i] = 0;

        k = 0;
        for (Py_ssize_t v=0; v<n; ++v) {
            if (c[v] >= 0) continue;  // already visited -> skip

            if (s) {
                GENIECLUST_ASSERT(k<max_k);
                s[k] = visit(v, -1);
            }
            else
                visit(v, -1);

            k++;
        }
    }

};




/*! Label connected components in a spanning forest (where skip_edges
 *  designate the edges omitted from the tree) and fetch their sizes
 *
 *  @param mst_i c_contiguous matrix of size m*2,
 *     where {mst_i[k,0], mst_i[k,1]} specifies the k-th (undirected) edge
 *     in the spanning tree (or forest); 0 <= mst_i[i,j] < n;
 *     edges with mst_i[i,0] < 0 or mst_i[i,1] < 0 are ignored.
 *  @param m number of rows in mst_i (edges)
 *  @param n length of c and the number of vertices in the spanning tree
 *  @param c [out] array of length n, where
 *      c[i] denotes the cluster ID (in {0, 1, ..., k-1} for some k)
 *      of the i-th object, i=0,...,n-1
 *  @param max_k the actual size of s (a safeguard)
 *  @param s [out] array of length max_k >= k, where k is the number of connected
 *      components in the forest; s[i] gives the size of the i-th cluster;
 *      pass NULL to get only the cluster labels;
 *      obviously, k<=n; e.g., if m==n-1, then k=sum(skip_edges)+1;
 *      if the size of s is > k, then the trailing elements will be left unset
 *  @param skip_edges Boolean array of length m or NULL; indicates the edges to skip
 *  @param cumdeg an array of length n+1 or NULL; see Cgraph_get_node_inclists
 *  @param inc an array of length 2*m or NULL; see Cgraph_get_node_inclists
 */
void Cmst_get_cluster_sizes(
    const Py_ssize_t* mst_i,
    Py_ssize_t m,
    Py_ssize_t n,
    Py_ssize_t* c,
    Py_ssize_t max_k=0,
    Py_ssize_t* s=nullptr,
    const Py_ssize_t* cumdeg=nullptr,
    const Py_ssize_t* inc=nullptr,
    const bool* skip_edges=nullptr
) {
    CMSTClusterSizeGetter get(mst_i, m, n, c, max_k, s, cumdeg, inc, skip_edges);
    get.process();  // modifies c in place
}


/* ************************************************************************** */




/** See Cmst_impute_missing_labels below.
 */
class CMSTMissingLabelsImputer : public CMSTProcessorBase
{
private:

    void visit(Py_ssize_t v, Py_ssize_t e)
    {
        if (skip_edges && skip_edges[e]) return;

        Py_ssize_t iv = (Py_ssize_t)(mst_i[2*e+1]==v);
        Py_ssize_t w = mst_i[2*e+(1-iv)];

        GENIECLUST_ASSERT(c[v] >= 0);
        GENIECLUST_ASSERT(c[w] < 0);

        c[w] = c[v];

        for (const Py_ssize_t* pe = inc+cumdeg[w]; pe != inc+cumdeg[w+1]; pe++) {
            if (*pe != e) visit(w, *pe);
        }
    }


public:
    CMSTMissingLabelsImputer(
        const Py_ssize_t* mst_i,
        Py_ssize_t m,
        Py_ssize_t n,
        Py_ssize_t* c,
        const Py_ssize_t* cumdeg=nullptr,
        const Py_ssize_t* inc=nullptr,
        const bool* skip_edges=nullptr
    ) : CMSTProcessorBase(mst_i, m, n, c, cumdeg, inc, skip_edges)
    {
        GENIECLUST_ASSERT(this->c);
        GENIECLUST_ASSERT(this->cumdeg);
        GENIECLUST_ASSERT(this->inc);
    }


    virtual void process()
    {
        for (Py_ssize_t v=0; v<n; ++v) {
            if (c[v] < 0) continue;

            for (const Py_ssize_t* pe = inc+cumdeg[v]; pe != inc+cumdeg[v+1]; pe++) {
                if (skip_edges && skip_edges[*pe]) continue;

                Py_ssize_t iv = (Py_ssize_t)(mst_i[2*(*pe)+1]==v);
                Py_ssize_t w = mst_i[2*(*pe)+(1-iv)];

                if (c[w] < 0) {  // descend into this branch to impute missing values
                    visit(v, *pe);
                }
            }
        }
    }

};



/*! Impute missing labels in all tree branches.
 *  All nodes in branches with class ID of -1 will be assigned their parent node's class.
 *
 *  @param mst_i c_contiguous matrix of size m*2,
 *     where {mst_i[k,0], mst_i[k,1]} specifies the k-th (undirected) edge
 *     in the spanning tree (or forest); 0 <= mst_i[i,j] < n;
 *     edges with mst_i[i,0] < 0 or mst_i[i,1] < 0 are ignored.
 *  @param m number of rows in mst_i (edges)
 *  @param n length of c and the number of vertices in the spanning tree
 *  @param c [in/out] c_contiguous vector of length n, where
 *      c[i] denotes the cluster ID (in {-1, 0, 1, ..., k-1} for some k)
 *      of the i-th object, i=0,...,n-1.  Class -1 represents missing values
 *      to be imputed
 *  @param skip_edges Boolean array of length m or NULL; indicates the edges to skip
 *  @param cumdeg an array of length n+1 or NULL; see Cgraph_get_node_inclists
 *  @param inc an array of length 2*m or NULL; see Cgraph_get_node_inclists
 */
void Cmst_impute_missing_labels(
    const Py_ssize_t* mst_i,
    Py_ssize_t m,
    Py_ssize_t n,
    Py_ssize_t* c,
    const Py_ssize_t* cumdeg=nullptr,
    const Py_ssize_t* inc=nullptr,
    const bool* skip_edges=nullptr
) {
    CMSTMissingLabelsImputer imp(mst_i, m, n, c, cumdeg, inc, skip_edges);
    imp.process();  // modifies c in place
}


/* ************************************************************************** */



/** See Cmst_trim_branches below.  [DEPRECATED]
 */
template <class FLOAT> class CMSTBranchTrimmer : public CMSTProcessorBase
{
private:
    const FLOAT* mst_d;
    const FLOAT min_d;
    const Py_ssize_t max_size;


    std::vector<Py_ssize_t> size;

    Py_ssize_t clk;   // the number of connected components
    std::vector<Py_ssize_t> clsize;


    Py_ssize_t visit_get_sizes(Py_ssize_t v, Py_ssize_t e)
    {
        if (skip_edges && skip_edges[e]) return 0;

        Py_ssize_t iv = (Py_ssize_t)(mst_i[2*e+1]==v);
        Py_ssize_t w = mst_i[2*e+(1-iv)];

        GENIECLUST_ASSERT(e >= 0 && e < m);
        GENIECLUST_ASSERT(v >= 0 && v < n);
        GENIECLUST_ASSERT(w >= 0 && w < n);
        GENIECLUST_ASSERT(c[w] < 0);

        Py_ssize_t this_size = 1;
        c[w] = c[v];

        for (const Py_ssize_t* pe = inc+cumdeg[w]; pe != inc+cumdeg[w+1]; pe++) {
            if (*pe != e) this_size += visit_get_sizes(w, *pe);
        }

        size[2*e + (1-iv)] = this_size;
        size[2*e + iv] = -1;

        return this_size;
    }


    void visit_mark(Py_ssize_t v, Py_ssize_t e)
    {
        if (skip_edges && skip_edges[e]) return;

        Py_ssize_t iv = (Py_ssize_t)(mst_i[2*e+1]==v);
        Py_ssize_t w = mst_i[2*e+(1-iv)];

        if (c[w] < 0) return;  // already visited

        c[w] = -1;

        for (const Py_ssize_t* pe = inc+cumdeg[w]; pe != inc+cumdeg[w+1]; pe++) {
            if (*pe != e) visit_mark(w, *pe);
        }
    }


public:
    CMSTBranchTrimmer(
        const FLOAT* mst_d,
        FLOAT min_d,
        Py_ssize_t max_size,
        const Py_ssize_t* mst_i,
        Py_ssize_t m,
        Py_ssize_t n,
        Py_ssize_t* c,
        const Py_ssize_t* cumdeg=nullptr,
        const Py_ssize_t* inc=nullptr,
        const bool* skip_edges=nullptr
    ) :
        CMSTProcessorBase(mst_i, m, n, c, cumdeg, inc, skip_edges),
        mst_d(mst_d), min_d(min_d), max_size(max_size),
        size(2*m, -1)
    {
        GENIECLUST_ASSERT(this->c);
        GENIECLUST_ASSERT(this->cumdeg);
        GENIECLUST_ASSERT(this->inc);

        GENIECLUST_ASSERT(m == n-1);

        // the number of connected components:
        clk = 1;
        if (skip_edges) {
            for (Py_ssize_t e = 0; e < m; ++e)
                if (skip_edges[e]) clk++;
        }
        clsize.resize(clk);
    }


    virtual void process()
    {
        for (Py_ssize_t v=0; v<n; ++v) c[v] = -1;
        for (Py_ssize_t i=0; i<clk; ++i) clsize[i] = 0;

        Py_ssize_t lastc = 0;
        for (Py_ssize_t v=0; v<n; ++v) {
            if (c[v] >= 0) continue;

            c[v] = lastc;
            Py_ssize_t this_size = 1;

            for (const Py_ssize_t* pe = inc+cumdeg[v]; pe != inc+cumdeg[v+1]; pe++) {
                if (skip_edges && skip_edges[*pe]) continue;
                this_size += visit_get_sizes(v, *pe);
            }
            clsize[lastc] = this_size;

            lastc++;
            if (lastc == clk) break;
        }

        GENIECLUST_ASSERT(lastc == clk);
        GENIECLUST_ASSERT(clk > 1 || clsize[0] == n);

        for (Py_ssize_t e=0; e<m; ++e) {
            if (skip_edges && skip_edges[e]) continue;
            GENIECLUST_ASSERT(size[2*e+0] > 0 || size[2*e+1] > 0);
            GENIECLUST_ASSERT(clsize[c[mst_i[2*e+0]]] == clsize[c[mst_i[2*e+1]]]);
            if (size[2*e+0] > 0)
                size[2*e+1] = clsize[c[mst_i[2*e+0]]] - size[2*e+0];
            else
                size[2*e+0] = clsize[c[mst_i[2*e+1]]] - size[2*e+1];
        }


        for (Py_ssize_t e=0; e<m; ++e) {
            if (skip_edges && skip_edges[e]) continue;
            if (mst_d[e] <= min_d) continue;

            Py_ssize_t iv = (size[2*e+0]>=size[2*e+1])?0:1;
            Py_ssize_t v = mst_i[2*e+iv];
            if (c[v] < 0) continue;
            if (size[2*e+(1-iv)] > max_size) continue;
            visit_mark(v, e);
        }
    }

};


/*! Trim tree branches of size <= max_size connected by an edge > min_d  [DEPRECATED]
 *
 *
 *  @param mst_d m edge weights
 *  @param mst_i c_contiguous matrix of size m*2,
 *     where {mst_i[k,0], mst_i[k,1]} specifies the k-th (undirected) edge
 *     in the spanning tree (or forest); 0 <= mst_i[i,j] < n;
 *     edges with mst_i[i,0] < 0 or mst_i[i,1] < 0 are ignored.
 *  @param m number of rows in mst_i (edges)
 *  @param c [out] vector of length n; c[i] == -1 marks a trimmed-out point,
 *     whereas c[i] >= 0 denotes a retained one
 *  @param n length of c and the number of vertices in the spanning tree, n == m+1
 *  @param min_d minimal edge weight to be considered trimmable
 *  @param max_size maximal allowable size of a branch to cut
 *  @param skip_edges Boolean array of length m or NULL; indicates the edges to skip
 *  @param cumdeg an array of length n+1 or NULL; see Cgraph_get_node_inclists
 *  @param inc an array of length 2*m or NULL; see Cgraph_get_node_inclists
 */
template <class FLOAT>
void Cmst_trim_branches(
    const FLOAT* mst_d,
    FLOAT min_d,
    Py_ssize_t max_size,
    const Py_ssize_t* mst_i,
    Py_ssize_t m,
    Py_ssize_t n,
    Py_ssize_t* c,
    const Py_ssize_t* cumdeg=nullptr,
    const Py_ssize_t* inc=nullptr,
    const bool* skip_edges=nullptr
) {
    CMSTBranchTrimmer tr(mst_d, min_d, max_size, mst_i, m, n, c, cumdeg, inc, skip_edges);
    tr.process();  // modifies c in place
}


/* ************************************************************************** */


/*! Merge all midliers with their nearest clusters  [DEPRECATED]
 *
 *  The i-th node is a midlier if it is a leaf in the spanning tree
 *  (and hence it meets c[i] < 0) which is amongst the
 *  M nearest neighbours of its adjacent vertex, j.
 *
 *  This procedure allocates c[i] to its its closest cluster, c[j].
 *
 *
 *  @param mst_i c_contiguous matrix of size num_edges*2,
 *     where {mst_i[k,0], mst_i[k,1]} specifies the k-th (undirected) edge
 *     in the spanning tree (or forest); 0 <= mst_i[i,j] < n;
 *     edges with mst_i[i,0] < 0 or mst_i[i,1] < 0 are ignored.
 *  @param num_edges number of rows in mst_i (edges)
 *  @param nn_i c_contiguous matrix of size n*num_neighbours;
 *     nn[i,:] gives the indexes of the i-th point's
 *     nearest neighbours; -1 indicates a "missing value"
 *  @param num_neighbours number of columns in nn
 *  @param M smoothing factor, 1 <= M <= num_neighbours
 *  @param c [in/out] c_contiguous vector of length n, where
 *      c[i] denotes the cluster ID (in {-1, 0, 1, ..., k-1} for some k)
 *      of the i-th object, i=0,...,n-1.  Class -1 represents the leaves of the
 *      input spanning tree
 *  @param n length of c and the number of vertices in the spanning tree
 */
void Cmerge_midliers(
    const Py_ssize_t* mst_i,
    Py_ssize_t num_edges,
    const Py_ssize_t* nn_i,
    Py_ssize_t num_neighbours,
    Py_ssize_t M,
    Py_ssize_t* c,
    Py_ssize_t n
) {
    if (M < 1 || M > num_neighbours)
        throw std::domain_error("incorrect smoothing factor M");

    for (Py_ssize_t i=0; i<num_edges; ++i) {
        Py_ssize_t u = mst_i[2*i+0];
        Py_ssize_t v = mst_i[2*i+1];
        if (u<0 || v<0)
            continue; // represents a no-edge -> ignore
        if (u>=n || v>=n)
            throw std::domain_error("all elements must be <= n");
        if (c[u] < 0 && c[v] < 0)
            continue; // throw std::domain_error("!(c[u] < 0 && c[v] < 0)");

        if (c[u] >= 0 && c[v] >= 0)
            continue;

        if (c[v] < 0)
            std::swap(u, v);

        GENIECLUST_ASSERT(c[u] <  0);  // u is a leaf
        GENIECLUST_ASSERT(c[v] >= 0);  // v is a non-leaf

        // check if u is amongst v's M nearest neighbours

        //c[u] = -1; // it's negative anyway
        for (Py_ssize_t j=0; j<M; ++j) {
            // -1s are ignored (they should be at the end of the array btw)
            if (nn_i[v*num_neighbours+j] == u) {
                // yes, it's a midlier point
                c[u] = c[v];
                break;
            }
        }
    }
}


/*! Merge all outliers and midliers with their nearest clusters  [DEPRECATED]
 *
 *  For each leaf in the MST, i (and hence a vertex which meets c[i] < 0),
 *  this procedure allocates c[i] to its its closest cluster, c[j],
 *  where j is the vertex adjacent to i.
 *
 *
 *  @param mst_i c_contiguous matrix of size m*2,
 *     where {mst_i[k,0], mst_i[k,1]} specifies the k-th (undirected) edge
 *     in the spanning tree (or forest); 0 <= mst_i[i,j] < n;
 *     edges with mst_i[i,0] < 0 or mst_i[i,1] < 0 are ignored.
 *  @param m number of rows in ind (edges)
 *  @param c [in/out] c_contiguous vector of length n, where
 *      c[i] denotes the cluster ID (in {-1, 0, 1, ..., k-1} for some k)
 *      of the i-th object, i=0,...,n-1.  Class -1 represents the leaves of the
 *      input spanning tree
 *  @param n length of c and the number of vertices in the spanning tree
 */
void Cmerge_all(
    const Py_ssize_t* mst_i,
    Py_ssize_t m,
    Py_ssize_t* c,
    Py_ssize_t n
) {
    for (Py_ssize_t i=0; i<m; ++i) {
        Py_ssize_t u = mst_i[2*i+0];
        Py_ssize_t v = mst_i[2*i+1];
        if (u<0 || v<0)
            continue; // represents a no-edge -> ignore
        if (u>=n || v>=n)
            throw std::domain_error("all elements must be <= n");
        if (c[u] < 0 && c[v] < 0)
            throw std::domain_error("!(c[u] < 0 && c[v] < 0)");

        if (c[u] < 0)
            c[u] = c[v];
        else if (c[v] < 0)
            c[v] = c[u];
        else
           continue;
    }
}


#endif
