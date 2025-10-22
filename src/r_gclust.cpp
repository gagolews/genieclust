/*  The Genie Clustering Algorithm - R Wrapper
 *
 *  Copyleft (C) 2018-2025, Marek Gagolewski <https://www.gagolewski.com>
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


#include "c_common.h"
#include "c_matrix.h"
#include "c_genie.h"
#include "c_postprocess.h"
#include <cmath>

using namespace Rcpp;


/* This function was originally part of the `genie` package for R */
void internal_generate_merge(Py_ssize_t n, NumericMatrix links, NumericMatrix merge)
{
    std::vector<Py_ssize_t> elements(n+1, 0);
    std::vector<Py_ssize_t> parents(n+1, 0);

    Py_ssize_t clusterNumber = 1;
    for (Py_ssize_t k=0; k<n-1; ++k, ++clusterNumber) {
        Py_ssize_t i = (Py_ssize_t)links(k, 0);
        Py_ssize_t j = (Py_ssize_t)links(k, 1);
        Py_ssize_t si = elements[i];
        Py_ssize_t sj = elements[j];
        elements[i] = clusterNumber;
        elements[j] = clusterNumber;

        if (si == 0)
            merge(k, 0) = -(double)i;
        else {
            while (parents[si] != 0) {
                size_t sinew = parents[si];
                parents[si] = clusterNumber;
                si = sinew;
            }
            if (si != 0) parents[si] = clusterNumber;
            merge(k,0) = (double)si;
        }

        if (sj == 0)
            merge(k, 1) = -(double)j;
        else {
            while (parents[sj] != 0) {
                Py_ssize_t sjnew = parents[sj];
                parents[sj] = clusterNumber;
                sj = sjnew;
            }
            if (sj != 0) parents[sj] = clusterNumber;
            merge(k,1) = (double)sj;
        }

        if (merge(k, 0) < 0) {
            if (merge(k, 1) < 0 && merge(k, 0) < merge(k, 1))
                std::swap(merge(k, 0), merge(k, 1));
        }
        else {
            if (merge(k, 0) > merge(k, 1))
                std::swap(merge(k, 0), merge(k, 1));
        }
    }
}


/* Originally, this function was part of the `genie` package for R */
void internal_generate_order(Py_ssize_t n, NumericMatrix merge, NumericVector order)
{
   std::vector< std::list<double> > relord(n+1);
   Py_ssize_t clusterNumber = 1;
   for (Py_ssize_t k=0; k<n-1; ++k, ++clusterNumber) {
      double i = merge(k, 0);
      if (i < 0)
         relord[clusterNumber].push_back(-i);
      else
         relord[clusterNumber].splice(relord[clusterNumber].end(), relord[(size_t)i]);

      double j = merge(k, 1);
      if (j < 0)
         relord[clusterNumber].push_back(-j);
      else
         relord[clusterNumber].splice(relord[clusterNumber].end(), relord[(size_t)j]);
   }

   GENIECLUST_ASSERT(relord[n-1].size() == (size_t)n);
   Py_ssize_t k = 0;
   for (std::list<double>::iterator it = relord[n-1].begin();
         it != relord[n-1].end(); ++it) {
      order[k++] = (*it);
   }
}


// [[Rcpp::export(".genie")]]
IntegerVector dot_genie(
    NumericMatrix mst,
    int k,
    double gini_threshold,
    String postprocess,
    bool skip_leaves,
    bool verbose
) {
    if (verbose) GENIECLUST_PRINT("[genieclust] Determining clusters.\n");

    if (gini_threshold < 0.0 || gini_threshold > 1.0)
        stop("`gini_threshold` must be in [0, 1]");

    Py_ssize_t n = mst.nrow()+1;

    if (k < 1 || k > n) stop("invalid number of clusters requested, `k`");

    CMatrix<Py_ssize_t> mst_i(n-1, 2);
    std::vector<double>  mst_d(n-1);


    for (Py_ssize_t i=0; i<n-1; ++i) {
        mst_i(i, 0) = (Py_ssize_t)mst(i, 0)-1;  // 1-based to 0-based indices
        mst_i(i, 1) = (Py_ssize_t)mst(i, 1)-1;  // 1-based to 0-based indices
        mst_d[i] = mst(i, 2);
    }

    CGenie<double> g(mst_d.data(), mst_i.data(), n, skip_leaves);
    g.compute(k, gini_threshold);


    if (verbose) GENIECLUST_PRINT("[genieclust] Postprocessing the outputs.\n");

    std::vector<Py_ssize_t> xres(n);
    Py_ssize_t k_detected = g.get_labels(k, xres.data());

    if (k_detected != k)
        Rf_warning("The number of clusters detected is different from the requested one, possibly due to the presence of outliers.");

    if (skip_leaves) {
        if (postprocess == "midliers") {
            if (Rf_isNull(mst.attr("nn.index")))
                stop("`nn.index` attribute of the MST not set; unable to proceed with this postprocessing action");

            NumericMatrix nn_r = mst.attr("nn.index");
            GENIECLUST_ASSERT(nn_r.nrow() == n);
            Py_ssize_t M = nn_r.ncol();
            GENIECLUST_ASSERT(M < n);
            CMatrix<Py_ssize_t> nn_i(n, M);
            for (Py_ssize_t i=0; i<n; ++i) {
                for (Py_ssize_t j=0; j<M; ++j) {
                    GENIECLUST_ASSERT(nn_r(i,j) >= 1);
                    GENIECLUST_ASSERT(nn_r(i,j) <= n);
                    nn_i(i,j) = (Py_ssize_t)nn_r(i,j)-1; // 0-based indexing
                }
            }

            Cmerge_midliers(mst_i.data(), n-1, nn_i.data(), M, M, xres.data(), n);
        }
        else if (postprocess == "all") {
            Cmerge_all(mst_i.data(), n-1, xres.data(), n);
        }
        else if (postprocess == "none") {
            ;  // pass
        }
        else
            stop("invalid `postprocess`");
    }

    IntegerVector res(n);
    for (Py_ssize_t i=0; i<n; ++i) {
        if (xres[i] < 0) res[i] = NA_INTEGER;  // outlier/noise point
        else res[i] = xres[i] + 1;
    }

    if (verbose) GENIECLUST_PRINT("[genieclust] Done.\n");

    return res;
}



// [[Rcpp::export(".gclust")]]
List dot_gclust(
    NumericMatrix mst,
    double gini_threshold,
    bool verbose
) {
    if (verbose) GENIECLUST_PRINT("[genieclust] Determining clusters.\n");

    if (gini_threshold < 0.0 || gini_threshold > 1.0)
        stop("`gini_threshold` must be in [0, 1]");

    Py_ssize_t n = mst.nrow()+1;
    CMatrix<Py_ssize_t> mst_i(n-1, 2);
    std::vector<double>  mst_d(n-1);

    for (Py_ssize_t i=0; i<n-1; ++i) {
        mst_i(i, 0) = (Py_ssize_t)mst(i, 0)-1; // 1-based to 0-based indices
        mst_i(i, 1) = (Py_ssize_t)mst(i, 1)-1; // 1-based to 0-based indices
        mst_d[i] = mst(i, 2);
    }

    CGenie<double> g(mst_d.data(), mst_i.data(), n/*, skip_leaves=M>1*/);
    g.compute(1, gini_threshold);


    if (verbose) GENIECLUST_PRINT("[genieclust] Postprocessing the outputs.\n");

    std::vector<Py_ssize_t> links(n-1);
    g.get_links(links.data());


    NumericMatrix links2(n-1, 2);
    NumericVector height(n-1, NA_REAL);
    Py_ssize_t k = 0;
    for (Py_ssize_t i=0; i<n-1; ++i) {
        if (links[i] >= 0) {
            links2(k, 0) = mst_i(links[i], 0) + 1;
            links2(k, 1) = mst_i(links[i], 1) + 1;
            height(k) = mst_d[ links[i] ];
            ++k;
        }
    }
    for (; k<n-1; ++k) {
        links2(k, 0) = links2(k, 1) = NA_REAL;
    }


    NumericMatrix merge(n-1, 2);
    internal_generate_merge(n, links2, merge);

    NumericVector order(n, NA_REAL);
    internal_generate_order(n, merge, order);

    if (verbose) GENIECLUST_PRINT("[genieclust] Done.\n");

    return List::create(
        _["merge"]  = merge,
        _["height"] = height,
        _["order"]  = order
    );
}
