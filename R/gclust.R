# This file is part of the genieclust package for R.

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2020-2025, Marek Gagolewski <https://www.gagolewski.com>      #
#                                                                              #
#                                                                              #
#   This program is free software: you can redistribute it and/or modify       #
#   it under the terms of the GNU Affero General Public License                #
#   Version 3, 19 November 2007, published by the Free Software Foundation.    #
#   This program is distributed in the hope that it will be useful,            #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the               #
#   GNU Affero General Public License Version 3 for more details.              #
#   You should have received a copy of the License along with this program.    #
#   If this is not the case, refer to <https://www.gnu.org/licenses/>.         #
#                                                                              #
# ############################################################################ #




.correct_height <- function(height)
{
    # correction for the departure from ultrametricity
    # TODO: allow one to choose?
    # cumsum(height)
    rev(cummin(rev(height)))
}





#' @title Hierarchical Clustering Algorithm Genie
#'
#' @description
#' A reimplementation of \emph{Genie} - a robust and outlier resistant
#' clustering algorithm (see Gagolewski, Bartoszuk, Cena, 2016).
#' The Genie algorithm is based on the minimum spanning tree (MST) of the
#' pairwise distance graph of a given point set.
#' Just like the single linkage, it consumes the edges
#' of the MST in an increasing order of weights. However, it prevents
#' the formation of clusters of highly imbalanced sizes; once the Gini index
#' (see \code{\link{gini_index}()}) of the cluster size distribution
#' raises above \code{gini_threshold}, the merging of a point group
#' of the smallest size is enforced.
#'
#' Genie's simplicity goes hand in hand with its usability; it often
#' outperforms other clustering approaches on benchmark data,
#' such as \url{https://github.com/gagolews/clustering-benchmarks}.
#'
#' The clustering can now also be computed with respect to the
#' mutual reachability distances (based, e.g., on the Euclidean metric),
#' which is used in the definition of the HDBSCAN* algorithm
#' (see Campello et al., 2013). If \eqn{M>1}, then the mutual reachability
#' distance \eqn{m(i,j)} with a smoothing factor \eqn{M} is used instead of the
#' chosen "raw" distance \eqn{d(i,j)}.  It holds
#' \eqn{m(i,j)=\max(d(i,j), c(i), c(j))}, where the core distance \eqn{c(i)} is
#' the distance to the \eqn{i}-th point's (\eqn{M-1})-th
#' nearest neighbour.  This makes "noise" and "boundary" points being
#' more "pulled away" from each other.
#'
#' The Genie correction together with the smoothing factor \eqn{M>1}
#' (note that \eqn{M=2} corresponds to the original distance) gives
#' a robustified version of the HDBSCAN* algorithm that is able to detect
#' a predefined number of clusters. Hence it does not dependent on the DBSCAN's
#' somewhat magical \code{eps} parameter or the HDBSCAN's
#' \code{min_cluster_size} one.
#'
#'
#' @details
#' As in the case of all the distance-based methods,
#' the standardisation of the input features is definitely worth giving a try.
#' Oftentimes, more sophisticated feature engineering (e.g., dimensionality
#' reduction) will lead to more meaningful results.
#'
#' If \code{d} is a numeric matrix or an object of class \code{dist},
#' \code{\link{mst}()} will be called to compute an MST, which generally
#' takes at most \eqn{O(n^2)} time. However, by default, a faster algorithm
#' based on K-d trees is selected automatically for low-dimensional Euclidean
#' spaces; see \code{\link[quitefastmst]{mst_euclid}}.
#'
#' Once a minimum spanning tree is determined, the Genie algorithm runs in
#' \eqn{O(n \sqrt{n})} time.  If you want to test different
#' \code{gini_threshold}s or \code{k}s,  it is best to explicitly compute
#' the MST first.
#'
#' According to the algorithm's original definition,
#' the resulting partition tree (dendrogram) might violate
#' the ultrametricity property (merges might occur at levels that
#' are not increasing w.r.t. a between-cluster distance).
#' \code{gclust()} automatically corrects departures from
#' ultrametricity by applying \code{height = rev(cummin(rev(height)))}.
#'
#'
#' @param d a numeric matrix (or an object coercible to one,
#'     e.g., a data frame with numeric-like columns) or an
#'     object of class \code{dist} (see \code{\link[stats]{dist}}),
#'     or an object of class \code{mst} (\code{\link{mst}}).
#' @param gini_threshold threshold for the Genie correction, i.e.,
#'     the Gini index of the cluster size distribution;
#'     threshold of 1.0 leads to the single linkage algorithm;
#'     low thresholds highly penalise the formation of small clusters.
#' @param distance metric used to compute the linkage, one of:
#'     \code{"euclidean"} (synonym: \code{"l2"}),
#'     \code{"manhattan"} (a.k.a. \code{"l1"} and \code{"cityblock"}),
#'     \code{"cosine"}.
#' @param verbose logical; whether to print diagnostic messages
#'     and progress information.
#' @param ... further arguments passed to \code{\link{mst}()}.
#' @param k the desired number of clusters to detect, \eqn{k=1} with
#'      \eqn{M>1} acts as a noise point detector.
#' @param detect_noise whether the minimum spanning tree's leaves
#'     should be marked as noise points, defaults to \code{TRUE} if \eqn{M>1}
#'     for compatibility with HDBSCAN*.
#' @param M smoothing factor; \eqn{M \leq 2} gives the selected \code{distance};
#'     otherwise, the mutual reachability distance is used.
#' @param postprocess one of \code{"boundary"} (default), \code{"none"}
#'     or \code{"all"};  in effect only if \eqn{M > 1}.
#'     By default, only "boundary" points are merged
#'     with their nearest "core" points (A point is a boundary point if it is
#'     a noise point and it is amongst its adjacent vertex's
#'     (\eqn{M-1})-th nearest neighbours). To force a classical
#'     k-partition of a data set (with no notion of noise),
#'     choose \code{"all"}.
#'
#'
#' @return
#' \code{gclust()} computes the whole clustering hierarchy; it
#' returns a list of class \code{hclust},
#' see \code{\link[stats]{hclust}}. Use \code{\link[stats]{cutree}} to obtain
#' an arbitrary \code{k}-partition.
#'
#' \code{genie()} returns a \code{k}-partition - a vector whose i-th element
#' denotes the i-th input point's cluster label between 1 and \code{k}
#' Missing values (\code{NA}) denote noise points (if \code{detect_noise}
#' is \code{TRUE}).
#'
#' @seealso
#' \code{\link{mst}()} for the minimum spanning tree routines.
#'
#' \code{\link{normalized_clustering_accuracy}()} (amongst others) for external
#' cluster validity measures.
#'
#'
#' @references
#' Gagolewski, M., Bartoszuk, M., Cena, A.,
#' Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
#' \emph{Information Sciences} 363, 2016, 8-23,
#' \doi{10.1016/j.ins.2016.05.003}.
#'
#' Campello, R.J.G.B., Moulavi, D., Sander, J.,
#' Density-based clustering based on hierarchical density estimates,
#' \emph{Lecture Notes in Computer Science} 7819, 2013, 160-172,
#' \doi{10.1007/978-3-642-37456-2_14}.
#'
#' Gagolewski, M., Cena, A., Bartoszuk, M., Brzozowski, L.,
#' Clustering with minimum spanning trees: How good can it be?,
#' \emph{Journal of Classification} 42, 2025, 90-112,
#' \doi{10.1007/s00357-024-09483-1}.
#'
#'
#' @examples
#' library("datasets")
#' data("iris")
#' X <- jitter(as.matrix(iris[2:3]))
#' h <- gclust(X)
#' y_pred <- cutree(h, 3)
#' y_test <- as.integer(iris[,5])
#' plot(X, col=y_pred, pch=y_test, asp=1, las=1)
#' adjusted_rand_score(y_test, y_pred)
#' normalized_clustering_accuracy(y_test, y_pred)
#'
#' y_pred2 <- genie(X, 3, M=5)  # clustering wrt 5-mutual reachability distance
#' plot(X[,1], X[,2], col=y_pred2, pch=y_test, asp=1, las=1)
#' noise <- is.na(y_pred2)  # noise/boundary points
#' points(X[noise, ], col="gray", pch=10)
#' normalized_clustering_accuracy(y_test[!noise], y_pred2[!noise])
#'
#' @rdname gclust
#' @export
gclust <- function(d, ...)
{
    UseMethod("gclust")
}



#' @export
#' @rdname gclust
#' @method gclust default
gclust.default <- function(d,
    gini_threshold=0.3,
    distance=c("euclidean", "l2", "manhattan", "cityblock", "l1", "cosine"),
    verbose=FALSE, ...)
{
    distance <- match.arg(distance)
    gclust.mst(
        mst.default(d, distance=distance, verbose=verbose, ...),
        gini_threshold=gini_threshold, verbose=verbose
    )
}


#' @export
#' @rdname gclust
#' @method gclust dist
gclust.dist <- function(d,
    gini_threshold=0.3,
    verbose=FALSE, ...)
{
    gclust.mst(
        mst.dist(d, verbose=verbose, ...),
        gini_threshold=gini_threshold, verbose=verbose
    )
}


#' @export
#' @rdname gclust
#' @method gclust mst
gclust.mst <- function(
    d,
    gini_threshold=0.3,
    verbose=FALSE, ...)
{
    gini_threshold <- as.double(gini_threshold)[1]
    stopifnot(gini_threshold >= 0.0, gini_threshold <= 1.0)
    verbose <- !identical(verbose, FALSE)

    result <- .gclust(d, gini_threshold, verbose)

    result[["height"]] <- .correct_height(result[["height"]])
    result[["labels"]] <- attr(d, "Labels") # yes, >L<abels
    result[["method"]] <- sprintf("Genie(%g)", gini_threshold)
    result[["call"]]   <- match.call()
    result[["dist.method"]] <- attr(d, "method")
    class(result) <- "hclust"

    result
}


#' @rdname gclust
#' @export
genie <- function(d, ...)
{
    UseMethod("genie")
}


#' @export
#' @rdname gclust
#' @method genie default
genie.default <- function(
    d,
    k,
    gini_threshold=0.3,
    distance=c("euclidean", "l2", "manhattan", "cityblock", "l1", "cosine"),
    M=1L,
    postprocess=c("boundary", "none", "all"),
    detect_noise=M>1L,
    verbose=FALSE,
    ...)
{
    distance <- match.arg(distance)
    postprocess <- match.arg(postprocess)
    genie.mst(
        mst.default(d, M=M, distance=distance, verbose=verbose, ...),
        k=k,
        gini_threshold=gini_threshold,
        postprocess=postprocess,
        detect_noise=detect_noise,
        verbose=verbose
    )
}


#' @export
#' @rdname gclust
#' @method genie dist
genie.dist <- function(
    d,
    k,
    gini_threshold=0.3,
    M=1L,
    postprocess=c("boundary", "none", "all"),
    detect_noise=M>1L,
    verbose=FALSE,
    ...)
{
    postprocess <- match.arg(postprocess)
    genie.mst(
        mst.dist(d, M=M, verbose=verbose, ...),
        k=k,
        gini_threshold=gini_threshold,
        postprocess=postprocess,
        detect_noise=detect_noise,
        verbose=verbose
    )
}


#' @export
#' @rdname gclust
#' @method genie mst
genie.mst <- function(d,
    k,
    gini_threshold=0.3,
    postprocess=c("boundary", "none", "all"),
    detect_noise=FALSE,
    verbose=FALSE,
    ...)
{
    gini_threshold <- as.double(gini_threshold)[1]
    stopifnot(gini_threshold >= 0.0, gini_threshold <= 1.0)
    postprocess <- match.arg(postprocess)
    verbose <- !identical(verbose, FALSE)
    detect_noise <- !identical(detect_noise, FALSE)

    structure(
        .genie(d, k, gini_threshold, postprocess, detect_noise, verbose),
        names=attr(d, "Labels")
    )
}


registerS3method("gclust", "default", "gclust.default")
registerS3method("gclust", "dist",    "gclust.dist")
registerS3method("gclust", "mst",     "gclust.mst")

registerS3method("genie", "default", "genie.default")
registerS3method("genie", "dist",    "genie.dist")
registerS3method("genie", "mst",     "genie.mst")
