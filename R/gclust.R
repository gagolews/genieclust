# This file is part of the genieclust package for R.

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2020-2021, Marek Gagolewski <https://www.gagolewski.com>      #
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
    # TODO: allow one choose?
    # cumsum(height)
    rev(cummin(rev(height)))
}





#' @title The Genie++ Hierarchical Clustering Algorithm
#'
#' @description
#' A reimplementation of \emph{Genie} - a robust and outlier resistant
#' clustering algorithm (see Gagolewski, Bartoszuk, Cena, 2016).
#' The Genie algorithm is based on a minimum spanning tree (MST) of the
#' pairwise distance graph of a given point set.
#' Just like single linkage, it consumes the edges
#' of the MST in increasing order of weights. However, it prevents
#' the formation of clusters of highly imbalanced sizes; once the Gini index
#' (see \code{\link{gini_index}()}) of the cluster size distribution
#' raises above \code{gini_threshold}, a forced merge of a point group
#' of the smallest size is performed. Its appealing simplicity goes hand
#' in hand with its usability; Genie often outperforms
#' other clustering approaches on benchmark data,
#' such as \url{https://github.com/gagolews/clustering_benchmarks_v1}.
#'
#' The clustering can now also be computed with respect to the
#' mutual reachability distance (based, e.g., on the Euclidean metric),
#' which is used in the definition of the HDBSCAN* algorithm
#' (see Campello et al., 2015). If \code{M} > 1, then the mutual reachability
#' distance \eqn{m(i,j)} with smoothing factor \code{M} is used instead of the
#' chosen "raw" distance \eqn{d(i,j)}. It holds \eqn{m(i,j)=\max(d(i,j), c(i), c(j))},
#' where \eqn{c(i)} is \eqn{d(i,k)} with \eqn{k} being the
#' (\code{M}-1)-th nearest neighbour of \eqn{i}.
#' This makes "noise" and "boundary" points being "pulled away" from each other.
#'
#' The Genie correction together with the smoothing factor \code{M} > 1 (note that
#' \code{M} = 2 corresponds to the original distance) gives a robustified version of
#' the HDBSCAN* algorithm that is able to detect a predefined number of
#' clusters. Hence it does not dependent on the DBSCAN's somehow magical
#' \code{eps} parameter or the HDBSCAN's \code{min_cluster_size} one.
#'
#'
#' @details
#' Note that as in the case of all the distance-based methods,
#' the standardisation of the input features is definitely worth giving a try.
#'
#' If \code{d} is a numeric matrix or an object of class \code{dist},
#' \code{\link{mst}()} will be called to compute an MST, which generally
#' takes at most \eqn{O(n^2)} time (the algorithm we provide is parallelised,
#' environment variable \code{OMP_NUM_THREADS} controls the number of threads
#' in use). However, see \code{\link{emst_mlpack}()} for a very fast alternative
#' in the case of Euclidean spaces of (very) low dimensionality and \code{M} = 1.
#'
#' Given an minimum spanning tree, the algorithm runs in \eqn{O(n \sqrt{n})} time.
#' Therefore, if you want to test different \code{gini_threshold}s,
#' (or \code{k}s), it is best to explicitly compute the MST first.
#'
#' According to the algorithm's original definition,
#' the resulting partition tree (dendrogram) might violate
#' the ultrametricity property (merges might occur at levels that
#' are not increasing w.r.t. a between-cluster distance).
#' Departures from ultrametricity are corrected by applying
#' \code{height = rev(cummin(rev(height)))}.
#'
#'
#' @param d a numeric matrix (or an object coercible to one,
#'     e.g., a data frame with numeric-like columns) or an
#'     object of class \code{dist}, see \code{\link[stats]{dist}}
#'     or an object of class \code{mst}, see \code{\link{mst}()}.
#' @param gini_threshold threshold for the Genie correction, i.e.,
#'     the Gini index of the cluster size distribution;
#'     Threshold of 1.0 disables the correction.
#'     Low thresholds highly penalise the formation of small clusters.
#' @param distance metric used to compute the linkage, one of:
#'     \code{"euclidean"} (synonym: \code{"l2"}),
#'     \code{"manhattan"} (a.k.a. \code{"l1"} and \code{"cityblock"}),
#'     \code{"cosine"}.
#' @param verbose logical; whether to print diagnostic messages
#'     and progress information.
#' @param cast_float32 logical; whether to compute the distances using 32-bit
#'     instead of 64-bit precision floating-point arithmetic (up to 2x faster).
#' @param ... further arguments passed to other methods.
#' @param k the desired number of clusters to detect, \code{k} = 1 with \code{M} > 1
#'     acts as a noise point detector.
#' @param detect_noise whether the minimum spanning tree's leaves
#'     should be marked as noise points, defaults to \code{TRUE} if \code{M} > 1
#'     for compatibility with HDBSCAN*.
#' @param M smoothing factor; \code{M} <= 2 gives the selected \code{distance};
#'     otherwise, the mutual reachability distance is used.
#' @param postprocess one of \code{"boundary"} (default), \code{"none"}
#'     or \code{"all"};  in effect only if \code{M} > 1.
#'     By default, only "boundary" points are merged
#'     with their nearest "core" points (A point is a boundary point if it is
#'     a noise point and it's amongst its adjacent vertex's
#'     \code{M}-1 nearest neighbours). To force a classical
#'     k-partition of a data set (with no notion of noise),
#'     choose "all".
#'
#'
#' @return
#' \code{gclust()} computes the whole clustering hierarchy; it
#' returns a list of class \code{hclust},
#' see \code{\link[stats]{hclust}}. Use \code{link{cutree}()} to obtain
#' an arbitrary k-partition.
#'
#' \code{genie()} returns a \code{k}-partition - a vector with elements in 1,...,k,
#' whose i-th element denotes the i-th input point's cluster identifier.
#' Missing values (\code{NA}) denote noise points (if \code{detect_noise}
#' is \code{TRUE}).
#'
#' @seealso
#' \code{\link{mst}()} for the minimum spanning tree routines.
#'
#' \code{\link{adjusted_rand_score}()} (amongst others) for external
#' cluster validity measures (partition similarity scores).
#'
#'
#' @references
#' Gagolewski M., Bartoszuk M., Cena A.,
#' Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
#' \emph{Information Sciences} 363, 2016, 8-23.
#'
#' Campello R., Moulavi D., Zimek A., Sander J.,
#' Hierarchical density estimates for data clustering, visualization,
#' and outlier detection,
#' ACM Transactions on Knowledge Discovery from Data 10(1), 2015, 5:1–5:51.
#'
#'
#' @examples
#' library("datasets")
#' data("iris")
#' X <- iris[1:4]
#' h <- gclust(X)
#' y_pred <- cutree(h, 3)
#' y_test <- iris[,5]
#' plot(iris[,2], iris[,3], col=y_pred,
#'    pch=as.integer(iris[,5]), asp=1, las=1)
#' adjusted_rand_score(y_test, y_pred)
#' pair_sets_index(y_test, y_pred)
#'
#' # Fast for low-dimensional Euclidean spaces:
#' h <- gclust(emst_mlpack(X))
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
    cast_float32=TRUE,
    verbose=FALSE, ...)
{
    stopifnot(gini_threshold >= 0.0, gini_threshold <= 1.0)
    distance <- match.arg(distance)
    d <- as.matrix(d)

    gclust.mst(mst.default(d, M=1L, distance=distance,
                verbose=verbose, cast_float32=cast_float32),
        gini_threshold=gini_threshold, verbose=verbose)
}


#' @export
#' @rdname gclust
#' @method gclust dist
gclust.dist <- function(d,
    gini_threshold=0.3,
    verbose=FALSE, ...)
{
    stopifnot(gini_threshold >= 0.0, gini_threshold <= 1.0)
    gclust.mst(mst.dist(d, M=1L, verbose=verbose),
        gini_threshold=gini_threshold, verbose=verbose)
}


#' @export
#' @rdname gclust
#' @method gclust mst
gclust.mst <- function(d,
    gini_threshold=0.3,
    verbose=FALSE, ...)
{
    stopifnot(gini_threshold >= 0.0, gini_threshold <= 1.0)
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
genie.default <- function(d,
    k,
    gini_threshold=0.3,
    distance=c("euclidean", "l2", "manhattan", "cityblock", "l1", "cosine"),
    M=1L,
    postprocess=c("boundary", "none", "all"),
    detect_noise=M>1L,
    cast_float32=TRUE,
    verbose=FALSE, ...)
{
    stopifnot(gini_threshold >= 0.0, gini_threshold <= 1.0)
    postprocess <- match.arg(postprocess)
    distance <- match.arg(distance)

    d <- as.matrix(d)

    genie.mst(mst.default(d, M=M, distance=distance,
                verbose=verbose, cast_float32=cast_float32),
        k=k,
        gini_threshold=gini_threshold,
        postprocess=postprocess,
        detect_noise=detect_noise,
        verbose=verbose)
}


#' @export
#' @rdname gclust
#' @method genie dist
genie.dist <- function(d,
    k,
    gini_threshold=0.3,
    M=1L,
    postprocess=c("boundary", "none", "all"),
    detect_noise=M>1L,
    verbose=FALSE, ...)
{
    stopifnot(gini_threshold >= 0.0, gini_threshold <= 1.0)
    postprocess <- match.arg(postprocess)

    genie.mst(mst.dist(d, M=M, verbose=verbose),
        k=k,
        gini_threshold=gini_threshold,
        postprocess=postprocess,
        detect_noise=detect_noise,
        verbose=verbose)
}


#' @export
#' @rdname gclust
#' @method genie mst
genie.mst <- function(d,
    k,
    gini_threshold=0.3,
    postprocess=c("boundary", "none", "all"),
    detect_noise=FALSE,
    verbose=FALSE, ...)
{
    stopifnot(gini_threshold >= 0.0, gini_threshold <= 1.0)
    postprocess <- match.arg(postprocess)

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
