# This file is part of the genieclust package for R.
#
# Copyright (C) 2018-2020 Marek Gagolewski (https://www.gagolewski.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License
# Version 3, 19 November 2007, published by the Free Software Foundation.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License Version 3 for more details.
# You should have received a copy of the License along with this program.
# If not, see <https://www.gnu.org/licenses/>.




.correct_height <- function(height)
{
    # correction for the departure from ultrametricity
    if (any(height < 0)) {
        height <- rev(cummin(rev(height)))
    }
    height
}





#' @title The Genie Hierarchical Clustering Algorithm
#'
#' @description
#' A reimplementation of the robust and outlier resistant
#' Genie (see Gagolewski, Bartoszuk, Cena, 2016) clustering algorithm.
#'
#' Given an minimum spanning tree (see \code{\link{mst}()}
#' or \code{\link{emst_mlpack}()} (the latter is very fast in the case
#' of Euclidean spaces of low dimensionality), the algorithm runs
#' in \eqn{O(n \sqrt{n})} time (computing a minimum spanning tree
#' takes a most \eqn{O(n^2)} time).
#'
#' @details
#' Note that as in the case of all the distance-based methods, standardisation
#' of the input features is definitely worth giving a try.
#'
#' If \code{d} is a numeric matrix or an object of class \code{dist},
#' \code{\link{mst}()} will be called to compute the minimum spanning tree.
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
#'     \code{"cosine"}
#' @param verbose logical; whether to print diagnostic messages
#'     and progress information
#' @param cast_float32 logical; whether to compute the distances using 32-bit
#'     instead of 64-bit precision floating-point arithmetic (up to 2x faster)
#' @param ... further arguments passed to other methods, such as
#' \code{\link{mst}}
#'
#' @return
#' A list of class \code{hclust}, see \code{\link[stats]{hclust}}.
#'
#' @references
#' Gagolewski M., Bartoszuk M., Cena A.,
#' Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
#' \emph{Information Sciences} 363, 2016, pp. 8-23.
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
#' if (require("emstreeR")) h <- gclust(emst_mlpack(X))
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
    verbose=FALSE,
    ...)
{
    stopifnot(gini_threshold >= 0.0, gini_threshold <= 1.0)
    distance <- match.arg(distance)
    d <- as.matrix(d)

    gclust.mst(mst.default(d, M=1, distance=distance,
                verbose=verbose, cast_float32=cast_float32),
        gini_threshold=gini_threshold, verbose=verbose, ...)
}


#' @export
#' @rdname gclust
#' @method gclust dist
gclust.dist <- function(d,
    gini_threshold=0.3,
    verbose=FALSE,
    ...)
{
    stopifnot(gini_threshold >= 0.0, gini_threshold <= 1.0)
    gclust.mst(mst.dist(d, M=1, verbose=verbose),
        gini_threshold=gini_threshold, verbose=verbose, ...)
}


#' @export
#' @rdname gclust
#' @method gclust mst
gclust.mst <- function(d,
    gini_threshold=0.3,
    verbose=FALSE,
    ...)
{
    stopifnot(gini_threshold >= 0.0, gini_threshold <= 1.0)
    result <- .gclust(d, gini_threshold, verbose)

    result[["height"]] <- .correct_height(result[["height"]])
    result[["labels"]] <- attr(d, "Labels")
    result[["method"]] <- sprintf("Genie(%g)", gini_threshold)
    result[["call"]]   <- match.call()
    result[["dist.method"]] <- attr(d, "method")
    class(result) <- "hclust"

    result
}


registerS3method("gclust", "default", "gclust.default")
registerS3method("gclust", "dist",    "gclust.dist")
registerS3method("gclust", "mst",     "gclust.mst")
