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




#' @title Minimum Spanning Tree of the Pairwise Distance Graph
#'
#' @description
#' Determine a(*) minimum spanning tree (MST) of the complete
#' undirected graph representing a set of \eqn{n} points
#' whose weights correspond to the pairwise distances between the points.
#'
#'
#' @details
#' (*) Note that if the distances are non unique,
#' there might be multiple minimum trees spanning a given graph.
#'
#'
#' Two MST algorithms are available.
#' First, our implementation of the Jarnik (Prim/Dijkstra)-like method
#' requires \eqn{O(n^2)} time. The algorithm is parallelised; the number of
#' threads is determined by the \code{OMP_NUM_THREADS} environment variable;
#' see \code{\link[base]{Sys.setenv}}. This method is recommended for
#' high-dimensional spaces. As a rule of thumb, datasets up to 100000 points
#' should be processed quite quickly. For 1M points, give it an hour or so.
#'
#' Second, we give access to the implementation of the Dual-Tree Boruvka
#' algorithm from the \code{mlpack} library. The algorithm is based on K-d trees
#' and is very fast but only for low-dimensional Euclidean spaces
#' (due to the curse of dimensionality). The Jarnik algorithm should be
#' used if there are more than 5-10 features.
#'
#'
#' If \code{d} is a numeric matrix of size \eqn{n} by \eqn{p}, representing
#' \eqn{n} points in a \eqn{p}-dimensional space,
#' the \eqn{n (n-1)/2} distances are computed on the fly: the algorithms
#' requires \eqn{O(n)} memory.
#'
#' If \code{M} >= 2, then the mutual reachability distance \eqn{m(i,j)}
#' with the smoothing factor \code{M} (see Campello et al. 2013)
#' is used instead of the chosen "raw" distance \eqn{d(i,j)}.
#' It holds \eqn{m(i, j)=\max(d(i,j), c(i), c(j))}, where \eqn{c(i)} is
#' \eqn{d(i, k)} with \eqn{k} being the (\code{M}-1)-th nearest neighbour of \eqn{i}.
#' This makes "noise" and "boundary" points being "pulled away" from each other.
#' The Genie++ clustering algorithm (see \code{\link{gclust}}) with respect to
#' the mutual reachability distance can mark some observations are noise points.
#'
#' Note that the case \code{M} = 2 corresponds to the original distance, but we
#' return the (1-)nearest neighbours as well.
#'
#'
#' @seealso
#' \code{\link{emst_mlpack}()} for a very fast alternative
#' in the case of (very) low-dimensional Euclidean spaces (and \code{M} = 1).
#'
#'
#' @references
#' Jarnik V., O jistem problemu minimalnim,
#' \emph{Prace Moravske Prirodovedecke Spolecnosti} 6, 1930, 57-63.
#'
#' Olson C.F., Parallel algorithms for hierarchical clustering,
#' \emph{Parallel Comput.} 21, 1995, 1313-1325.
#'
#' Prim R., Shortest connection networks and some generalisations,
#' \emph{Bell Syst. Tech. J.} 36, 1957, 1389-1401.
#'
#' March W.B., Ram P., Gray A.G.,
#' Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis, and Applications,
#' \emph{Proc. ACM SIGKDD'10}, 2010, 603-611,
#' \url{https://mlpack.org/papers/emst.pdf}.
#'
#' Curtin R.R., Edel M., Lozhnikov M., Mentekidis Y., Ghaisas S., Zhang S.,
#' mlpack 3: A fast, flexible machine learning library,
#' \emph{Journal of Open Source Software} 3(26), 2018, 726.
#'
#' Campello R.J.G.B., Moulavi D., Sander J.,
#' Density-based clustering based on hierarchical density estimates,
#' \emph{Lecture Notes in Computer Science} 7819, 2013, 160-172,
#' \doi{10.1007/978-3-642-37456-2_14}.
#'
#'
#' @param d either a numeric matrix (or an object coercible to one,
#'     e.g., a data frame with numeric-like columns) or an
#'     object of class \code{dist}; see \code{\link[stats]{dist}}
#'
#' @param distance metric used in the case where \code{d} is a matrix; one of:
#'     \code{"euclidean"} (synonym: \code{"l2"}),
#'     \code{"manhattan"} (a.k.a. \code{"l1"} and \code{"cityblock"}),
#'     \code{"cosine"}
#'
#' @param M smoothing factor; \code{M} = 1 gives the selected \code{distance};
#'     otherwise, the mutual reachability distance is used
#'
#' @param algorithm MST algorithm to use: \code{"auto"} (default),
#'     \code{"jarnik"}, or \code{"mlpack"};
#'     if \code{"auto"}, select \code{"mlpack"} for low-dimensional Euclidean
#'     spaces and \code{"jarnik"} otherwise
#'
#' @param leaf_size size of leaves in the K-d tree (\code{"mlpack"});
#'     controls the trade-off between speed and memory consumption
#'
#' @param cast_float32 logical; whether to compute the distances using 32-bit
#'     instead of 64-bit precision floating-point arithmetic (up to 2x faster)
#'
#' @param verbose logical; whether to print diagnostic messages
#'     and progress information
#'
#' @param ... further arguments passed to or from other methods
#'
#'
#' @return
#' Returns a numeric matrix of class \code{mst} with n-1 rows and 3 columns:
#' \code{from}, \code{to}, and \code{dist} sorted nondecreasingly.
#' Its i-th row specifies the i-th edge of the MST which is incident to the
#' vertices \code{from[i]} and \code{to[i]} \code{from[i] < to[i]}  (in 1,...,n)
#' and \code{dist[i]} gives the corresponding weight, i.e., the
#' distance between the point pair.
#'
#' The \code{Size} attribute specifies the number of points, \eqn{n}.
#' The \code{Labels} attribute gives the labels of the input points (optionally).
#' The \code{method} attribute gives the name of the distance used.
#'
#' If \code{M} > 1, the \code{nn} attribute gives the indices of the \code{M}-1
#' nearest neighbours of each point.
#'
#'
#' @examples
#' library("datasets")
#' data("iris")
#' X <- iris[1:4]
#' tree <- mst(X)
#'
#' @rdname mst
#' @export
mst <- function(d, ...)
{
    UseMethod("mst")
}



#' @export
#' @rdname mst
#' @method mst default
mst.default <- function(
    d,
    distance=c("euclidean", "l2", "manhattan", "cityblock", "l1", "cosine"),
    M=1L,
    algorithm=c("auto", "jarnik", "mlpack"),
    leaf_size=1L,
    cast_float32=FALSE,
    verbose=FALSE, ...)
{
    d <- as.matrix(d)
    distance <- match.arg(distance)
    algorithm <- match.arg(algorithm)
    cast_float32 <- !identical(cast_float32, FALSE)
    verbose <- !identical(verbose, FALSE)
    M <- as.integer(M)[1]
    leaf_size <- as.integer(leaf_size)[1]

    if (algorithm == "auto") {
        if (distance %in% c("euclidean", "l2") && ncol(d) <= 7L && M == 1L)
            algorithm <- "mlpack"
        else
            algorithm <- "jarnik"
    }

    if (algorithm == "mlpack") {
        stopifnot(M == 1L)
        stopifnot(distance %in% c("euclidean", "l2"))
        result <- .emst_mlpack(d, leaf_size, verbose)
    }
    else {
        result <- .mst.default(d, distance, M, cast_float32, verbose)
    }

    stopifnot(result[, 1] < result[, 2])
    stopifnot(!is.unsorted(result[, 3]))

    structure(
        result,
        class="mst",
        Size=nrow(d),
        Labels=dimnames(d)[[1]],  # dist() returns `Labels`, not `labels`
        method=if (M == 1L) distance else
            sprintf("mutual reachability distance (%s, M=%d)", distance, M)
    )
}


#' @export
#' @rdname mst
#' @method mst dist
mst.dist <- function(
    d,
    M=1L,
    verbose=FALSE, ...)
{
    #cast_float32 <- !identical(cast_float32, FALSE)
    verbose <- !identical(verbose, FALSE)
    M <- as.integer(M)[1]
    result <- .mst.dist(d, M, verbose)

    structure(
        result,
        class="mst",
        Size=nrow(result)+1,
        Labels=attr(d, "Labels"),  # dist() returns `Labels`, not `labels`
        method=if (M == 1L) attr(d, "method") else
            sprintf("mutual reachability distance (%s, M=%d)", attr(d, "method"), M)
    )
}


registerS3method("mst", "default", "mst.default")
registerS3method("mst", "dist",    "mst.dist")





#' @title Euclidean Minimum Spanning Tree [DEPRECATED]
#'
#' @description
#' This function is deprecated. Use \code{\link{mst}()} instead.
#'
#'
#' @param d a numeric matrix (or an object coercible to one,
#'     e.g., a data frame with numeric-like columns)
#'
#' @param leaf_size size of leaves in the K-d tree,
#'     controls the trade-off between speed and memory consumption
#'
#' @param verbose logical; whether to print diagnostic messages
#'
#'
#' @return
#' An object of class \code{mst}, see \code{\link{mst}()} for details.
#'
#' @export
emst_mlpack <- function(d, leaf_size=1, verbose=FALSE)
{
    # if (!requireNamespace("mlpack", quietly=TRUE)) {
    #     warning("Package `mlpack` is not installed. Using mst() instead.")
    #     return(mst.default(d, verbose=verbose, cast_float32=FALSE))
    # }
    # mst <- mlpack::emst(d, leaf_size=leaf_size, naive=naive, verbose=verbose)$output
    # mst[, 1] <- mst[, 1] + 1  # 0-based -> 1-based indexing
    # mst[, 2] <- mst[, 2] + 1  # 0-based -> 1-based indexing

    warning("`emst_mlpack` is deprecated; use `mst(..., algorithm='mlpack')` instead")
    mst.default(d, leaf_size=leaf_size, verbose=verbose, algorithm="mlpack")
}
