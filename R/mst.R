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
#' If \code{d} is a matrix and the use of Euclidean distance is requested
#' (the default), then \code{\link{mst_euclid}} is called
#' to determine the MST.  As it is based on K-d trees, it is quite fast
#' in low dimensional spaces, even for 10M points.
#'
#' Otherwise, a much slower implementation of the Jarnik (Prim/Dijkstra)-like
#' method, which requires \eqn{O(n^2)} time.  The algorithm is parallelised;
#' the number of threads is determined by the \code{OMP_NUM_THREADS} environment
#' variable. As a rule of thumb, datasets up to 100k points
#' should be processed quite quickly (within a few minutes).
#'
#' If \eqn{M>1}, then the mutual reachability distance \eqn{m(i,j)}
#' with the smoothing factor \eqn{M} (see Campello et al. 2013)
#' is used instead of the chosen "raw" distance \eqn{d(i,j)}.
#' It holds \eqn{m(i, j)=\max\{d(i,j), c(i), c(j)\}}, where \eqn{c(i)} is
#' the core distance, i.e., the distance between the \eqn{i}-th point and
#' its (\code{M}-1)-th nearest neighbour.
#' This makes "noise" and "boundary" points being "pulled away" from each other.
#' The Genie++ clustering algorithm (see \code{\link{gclust}}) with respect to
#' the mutual reachability distance can mark some observations as noise points.
#'
#'
#' @seealso
#' \code{\link{mst_euclid}}
#'
#' @references
#' Jarnik V., O jistem problemu minimalnim,
#' \emph{Prace Moravske Prirodovedecke Spolecnosti} 6, 1930, 57-63.
#'
#' Olson C.F., Parallel algorithms for hierarchical clustering,
#' \emph{Parallel Computing} 21, 1995, 1313-1325.
#'
#' Prim R., Shortest connection networks and some generalisations,
#' \emph{The Bell System Technical Journal} 36(6), 1957, 1389-1401.
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
#' @param M smoothing factor; \code{M} = 1 selects the requested
#'      \code{distance}; otherwise, the corresponding degree-\code{M} mutual
#'      reachability distance is used; \code{M} should be rather small,
#'      say, \eqn{\leq 20}
#'
#' @param verbose logical; whether to print diagnostic messages
#'     and progress information
#'
#' @param ... further arguments passed to or from other methods,
#'     in particular, to \code{\link{mst_euclid}}
#'
#'
#' @return
#' Returns a numeric matrix of class \code{mst} with \eqn{n-1} rows and
#' three columns: \code{from}, \code{to}, and \code{dist} sorted
#' nondecreasingly. Its i-th row specifies the i-th edge of the MST
#' which is incident to the vertices \code{from[i]} and \code{to[i]} with
#' \code{from[i] < to[i]}  (in 1,...,n)
#' and \code{dist[i]} gives the corresponding weight, i.e., the
#' distance between the point pair.
#'
#' The \code{Size} attribute specifies the number of points, \eqn{n}.
#' The \code{Labels} attribute gives the labels of the input points (optionally).
#' The \code{method} attribute provides the name of the distance used.
#'
#' If \code{M} > 1, the \code{nn.index} attribute gives the indices
#' of the \code{M}-1 nearest neighbours of each point
#' and \code{nn.dist} provides the corresponding distances
#' (if returned by the underlying function).
#'
#'
#' @examples
#' library("datasets")
#' data("iris")
#' X <- jitter(as.matrix(iris[1:2]))  # some data
#' T <- mst(X)
#' plot(X, asp=1, las=1)
#' segments(X[T[, 1], 1], X[T[, 1], 2],
#'          X[T[, 2], 1], X[T[, 2], 2])
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
    verbose=FALSE, ...)
{
    d <- as.matrix(d)
    M <- as.integer(M)[1]
    distance <- match.arg(distance)
    verbose <- !identical(verbose, FALSE)

    if (distance %in% c("euclidean", "l2")) {
        .res <- mst_euclid(d, M, ..., verbose=verbose)
        result <- cbind(.res[["mst.index"]], .res[["mst.dist"]])
        attr(result, "nn.index")  <- .res[["nn.index"]]
        attr(result, "nn.dist")   <- .res[["nn.dist"]]
    }
    else {
        result <- .oldmst.matrix(d, distance, M, ..., verbose=verbose)
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
    result <- .oldmst.dist(d, M, verbose)

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
