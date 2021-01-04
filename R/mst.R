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




#' @title Minimum Spanning Tree of the Pairwise Distance Graph
#'
#' @description
#' An parallelised implementation of a Jarník (Prim/Dijkstra)-like
#' algorithm for determining
#' a(*) minimum spanning tree (MST) of a complete undirected graph
#' representing a set of n points
#' with weights given by a pairwise distance matrix.
#'
#' (*) Note that there might be multiple minimum trees spanning a given graph.
#'
#' @details
#' If \code{d} is a numeric matrix of size \eqn{n p},
#' the \eqn{n (n-1)/2} distances are computed on the fly, so that \eqn{O(n M)}
#' memory is used.
#'
#'
#' The algorithm is parallelised; set the \code{OMP_NUM_THREADS} environment
#' variable \code{\link[base]{Sys.setenv}} to control the number of threads
#' used.
#'
#' Time complexity is \eqn{O(n^2)} for the method accepting an object of
#' class \code{dist} and \eqn{O(p n^2)} otherwise.
#'
#' If \code{M} >= 2, then the mutual reachability distance \eqn{m(i,j)} with smoothing
#' factor \code{M} (see Campello et al. 2015)
#' is used instead of the chosen "raw" distance \eqn{d(i,j)}.
#' It holds \eqn{m(i, j)=\max(d(i,j), c(i), c(j))}, where \eqn{c(i)} is
#' \eqn{d(i, k)} with \eqn{k} being the (\code{M}-1)-th nearest neighbour of \eqn{i}.
#' This makes "noise" and "boundary" points being "pulled away" from each other.
#' Genie++ clustering algorithm (see \code{\link{gclust}})
#' with respect to the mutual reachability distance gains the ability to
#' identify some observations are noise points.
#'
#' Note that the case \code{M} = 2 corresponds to the original distance, but we are
#' determining the 1-nearest neighbours separately as well, which is a bit
#' suboptimal; you can file a feature request if this makes your data analysis
#' tasks too slow.
#'
#'
#' @seealso
#' \code{\link{emst_mlpack}()} for a very fast alternative
#' in case of (very) low-dimensional Euclidean spaces (and \code{M} = 1).
#'
#'
#' @references
#' Jarník V., O jistém problému minimálním,
#' Práce Moravské Přírodovědecké Společnosti 6 (1930) 57–63.
#'
#' Olson C.F., Parallel algorithms for hierarchical clustering,
#' Parallel Comput. 21 (1995) 1313–1325.
#'
#' Prim R., Shortest connection networks and some generalisations,
#' Bell Syst. Tech. J. 36 (1957) 1389–1401.
#'
#' Campello R., Moulavi D., Zimek A., Sander J.,
#' Hierarchical density estimates for data clustering, visualization,
#' and outlier detection,
#' ACM Transactions on Knowledge Discovery from Data 10(1) (2015) 5:1–5:51.
#'
#'
#' @param d either a numeric matrix (or an object coercible to one,
#'     e.g., a data frame with numeric-like columns) or an
#'     object of class \code{dist}, see \code{\link[stats]{dist}}.
#' @param distance metric used to compute the linkage, one of:
#'     \code{"euclidean"} (synonym: \code{"l2"}),
#'     \code{"manhattan"} (a.k.a. \code{"l1"} and \code{"cityblock"}),
#'     \code{"cosine"}.
#' @param M smoothing factor; \code{M} = 1 gives the selected \code{distance};
#'     otherwise, the mutual reachability distance is used.
#' @param verbose logical; whether to print diagnostic messages
#'     and progress information.
#' @param cast_float32 logical; whether to compute the distances using 32-bit
#'     instead of 64-bit precision floating-point arithmetic (up to 2x faster).
#' @param ... further arguments passed to or from other methods.
#'
#' @return
#' Matrix of class \code{mst} with n-1 rows and 3 columns:
#' \code{from}, \code{to} and \code{dist}. It holds \code{from} < \code{to}.
#' Moreover, \code{dist} is sorted nondecreasingly.
#' The i-th row gives the i-th edge of the MST.
#' \code{(from[i], to[i])} defines the vertices (in 1,...,n)
#' and \code{dist[i]} gives the weight, i.e., the
#' distance between the corresponding points.
#'
#' The \code{method} attribute gives the name of the distance used.
#' The \code{Labels} attribute gives the labels of all the input points.
#'
#' If \code{M} > 1, the \code{nn} attribute gives the indices of the \code{M}-1
#' nearest neighbours of each point.
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
mst.default <- function(d,
    distance=c("euclidean", "l2", "manhattan", "cityblock", "l1", "cosine"),
    M=1L,
    cast_float32=TRUE,
    verbose=FALSE, ...)
{
    distance <- match.arg(distance)
    d <- as.matrix(d)

    result <- .mst.default(d, distance, M, cast_float32, verbose)
    attr(result, "method") <- if (M == 1L) distance else
        sprintf("mutual reachability distance (%s, M=%d)", distance, M)
    attr(result, "Labels") <- dimnames(d)[[1]]

    class(result) <- "mst"

    result
}


#' @export
#' @rdname mst
#' @method mst dist
mst.dist <- function(d,
    M=1L,
    verbose=FALSE, ...)
{
    result <- .mst.dist(d, M, verbose)
    attr(result, "method") <- if (M == 1L) attr(d, "method") else
        sprintf("mutual reachability distance (%s, M=%d)", attr(d, "method"), M)
    attr(result, "Labels") <- attr(d, "Labels")
    class(result) <- "mst"

    result
}


registerS3method("mst", "default", "mst.default")
registerS3method("mst", "dist",    "mst.dist")





#' @title Euclidean Minimum Spanning Tree
#'
#' @description
#' Provides access to an implementation of the Dual-Tree Borůvka
#' algorithm based on kd-trees from MLPACK. It is fast for (very) low-dimensional
#' Euclidean spaces. For higher dimensional spaces (say, over 5 features)
#' or other metrics, use the parallelised Prim-like algorithm implemented
#' in \code{\link{mst}()}.
#'
#'
#' @param X a numeric matrix (or an object coercible to one,
#'     e.g., a data frame with numeric-like columns)
#' @param verbose logical; whether to print diagnostic messages
#'
#' @return
#' An object of class \code{mst}, see \code{\link{mst}()} for details.
#'
#' @references
#' March W.B., Ram P., Gray A.G.,
#' Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis, and Applications,
#' Proc. ACM SIGKDD'10 (2010) 603-611, \url{https://mlpack.org/papers/emst.pdf}.
#'
#' Curtin R.R., Edel M., Lozhnikov M., Mentekidis Y., Ghaisas S., Zhang S.,
#' mlpack 3: A fast, flexible machine learning library,
#' Journal of Open Source Software 3(26), 726, 2018.
#'
#' @export
emst_mlpack <- function(X, verbose=FALSE)
{
    X <- as.matrix(X)

    if (!verbose)
        capture.output({mst <- .emst_mlpack(X)})
    else
        mst <- .emst_mlpack(X)

    structure(
        mst,
        class="mst",
        method="euclidean",
        Labels=dimnames(X)[[1]]
    )
}
