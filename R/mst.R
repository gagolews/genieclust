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




#' @title Minimum Spanning Tree of the Pairwise Distance Graph
#'
#' @description
#' An parallelised implementation of a Jarník (Prim/Dijkstra)-like
#' algorithm for determining
#' a(*) minimum spanning tree (MST) of a complete undirected graph
#' representing a set of n points
#' with weights given by a pairwise distance matrix.
#'
#' @details
#' If \code{d} is a numeric matrix of size n*p,
#' the n*(n-1)/2 distances are computed on the fly, so that \eqn{O(n M)}
#' memory is used.
#'
#' (*) Note that there might be multiple minimum trees spanning a given graph.
#'
#' The algorithm is parallelised; set the \code{OMP_NUM_THREADS} environment
#' variable \code{\link[base]{Sys.setenv}} to control the number of threads
#' used.
#'
#' Time complexity is \eqn{O(n^2)} for the method accepting an object of
#' class \code{dist} \eqn{and O(p n^2)} otherwise.
#'
#' If M>2, then the mutual reachability distance m(i,j) with smoothing factor M
#' (see Campello et al. 2015)
#' is used instead of the chosen "raw" distance d(i,j).
#' It holds m(i, j)=max(d(i,j), c(i), c(j)), where c(i) is
#' d(i, k) with k being the (M-1)-th nearest neighbour of i.
#' This makes "noise" and "boundary" points being "pulled away" from each other.
#' Genie++ clustering algorithm (see \code{\link{gclust}})
#' with respect to the mutual reachability distance gains the ability to
#' identify some observations are noise points.
#'
#'
#' @seealso
#' \code{\link{emst_mlpack}()} for a very fast alternative
#' in case of (very) low-dimensional Euclidean spaces (and M=1).
#'
#'
#' @references
#' V. Jarník, O jistém problému minimálním,
#' Práce Moravské Přírodovědecké Společnosti 6 (1930) 57–63.
#'
#' C.F. Olson, Parallel algorithms for hierarchical clustering,
#' Parallel Comput. 21 (1995) 1313–1325.
#'
#' R. Prim, Shortest connection networks and some generalisations,
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
#'     \code{"cosine"}
#' @param M smoothing factor; M<=2 gives the selected \code{distance};
#'     otherwise, the mutual reachability distance is used
#' @param verbose logical; whether to print diagnostic messages
#'     and progress information
#' @param cast_float32 logical; whether to compute the distances using 32-bit
#'     instead of 64-bit precision floating-point arithmetic (up to 2x faster)
#' @param ... further arguments passed to or from other methods.
#'
#' @return
#' Matrix of class \code{mst} with n-1 rows and 3 columns:
#' \code{from}, \code{to} and \code{dist}. It holds \code{from} < \code{to}.
#' Moreover, \code{dist} is sorted nondecreasingly.
#' The i-th row gives the i-th edge of the MST.
#' \code{(from[i], to[i])} defines the vertex indices (in 1,...,n)
#' and \code{dist[i]} gives the weight, i.e., the
#' distance between the corresponding points.
#'
#' The \code{dist.method} attribute gives the name of the distance used.
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
    M=1,
    cast_float32=TRUE,
    verbose=FALSE,
    ...)
{
    distance <- match.arg(distance)
    d <- as.matrix(d)

    result <- .mst.default(d, distance, M, cast_float32, verbose)
    attr(result, "dist.method") <- if (M <= 2L) distance else
        sprintf("mutual reachability distance (%s, M=%d)", distance, M)
    class(result) <- "mst"

    result
}


#' @export
#' @rdname mst
#' @method mst dist
mst.dist <- function(d,
    M=1,
    verbose=FALSE,
    ...)
{
    result <- .mst.dist(d, M, verbose)
    attr(result, "dist.method") <- if (M <= 2L) attr(d, "method") else
        sprintf("mutual reachability distance (%s, M=%d)", attr(d, "method"), M)
    class(result) <- "mst"

    result
}


registerS3method("mst", "default", "mst.default")
registerS3method("mst", "dist",    "mst.dist")





#' @title Euclidean Minimum Spanning Tree
#'
#' @description
#' Provides access to an implementation of the Dual-Tree Boruvka
#' algorithm based on kd-trees. It is fast for (very) low-dimensional
#' Euclidean spaces. For higher dimensional spaces (say, over 5 features)
#' or other metrics,
#' use the parallelised Prim-like algorithm implemented in \code{\link{mst}()}.
#'
#'
#' @details
#' Calls \code{emstreeR::mlpack_mst()} and converts the result
#' so that it is compatible with the output of \code{\link{mst}()}.
#'
#' If the \code{emstreeR} package is not available, an error is generated.
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
#' Proc. ACM SIGKDD'10 (2010) 603-611, \url{https://mlpack.org/papers/emst.pdf}
#'
#' @export
emst_mlpack <- function(X, verbose=FALSE)
{
    X <- as.matrix(X)
    if (requireNamespace("emstreeR", quietly=TRUE)) {

        if (!verbose)
            capture.output({mst <- emstreeR::mlpack_mst(X)})
        else
            mst <- emstreeR::mlpack_mst(X)

        structure(t(mst),
            class="mst",
            dist.method="euclidean"
        )

    } else {
        stop("emstreeR` package is not installed.")
    }
}
