# This file is part of the genieclust package for R.

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2020-2026, Marek Gagolewski <https://www.gagolewski.com>      #
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
#' clustering algorithm by Gagolewski, Bartoszuk, and Cena (2016).
#' The Genie algorithm is based on the minimum spanning tree (MST) of the
#' pairwise distance graph of a given point set.
#' Just like the Single Linkage method, it consumes the edges
#' of the MST in an increasing order of weights. However, it prevents
#' the formation of clusters of highly imbalanced sizes; once the Gini index
#' (see \code{\link{gini_index}()}) of the cluster size distribution
#' raises above \code{gini_threshold}, merging a point group
#' of the smallest size is enforced.
#'
#' A clustering can also be computed with respect to the
#' \eqn{M}-mutual reachability distance (based, e.g., on the Euclidean metric),
#' which is used in the definition of the HDBSCAN* algorithm
#' (see \code{\link[deadwood]{mst}()} for the definition).
#' For the smoothing factor \eqn{M>0}, outliers are pulled away from
#' their neighbours.  This way, the Genie algorithm gives an alternative
#' to the HDBSCAN* algorithm (Campello et al., 2013) that is able to detect
#' a predefined number of clusters and indicate outliers (Gagolewski, 2026)
#' without depending on DBSCAN*'s \code{eps} or HDBSCAN*'s
#' \code{min_cluster_size} parameters.  Also make sure to check out
#' the Lumbermark method that is also based on MSTs.
#'
#'
#' @details
#' As with all distance-based methods (this includes k-means and DBSCAN as well),
#' applying data preprocessing and feature engineering techniques
#' (e.g., feature scaling, feature selection, dimensionality reduction)
#' might lead to more meaningful results.
#'
#' If \code{d} is a numeric matrix or an object of class \code{dist},
#' \code{\link[deadwood]{mst}()} will be called to compute an MST, which generally
#' takes at most \eqn{O(n^2)} time. However, by default, a faster algorithm
#' based on K-d trees is selected automatically for low-dimensional Euclidean
#' spaces; see \code{\link[quitefastmst]{mst_euclid}} from
#' the \pkg{quitefastmst} package.
#'
#' Once a minimum spanning tree is determined, the Genie algorithm runs in
#' \eqn{O(n \sqrt{n})} time.  If you want to test different
#' \code{gini_threshold}s or \eqn{k}s, it is best to compute
#' the MST explicitly beforehand.
#'
#' Due to Genie's original definition, the resulting partition tree (dendrogram)
#' might violate the ultrametricity property (merges might occur at levels that
#' are not increasing w.r.t. a between-cluster distance).
#' \code{gclust()} automatically corrects departures from
#' ultrametricity by applying \code{height = rev(cummin(rev(height)))}.
#'
#'
#' @param d a numeric matrix (or an object coercible to one,
#'     e.g., a data frame with numeric-like columns) or an
#'     object of class \code{dist} (see \code{\link[stats]{dist}}),
#'     or an object of class \code{mst} (see \code{\link[deadwood]{mst}})
#'
#' @param gini_threshold threshold for the Genie correction, i.e.,
#'     the Gini index of the cluster size distribution;
#'     threshold of 1.0 leads to the single linkage algorithm;
#'     low thresholds highly penalise the formation of small clusters
#'
#' @param distance metric used to compute the linkage, one of:
#'     \code{"euclidean"} (synonym: \code{"l2"}),
#'     \code{"manhattan"} (a.k.a. \code{"l1"} and \code{"cityblock"}),
#'     \code{"cosine"}
#'
#' @param verbose logical; whether to print diagnostic messages
#'     and progress information
#'
#' @param k the desired number of clusters to detect
#'
#' @param M smoothing factor; \eqn{M \leq 1} gives the selected \code{distance};
#'     otherwise, the mutual reachability distance is used
#'
#' @param ... further arguments passed to \code{\link[deadwood]{mst}()}
#'
#' @return
#' \code{gclust()} computes the entire clustering hierarchy; it
#' returns a list of class \code{hclust}; see \code{\link[stats]{hclust}}.
#' Use \code{\link[stats]{cutree}} to obtain an arbitrary \eqn{k}-partition.
#'
#' \code{genie()} returns an object of class \code{mstclust}, which defines
#' a \eqn{k}-partition, i.e., a vector whose \eqn{i}-th element denotes
#' the \eqn{i}-th input point's cluster label
#' between 1 and \eqn{k}.
#'
#' In both cases, the \code{mst} attribute gives the computed minimum
#' spanning tree which can be reused in further calls to the functions
#' from \pkg{genieclust}, \pkg{lumbermark}, and \pkg{deadwood}.
#' For \code{genie()}, the \code{cut_edges} attribute gives the \eqn{k-1}
#' indexes of the MST edges whose omission leads to the requested
#' \eqn{k}-partition (connected components of the resulting spanning forest).
#' In \code{gclust()}, these are exactly the last \eqn{k-1} indexes in the
#' \code{links} attribute (but sorted).
#'
#'
#' @seealso
#' \code{\link[deadwood]{mst}()} for the minimum spanning tree routines
#'
#' \code{\link{normalized_clustering_accuracy}()} (amongst others) for external
#' cluster validity measures
#'
#'
#' @references
#' M. Gagolewski, M. Bartoszuk, A. Cena,
#' Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
#' \emph{Information Sciences} 363, 2016, 8-23,
#' \doi{10.1016/j.ins.2016.05.003}
#'
#' R.J.G.B. Campello, D. Moulavi, J. Sander,
#' Density-based clustering based on hierarchical density estimates,
#' \emph{Lecture Notes in Computer Science} 7819, 2013, 160-172,
#' \doi{10.1007/978-3-642-37456-2_14}
#'
#' M. Gagolewski, A. Cena, M. Bartoszuk, Ł. Brzozowski,
#' Clustering with minimum spanning trees: How good can it be?,
#' \emph{Journal of Classification} 42, 2025, 90-112,
#' \doi{10.1007/s00357-024-09483-1}
#'
#' M. Gagolewski, genieclust: Fast and robust hierarchical clustering,
#' \emph{SoftwareX} 15, 2021, 100722, \doi{10.1016/j.softx.2021.100722}
#'
#' M. Gagolewski, deadwood, in preparation, 2026
#'
#' M. Gagolewski, quitefastmst, in preparation, 2026
#'
#'
#' @examples
#' library("datasets")
#' data("iris")
#' X <- jitter(as.matrix(iris[3:4]))
#' h <- gclust(X)
#' y_pred <- cutree(h, 3)
#' y_test <- as.integer(iris[,5])
#' plot(X, col=y_pred, pch=y_test, asp=1, las=1)
#' adjusted_rand_score(y_test, y_pred)
#' normalized_clustering_accuracy(y_test, y_pred)
#'
#' # detect 3 clusters and find outliers with Deadwood
#' library("deadwood")
#' y_pred2 <- genie(X, k=3, M=5)  # the 5-mutual reachability distance
#' plot(X, col=y_pred2, asp=1, las=1)
#' is_outlier <- deadwood(y_pred2)
#' points(X[!is_outlier, ], col=y_pred2[!is_outlier], pch=16)
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
gclust.default <- function(
    d,
    gini_threshold=0.3,
    M=0L,
    distance=c("euclidean", "l2", "manhattan", "cityblock", "l1", "cosine"),
    verbose=FALSE,
    ...
) {
    distance <- match.arg(distance)
    tree <- mst(d, M=M, distance=distance, verbose=verbose, ...)
    gclust.mst(
        tree,
        gini_threshold=gini_threshold,
        verbose=verbose
    )
}


#' @export
#' @rdname gclust
#' @method gclust dist
gclust.dist <- function(
    d,
    gini_threshold=0.3,
    M=0L,
    verbose=FALSE,
    ...
) {
    gclust.mst(
        mst(d, M=M, verbose=verbose, ...),
        gini_threshold=gini_threshold, verbose=verbose
    )
}


#' @export
#' @rdname gclust
#' @method gclust mst
gclust.mst <- function(
    d,
    gini_threshold=0.3,
    verbose=FALSE,
    ...
) {
    gini_threshold <- as.double(gini_threshold)[1]
    stopifnot(gini_threshold >= 0.0, gini_threshold <= 1.0)
    verbose <- !identical(verbose, FALSE)

    result <- .gclust(d, gini_threshold, verbose)

    result[["height"]] <- .correct_height(result[["height"]])
    result[["labels"]] <- attr(d, "Labels")  # yes, >L<abels
    result[["method"]] <- sprintf("Genie(%g)", gini_threshold)
    result[["call"]]   <- match.call()
    result[["dist.method"]] <- attr(d, "method")

    attr(result, "mst") <- d
    stopifnot(length(attr(result, "links")) == NROW(result[["merge"]]))

    if (is.na(tail(attr(result, "links"), 1)))
        warning("incomplete cluster hierarchy")

    class(result) <- c("msthclust", "hclust")

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
    M=0L,
    distance=c("euclidean", "l2", "manhattan", "cityblock", "l1", "cosine"),
    verbose=FALSE,
    ...
) {
    distance <- match.arg(distance)
    tree <- mst(d, M=M, distance=distance, verbose=verbose, ...)
    genie.mst(
        tree,
        k=k,
        gini_threshold=gini_threshold,
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
    M=0L,
    verbose=FALSE,
    ...
) {
    genie.mst(
        mst(d, M=M, verbose=verbose, ...),
        k=k,
        gini_threshold=gini_threshold,
        verbose=verbose
    )
}


#' @export
#' @rdname gclust
#' @method genie mst
genie.mst <- function(
    d,
    k,
    gini_threshold=0.3,
    verbose=FALSE,
    ...
) {
    gini_threshold <- as.double(gini_threshold)[1]
    stopifnot(gini_threshold >= 0.0, gini_threshold <= 1.0)

    verbose <- !identical(verbose, FALSE)

    clusters <- .genie(
        d,
        k=k,
        gini_threshold=gini_threshold,
        verbose=verbose
    )

    stopifnot(length(attr(clusters, "cut_edges")) == k-1)

    structure(
        clusters,
        names=attr(d, "Labels"),
        mst=d,
        #cut_edges=cut_edges,  already there
        class="mstclust"
    )
}


registerS3method("gclust", "default", "gclust.default")
registerS3method("gclust", "dist",    "gclust.dist")
registerS3method("gclust", "mst",     "gclust.mst")

registerS3method("genie", "default", "genie.default")
registerS3method("genie", "dist",    "genie.dist")
registerS3method("genie", "mst",     "genie.mst")
