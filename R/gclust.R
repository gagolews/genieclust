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
#' The clustering can also be computed with respect to the
#' \eqn{M}-mutual reachability distance (based, e.g., on the Euclidean metric),
#' which is used in the definition of the HDBSCAN* algorithm
#' (see \code{\link{mst}()} for the definition).
#' For the smoothing factor \eqn{M>0}, outliers are pulled away from
#' their neighbours.  This way, the Genie algorithm gives an alternative
#' to the HDBSCAN* algorithm (Campello et al., 2013) that is able to detect
#' a predefined number of clusters and indicate outliers (Gagolewski, 2025)
#' without depending on DBSCAN*'s \code{eps} or HDBSCAN*'s \code{min_cluster_size}
#' parameters.
#'
#'
#' @details
#' As in the case of all the distance-based methods,
#' the standardisation of the input features is definitely worth giving a try.
#' Oftentimes, applying some more sophisticated feature engineering techniques
#' (e.g., dimensionality reduction) might lead to more meaningful results.
#'
#' If \code{d} is a numeric matrix or an object of class \code{dist},
#' \code{\link{mst}()} will be called to compute an MST, which generally
#' takes at most \eqn{O(n^2)} time. However, by default, a faster algorithm
#' based on K-d trees is selected automatically for low-dimensional Euclidean
#' spaces; see \code{\link[quitefastmst]{mst_euclid}} from
#' the \pkg{quitefastmst} package.
#'
#' Once a minimum spanning tree is determined, the Genie algorithm runs in
#' \eqn{O(n \sqrt{n})} time.  If you want to test different
#' \code{gini_threshold}s or \code{k}s, it is best to compute
#' the MST explicitly beforehand.
#'
#' Due to Genie's original definition, the resulting partition tree (dendrogram)
#' might violate the ultrametricity property (merges might occur at levels that
#' are not increasing w.r.t. a between-cluster distance).
#' \code{gclust()} automatically corrects departures from
#' ultrametricity by applying \code{height = rev(cummin(rev(height)))}.
#'
#' TODO If \eqn{M > 0}, all MST leaves are, by default, left out from the clustering
#' process; see \code{skip_leaves}.  Afterwards, some of them (midliers)
#' are merged with the nearest clusters at the postprocessing stage,
#' and other ones are marked as outliers; see [4]_ for discussion.
#'
#'
#' @param d a numeric matrix (or an object coercible to one,
#'     e.g., a data frame with numeric-like columns) or an
#'     object of class \code{dist} (see \code{\link[stats]{dist}}),
#'     or an object of class \code{mst} (\code{\link{mst}})
#' @param gini_threshold threshold for the Genie correction, i.e.,
#'     the Gini index of the cluster size distribution;
#'     threshold of 1.0 leads to the single linkage algorithm;
#'     low thresholds highly penalise the formation of small clusters
#' @param distance metric used to compute the linkage, one of:
#'     \code{"euclidean"} (synonym: \code{"l2"}),
#'     \code{"manhattan"} (a.k.a. \code{"l1"} and \code{"cityblock"}),
#'     \code{"cosine"}
#' @param verbose logical; whether to print diagnostic messages
#'     and progress information
#' @param k the desired number of clusters to detect, \eqn{k=1} with
#'      \eqn{M>0} acts as an outlier detector
#' @param M smoothing factor; \eqn{M \leq 1} gives the selected \code{distance};
#'     otherwise, the mutual reachability distance is used
#' @param ... further arguments passed to \code{\link{mst}()}
#'
'
#' @return
#' \code{gclust()} computes the entire clustering hierarchy; it
#' returns a list of class \code{hclust}; see \code{\link[stats]{hclust}}.
#' Use \code{\link[stats]{cutree}} to obtain an arbitrary \code{k}-partition.
#'
#' \code{genie()} returns a \code{k}-partition, i.e., a vector whose
#' \eqn{i}-th element denotes the \eqn{i}-th input point's cluster label
#' between 1 and \code{k}.
#' TODO If \code{skip_leaves} is \code{TRUE} and \code{postprocess}
#' is not \code{"all"}, missing values (\code{NA}) denote outliers.
#'
#'
#' @seealso
#' \code{\link{mst}()} for the minimum spanning tree routines
#'
#' \code{\link{normalized_clustering_accuracy}()} (amongst others) for external
#' cluster validity measures
#'
#'
#' @references
#' Gagolewski M., Bartoszuk M., Cena A.,
#' Genie: A new, fast, and outlier-resistant hierarchical clustering algorithm,
#' \emph{Information Sciences} 363, 2016, 8-23,
#' \doi{10.1016/j.ins.2016.05.003}
#'
#' Campello R.J.G.B., Moulavi D., Sander J.,
#' Density-based clustering based on hierarchical density estimates,
#' \emph{Lecture Notes in Computer Science} 7819, 2013, 160-172,
#' \doi{10.1007/978-3-642-37456-2_14}
#'
#' Gagolewski M., Cena A., Bartoszuk M., Brzozowski L.,
#' Clustering with minimum spanning trees: How good can it be?,
#' \emph{Journal of Classification} 42, 2025, 90-112,
#' \doi{10.1007/s00357-024-09483-1}
#'
#' Gagolewski M., TODO, 2025
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
#' y_pred2 <- genie(X, 3, M=5)  # the 5-mutual reachability distance
#' plot(X[,1], X[,2], col=y_pred2, pch=y_test, asp=1, las=1)
#' is_outlier <- is.na(y_pred2)
#' points(X[is_outlier, ], col="gray", pch=10)
#' normalized_clustering_accuracy(y_test[!is_outlier], y_pred2[!is_outlier])
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
    distance=c("euclidean", "l2", "manhattan", "cityblock", "l1", "cosine"),
    verbose=FALSE,
    ...
) {
    distance <- match.arg(distance)
    tree <- mst.default(d, distance=distance, verbose=verbose, ...)
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
    verbose=FALSE,
    ...
) {
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
    M=0L,
    #preprocess=c("none"),  # TODO
    #postprocess=c("midliers", "none", "all"),  # TODO
    verbose=FALSE,
    ...
) {
    distance <- match.arg(distance)
    #preprocess  <- match.arg(preprocess)
    #postprocess <- match.arg(postprocess)
    tree <- mst.default(d, M=M, distance=distance, verbose=verbose, ...)
    genie.mst(
        tree,
        k=k,
        gini_threshold=gini_threshold,
        #preprocess=preprocess,
        #postprocess=postprocess,
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
    #preprocess=c("none"),  # TODO
    #postprocess=c("midliers", "none", "all"),  # TODO
    verbose=FALSE,
    ...
) {
    #preprocess  <- match.arg(preprocess)
    #postprocess <- match.arg(postprocess)
    genie.mst(
        mst.dist(d, M=M, verbose=verbose, ...),
        k=k,
        gini_threshold=gini_threshold,
        #preprocess=preprocess,
        #postprocess=postprocess,
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
    #preprocess=c("none"),  # TODO
    #postprocess=c("midliers", "none", "all"),  # TODO
    verbose=FALSE,
    ...
) {
    gini_threshold <- as.double(gini_threshold)[1]
    stopifnot(gini_threshold >= 0.0, gini_threshold <= 1.0)

    #preprocess  <- match.arg(preprocess)
    #postprocess <- match.arg(postprocess)
    verbose <- !identical(verbose, FALSE)

    structure(
        .genie(
            d,
            k=k,
            gini_threshold=gini_threshold,
            #preprocess=preprocess,
            #postprocess=postprocess,
            verbose=verbose
        ),
        names=attr(d, "Labels")
    )
}


registerS3method("gclust", "default", "gclust.default")
registerS3method("gclust", "dist",    "gclust.dist")
registerS3method("gclust", "mst",     "gclust.mst")

registerS3method("genie", "default", "genie.default")
registerS3method("genie", "dist",    "genie.dist")
registerS3method("genie", "mst",     "genie.mst")
