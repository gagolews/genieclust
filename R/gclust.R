#' @title The Genie++ Hierarchical Clustering Algorithm
#'
#' @description
#' TODO
#'
#' @param d either a numeric matrix (or an object coercible to one,
#'     e.g., a data frame with numeric-like columns) or an
#'     object of class \code{dist}, see \code{\link[stats]{dist}}.
#' @param gini_threshold threshold for the Genie correction, i.e.,
#'     the Gini index of the cluster size distribution;
#'     Threshold of 1.0 disables the correction.
#'     Low thresholds highly penalise the formation of small clusters.
#' @param M smoothing factor; M=1 gives the original Genie algorithm.
#' @param postprocess one of "boundary" (default), "none", "all";
#'     in effect only if M>1. By default, only "boundary" points are merged
#'     with their nearest "core" points. To force a classical
#'     n_clusters-partition of a data set (with no notion of noise),
#'     choose "all".
#' @param distance metric used to compute the linkage, one of: "euclidean"
#'     (synonym: "l2"), "manhattan" (a.k.a. "l1" and "cityblock"), "cosine"
#' @param ... further arguments passed to or from other methods.
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
#' #h <- gclust(X)
#' # y_pred <- cutree(h, 3)
#' y_test <- iris[,5]
#' #plot(iris[,2], iris[,3], col=y_pred,
#' #   pch=as.integer(iris[,5]), asp=1, las=1)
#' #adjusted_rand_score(y_test, y_pred)
#'
#' @rdname gclust
#' @export
gclust <- function(d, ...)
{
    UseMethod("gclust")
}


.correct_height <- function(height)
{
    # correction for the departure from ultrametricity
    if (any(height < 0)) {
        height <- rev(cummin(rev(height)))
    }
    height
}


#' @export
#' @rdname gclust
#' @method gclust default
gclust.default <- function(d,
    gini_threshold=0.3,
    M=1L,
    postprocess=c("boundary", "none", "all"),
    distance=c("euclidean", "l2", "manhattan", "cityblock", "l1", "cosine"),
    ...)
{
    postprocess <- match.arg(postprocess)
    distance <- match.arg(distance)

    d <- as.matrix(d)

    result <- .gclust.default(d, gini_threshold, M, postprocess, distance)

    result[["height"]] <- .correct_height(result[["height"]])
    result[["labels"]] <- dimnames(d)[[1L]]
    result[["method"]] <- sprintf("Genie++(%g)", gini_threshold)
    result[["call"]]   <- match.call()
    result[["dist.method"]] <- if (M == 1L) distance else
        sprintf("mutual reachability distance (%s, M=%d)", distance, M)
    class(result) <- "hclust"

    result
}


#' @export
#' @rdname gclust
#' @method gclust dist
gclust.dist <- function(d,
    gini_threshold=0.3,
    M=1L,
    postprocess=c("boundary", "none", "all"),
    ...)
{
    postprocess <- match.arg(postprocess)

    result <- .gclust.dist(d, gini_threshold, M, postprocess)

    result[["height"]] <- .correct_height(result[["height"]])
    result[["labels"]] <- attr(d, "Labels")
    result[["method"]] <- sprintf("Genie++(%g)", gini_threshold)
    result[["call"]]   <- match.call()
    result[["dist.method"]] <- if (M == 1L) attr(d, "method") else
        sprintf("mutual reachability distance (%s, M=%d)", attr(d, "method"), M)
    class(result) <- "hclust"

    result
}



registerS3method("gclust", "default", "gclust.default")
registerS3method("gclust", "dist", "gclust.dist")
