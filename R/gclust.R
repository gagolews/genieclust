#' @title The Genie++ Hierarchical Clustering Algorithm
#'
#' @description
#' TODO
#'
#' @param d TODO
#' @param gini_threshold TODO
#' @param M TODO
#' @param postprocess TODO
#' @param distance TODO
#' @param ... TODO
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
gclust.default <- function(d, gini_threshold=0.3, M=1L,
    postprocess="boundary", distance="euclidean", ...)
{
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
gclust.dist <- function(d, gini_threshold=0.3, M=1L, postprocess="boundary", ...)
{
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
