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



#' @title The Genie Hierarchical Clustering Algorithm (with Extras)
#'
#' @description
#' See \code{\link{genie}()} for more details.
#'
#' @useDynLib genieclust, .registration=TRUE
#' @importFrom Rcpp evalCpp
#' @importFrom stats hclust
#' @importFrom stats cutree
#' @importFrom quitefastmst knn_euclid
#' @importFrom quitefastmst mst_euclid
#' @importFrom stats dist
#' @importFrom utils capture.output
#' @keywords internal
"_PACKAGE"
