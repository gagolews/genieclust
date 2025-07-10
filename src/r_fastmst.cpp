/*  Functions to compute k-nearest neighbours and minimum spanning trees
 *  with respect to the Euclidean metric and the thereon-based mutual
 *  reachability distances. The module provides access to a quite fast
 *  implementation of K-d trees.
 *
 *  For best speed, consider building the package from sources
 *  using, e.g., `-O3 -march=native` compiler flags.
 *
 *  Copyleft (C) 2025-2025, Marek Gagolewski <https://www.gagolewski.com>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Affero General Public License
 *  Version 3, 19 November 2007, published by the Free Software Foundation.
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU Affero General Public License Version 3 for more details.
 *  You should have received a copy of the License along with this program.
 *  If this is not the case, refer to <https://www.gnu.org/licenses/>.
 */


#include "c_common.h"
#include "c_matrix.h"
#include "c_fastmst.h"
#include <cmath>

using namespace Rcpp;


//' @title
//' Get or Set the Number of Threads
//'
//' @description
//' These functions get or set the maximal number of OpenMP threads that
//' can be used by \code{\link{knn_euclid}} and \code{\link{mst_euclid}},
//' amongst others.
//'
//' @param n_threads maximum number of threads to use;
//'
//' @return
//' \code{omp_get_max_threads} returns the maximal number
//' of threads that will be used during the next call to a parallelised
//' function, not the maximal number of threads possibly available.
//' It there is no built-in support for OpenMP, 1 is always returned.
//'
//' For \code{omp_set_num_threads}, the previous value of \code{max_threads}
//' is output.
//'
//'
//' @rdname omp
//' @export
// [[Rcpp::export("omp_set_num_threads")]]
int Romp_set_num_threads(int n_threads)
{
    return Comp_set_num_threads(n_threads);
}



//' @rdname omp
//' @export
// [[Rcpp::export("omp_get_max_threads")]]
int Romp_get_max_threads()
{
    return Comp_get_max_threads();
}



//' @title Quite Fast Euclidean Nearest Neighbours
//'
//' @description
//' If \code{Y} is \code{NULL}, then the function determines the first \code{k}
//' amongst the nearest neighbours of each point in \code{X} with respect
//' to the Euclidean distance. It is assumed that each query point is
//' not its own neighbour.
//'
//' Otherwise, for each point in \code{Y}, this function determines the \code{k}
//' nearest points thereto from \code{X}.
//'
//' @details
//' The implemented algorithms, see the \code{algorithm} parameter, assume
//' that \code{k} is rather small; say, \code{k <= 20}.
//'
//' Our implementation of K-d trees (Bentley, 1975) has been quite optimised;
//' amongst others, it has good locality of reference, features the sliding
//' midpoint (midrange) rule suggested by Maneewongvatana and Mound (1999),
//' and a node pruning strategy inspired by the discussion
//' by Sample et al. (2001).  Still, it is well-known that K-d trees
//' perform well only in spaces of low intrinsic dimensionality.  Thus,
//' due to the so-called curse of dimensionality, for high \code{d},
//' the brute-force algorithm is recommended.
//'
//' The number of threads used is controlled via the \code{OMP_NUM_THREADS}
//' environment variable or via the \code{\link{omp_set_num_threads}} function
//' at runtime. For best speed, consider building the package
//' from sources using, e.g., \code{-O3 -march=native} compiler flags.
//'
//'
//' @references
//' J.L. Bentley, Multidimensional binary search trees used for associative
//' searching, \emph{Communications of the ACM} 18(9), 509–517, 1975,
//' \doi{10.1145/361002.361007}.
//'
//' S. Maneewongvatana, D.M. Mount, It's okay to be skinny, if your friends
//' are fat, \emph{4th CGC Workshop on Computational Geometry}, 1999.
//'
//' N. Sample, M. Haines, M. Arnold, T. Purcell, Optimizing search
//' strategies in K-d Trees, \emph{5th WSES/IEEE Conf. on Circuits, Systems,
//' Communications & Computers} (CSCC'01), 2001.
//'
//'
//' @param X the "database"; a matrix of shape (n,d)
//' @param k number of nearest neighbours (should be rather small, say, <= 20)
//' @param Y the "query points"; \code{NULL} or a matrix of shape (m,d);
//'     note that setting \code{Y=X}, contrary to \code{NULL},
//'     will include the query points themselves amongst their own neighbours
//' @param algorithm \code{"auto"}, \code{"kd_tree"} or \code{"brute"};
//'     K-d trees can only be used for d between 2 and 20 only;
//'     \code{"auto"} selects \code{"kd_tree"} in low-dimensional spaces
//' @param max_leaf_size maximal number of points in the K-d tree leaves;
//'        smaller leaves use more memory, yet are not necessarily faster;
//'        use \code{0} to select the default value, currently set to 32
//' @param squared whether to return the squared Euclidean distance
//' @param verbose whether to print diagnostic messages
//'
//'
//' @return
//' A list with two elements, \code{nn.index} and \code{nn.dist}.
//'
//' \code{nn.dist} has shape (n,k) or (m,k);
//' \code{nn.dist[i,]} is sorted nondecreasingly for all \code{i}.
//' \code{nn.dist[i,j]} gives the weight of the edge \code{{i, ind[i,j]}},
//' i.e., the distance between the \code{i}-th point and its \code{j}-th NN.
//'
//' \code{nn.index} is of the same shape.
//' \code{nn.index[i,j]} is the index (between \code{1} and \code{n})
//' of the \code{j}-th nearest neighbour of \code{i}.
//'
//'
//' @examples
//' library("datasets")
//' data("iris")
//' X <- jitter(as.matrix(iris[1:2]))  # some data
//' neighbours <- knn_euclid(X, 1)  # 1-NNs of each point
//' plot(X, asp=1, las=1)
//' segments(X[,1], X[,2], X[neighbours$nn.index,1], X[neighbours$nn.index,2])
//'
//' knn_euclid(X, 5, matrix(c(6, 4), nrow=1))  # five closest points to (6, 4)
//'
//'
//' @seealso \code{\link{mst_euclid}}
//'
//' @rdname fastknn
//' @export
// [[Rcpp::export("knn_euclid")]]
List knn_euclid(
    SEXP X,
    int k=1,
    SEXP Y=R_NilValue,
    Rcpp::String algorithm="auto",
    int max_leaf_size=0,
    bool squared=false,
    bool verbose=false
) {
    using FLOAT = double;  // float is not faster..

    Rcpp::NumericMatrix _X;
    if (!Rf_isMatrix(X))  _X = Rcpp::internal::convert_using_rfunction(X, "as.matrix");
    else _X = X;

    Py_ssize_t n = (Py_ssize_t)_X.nrow();
    Py_ssize_t d = (Py_ssize_t)_X.ncol();
    Py_ssize_t m;
    bool use_kdtree;

    if (n < 1 || d <= 1) stop("X is ill-shaped");
    if (k < 1) stop("`k` must be >= 1");

    if (algorithm == "auto") {
        if (2 <= d && d <= 20)
            algorithm = "kd_tree";
        else
            algorithm = "brute";
    }

    if (algorithm == "kd_tree") {
        if (d < 2 || d > 20) stop("kd_tree can only be used for 2 <= d <= 20");
        if (max_leaf_size == 0) max_leaf_size = 32;  // the current default

        if (max_leaf_size <= 0) stop("max_leaf_size must be positive");
        use_kdtree = true;
    }
    else if (algorithm == "brute")
        use_kdtree = false;
    else
        stop("invalid 'algorithm'");


    std::vector<FLOAT> XC(n*d);
    Py_ssize_t j = 0;
    for (Py_ssize_t i=0; i<n; ++i)
        for (Py_ssize_t u=0; u<d; ++u)
            XC[j++] = (FLOAT)_X(i, u);  // row-major


    std::vector<FLOAT>      nn_dist;
    std::vector<Py_ssize_t> nn_ind;
    if (Rf_isNull(Y)) {
        if (k >= n) stop("too many neighbours requested");
        m = n;

        nn_dist.resize(n*k);
        nn_ind.resize(n*k);

        if (use_kdtree)
            Cknn1_euclid_kdtree(
                XC.data(), n, d, k, nn_dist.data(),
                nn_ind.data(), max_leaf_size, squared, verbose
            );
        else
            Cknn1_euclid_brute(
                XC.data(), n, d, k, nn_dist.data(),
                nn_ind.data(), squared, verbose
            );
    }
    else {
        if (k >  n) stop("too many neighbours requested");

        Rcpp::NumericMatrix _Y;
        if (!Rf_isMatrix(Y)) _Y = Rcpp::internal::convert_using_rfunction(Y, "as.matrix");
        else _Y = Y;

        m = (Py_ssize_t)_Y.nrow();
        if (_Y.ncol() != d) stop("Y's dimensionality does not match that of X");

        nn_dist.resize(m*k);
        nn_ind.resize(m*k);

        std::vector<FLOAT> YC(m*d);
        Py_ssize_t j = 0;
        for (Py_ssize_t i=0; i<m; ++i)
            for (Py_ssize_t u=0; u<d; ++u)
                YC[j++] = (FLOAT)_Y(i, u);  // row-major

        if (use_kdtree)
            Cknn2_euclid_kdtree(
                XC.data(), n, YC.data(), m, d, k, nn_dist.data(), nn_ind.data(),
                max_leaf_size, squared, verbose
            );
        else
            Cknn2_euclid_brute(
                XC.data(), n, YC.data(), m, d, k, nn_dist.data(), nn_ind.data(),
                squared, verbose
            );
    }

    Rcpp::IntegerMatrix out_ind(m, k);
    Rcpp::NumericMatrix out_dist(m, k);
    Py_ssize_t u = 0;
    for (Py_ssize_t i=0; i<m; ++i) {
        for (Py_ssize_t j=0; j<k; ++j) {
            out_ind(i, j)  = nn_ind[u]+1.0;  // R-based indexing
            out_dist(i, j) = nn_dist[u];
            u++;
        }
    }

    return Rcpp::List::create(
        Rcpp::Named("nn.index")=out_ind,
        Rcpp::Named("nn.dist")=out_dist
    );
}




//' @title Quite Fast Euclidean Minimum Spanning Trees (Also WRT Mutual Reachability Distances)
//'
//' @description
//' The function determines the/a(*) minimum spanning tree (MST) of a set
//' of \eqn{n} points, i.e., an acyclic undirected graph whose vertices represent
//' the points, and \eqn{n-1} edges with the minimal sum of weights, given by
//' the pairwise distances.  MSTs have many uses in, amongst others,
//' topological data analysis (clustering, dimensionality reduction, etc.).
//'
//' For \eqn{M\leq 2}, we get a spanning tree that minimises the sum of Euclidean
//' distances between the points. If \eqn{M=2}, the function additionally returns
//' the distance to each point's nearest neighbour.
//'
//' If \eqn{M>2}, the spanning tree is the smallest wrt the degree-M
//' mutual reachability distance (Campello et al., 2013) given by
//' \eqn{d_M(i, j)=\max\{ c_M(i), c_M(j), d(i, j)\}}, where \eqn{d(i,j)}
//' is the Euclidean distance between the \eqn{i}-th and the \eqn{j}-th point,
//' and \eqn{c_M(i)} is the \eqn{i}-th \eqn{M}-core distance defined as the distance
//' between the \eqn{i}-th point and its \eqn{(M-1)}-th nearest neighbour
//' (not including the query points themselves).
//' In clustering and density estimation, M plays the role of a smoothing
//' factor; see (Campello et al. 2015) and the references therein for discussion.
//'
//'
//' @details
//' (*) We note that if there are many pairs of equidistant points,
//' there can be many minimum spanning trees. In particular, it is likely
//' that there are point pairs with the same mutual reachability distances.
//' To make the definition less ambiguous (albeit with no guarantees),
//' internally, the brute-force algorithm relies on the adjusted distance
//' \eqn{d_M(i, j)=\max\{c_M(i), c_M(j), d(i, j)\}+\varepsilon d(i, j)},
//' where \eqn{\varepsilon} is close to 0; see \code{dcore_dist_adj}.
//' For the K-d tree-based methods, on the other hand, negative
//' \code{dcore_dist_adj} indicates the preference towards connecting to
//' farther points wrt the original metric in the case of the same
//' core distance instead of closer ones if the adjustment is positive.
//' When preferring farther points, the resulting spanning tree tends to
//' have more leaves.
//' Furthermore, setting \code{dcore_dist_adj} to minus infinity,
//' prefers NNs with smaller core distances. This results in even more leaves.
//'
//' The implemented algorithms, see the \code{algorithm} parameter, assume
//' that \code{M} is rather small; say, \eqn{M \leq 20}.
//'
//' Our implementation of K-d trees (Bentley, 1975) has been quite optimised;
//' amongst others, it has good locality of reference (at the cost of making
//' a copy of the input dataset), features the sliding
//' midpoint (midrange) rule suggested by Maneewongvatana and Mound (1999),
//' and a node pruning strategy inspired by the discussion
//' by Sample et al. (2001).
//'
//' The "single-tree" version of the Borůvka algorithm is naively
//' parallelisable: in every iteration, it seeks each point's nearest "alien",
//' i.e., the nearest point thereto from another cluster.
//' The "dual-tree" Borůvka version of the algorithm is, in principle, based
//' on (March et al., 2010). As far as our implementation is concerned,
//' the dual-tree approach is often only faster in 2- and 3-dimensional spaces,
//' for \eqn{M\leq 2}, and in a single-threaded setting.  For another (approximate)
//' adaptation of the dual-tree algorithm to the mutual reachability distance;
//' see (McInnes and Healy, 2017).
//'
//' Nevertheless, it is well-known that K-d trees perform well only in spaces
//' of low intrinsic dimensionality (a.k.a. the "curse").  For high \code{d},
//' the "brute-force" algorithm is recommended.  Here, we provided a
//' parallelised (see Olson, 1995) version of the Jarník (1930) (a.k.a.
//' Prim (1957) or Dijkstra) algorithm, where the distances are computed
//' on the fly (only once for \code{M<=2}).
//'
//' The number of threads used is controlled via the \code{OMP_NUM_THREADS}
//' environment variable or via the \code{\link{omp_set_num_threads}} function
//' at runtime. For best speed, consider building the package
//' from sources using, e.g., \code{-O3 -march=native} compiler flags.
//'
//'
//' @references
//' V. Jarník, O jistém problému minimálním,
//' \emph{Práce Moravské Přírodovědecké Společnosti} 6, 1930, 57–63.
//'
//' C.F. Olson, Parallel algorithms for hierarchical clustering,
//' Parallel Computing 21(8), 1995, 1313–1325.
//'
//' R. Prim, Shortest connection networks and some generalizations,
//' \emph{The Bell System Technical Journal} 36(6), 1957, 1389–1401.
//'
//' O. Borůvka, O jistém problému minimálním, \emph{Práce Moravské
//' Přírodovědecké Společnosti} 3, 1926, 37–58.
//'
//' W.B. March, R. Parikshit, A.G. Gray, Fast Euclidean minimum spanning
//' tree: Algorithm, analysis, and applications, \emph{Proc. 16th ACM SIGKDD
//' Intl. Conf. Knowledge Discovery and Data Mining (KDD '10)}, 2010, 603–612.
//'
//' J.L. Bentley, Multidimensional binary search trees used for associative
//' searching, \emph{Communications of the ACM} 18(9), 509–517, 1975,
//' \doi{10.1145/361002.361007}.
//'
//' S. Maneewongvatana, D.M. Mount, It's okay to be skinny, if your friends
//' are fat, \emph{4th CGC Workshop on Computational Geometry}, 1999.
//'
//' N. Sample, M. Haines, M. Arnold, T. Purcell, Optimizing search
//' strategies in K-d Trees, \emph{5th WSES/IEEE Conf. on Circuits, Systems,
//' Communications & Computers} (CSCC'01), 2001.
//'
//' R.J.G.B. Campello, D. Moulavi, J. Sander, Density-based clustering based
//' on hierarchical density estimates, \emph{Lecture Notes in Computer Science}
//' 7819, 2013, 160–172. \doi{10.1007/978-3-642-37456-2_14}.
//'
//' R.J.G.B. Campello, D. Moulavi, A. Zimek. J. Sander, Hierarchical
//' density estimates for data clustering, visualization, and outlier detection,
//' \emph{ACM Transactions on Knowledge Discovery from Data (TKDD)} 10(1),
//' 2015, 1–51, \doi{10.1145/2733381}.
//'
//' L. McInnes, J. Healy, Accelerated hierarchical density-based
//' clustering, \emph{IEEE Intl. Conf. Data Mining Workshops (ICMDW)}, 2017,
//' 33–42, \doi{10.1109/ICDMW.2017.12}.
//'
//'
//'
//' @param X the "database"; a matrix of shape (n,d)
//' @param M the degree of the mutual reachability distance
//'          (should be rather small, say, \eqn{\leq 20}).
//'          \eqn{M\leq 2} denotes the ordinary Euclidean distance
//' @param algorithm \code{"auto"}, \code{"kd_tree_single"}
//'          \code{"kd_tree_dual"} or \code{"brute"};
//'     K-d trees can only be used for d between 2 and 20 only;
//'     \code{"auto"} selects \code{"kd_tree_dual"} for \eqn{d\leq 3},
//'     \eqn{M\leq 2}, and in a single-threaded setting only.
//'     \code{"kd_tree_single"} is used otherwise, unless \eqn{d>20}.
//' @param max_leaf_size maximal number of points in the K-d tree leaves;
//'        smaller leaves use more memory, yet are not necessarily faster;
//'        use \code{0} to select the default value, currently set to 32 for the
//'        single-tree and 8 for the dual-tree Boruvka algorithm
//' @param first_pass_max_brute_size minimal number of points in a node to
//'        treat it as a leaf (unless it's actually a leaf) in the first
//'        iteration of the algorithm; use \code{0} to select the default value,
//'        currently set to 32
//' @param dcore_dist_adj mutual reachability distance adjustment,
//'        a constant close to 0; in the case of ambiguity caused by equal
//'        core distances, a negative value will prefer connecting to farther
//'        points wrt the original distance, and closer ones in the case of
//'        a positive value
//' @param verbose whether to print diagnostic messages
//'
//'
//' @return
//' A list with two (M=1) or four (M>1) elements, \code{mst.index} and
//' \code{mst.dist}, and additionally \code{nn.index} and \code{nn.dist}.
//'
//' \code{mst.index} is a matrix with \eqn{n-1} rows and \code{2} columns,
//' whose rows define the tree edges.
//'
//' \code{mst.dist} is a vector of length
//' \code{n-1} giving the weights of the corresponding edges.
//'
//' The tree edges are ordered w.r.t. weights nondecreasingly, and then by
//' the indexes (lexicographic ordering of the \code{(weight, index1, index2)}
//' triples).  For each \code{i}, it holds \code{mst_ind[i,1]<mst_ind[i,2]}.
//'
//' \code{nn.index} is an \code{n} by \code{M-1} matrix giving the indexes
//' of each point's nearest neighbours. \code{nn.dist} provides the
//' corresponding distances.
//'
//'
//' @examples
//' library("datasets")
//' data("iris")
//' X <- jitter(as.matrix(iris[1:2]))  # some data
//' T <- mst_euclid(X)                 # Euclidean MST of X
//' plot(X, asp=1, las=1)
//' segments(X[T$mst.index[, 1], 1], X[T$mst.index[, 1], 2],
//'          X[T$mst.index[, 2], 1], X[T$mst.index[, 2], 2])
//'
//' @seealso \code{\link{knn_euclid}}
//'
//' @rdname fastmst
//' @export
// [[Rcpp::export("mst_euclid")]]
List mst_euclid(
    SEXP X,
    int M=1,
    Rcpp::String algorithm="auto",
    int max_leaf_size=0,
    int first_pass_max_brute_size=0,
    double dcore_dist_adj=-0.00000001490116119384765625,
    bool verbose=false
) {
    using FLOAT = double;  // float is not faster..

    Rcpp::NumericMatrix _X;
    if (!Rf_isMatrix(X)) _X = Rcpp::internal::convert_using_rfunction(X, "as.matrix");
    else _X = X;

    Py_ssize_t n = (Py_ssize_t)_X.nrow();
    Py_ssize_t d = (Py_ssize_t)_X.ncol();
    bool use_kdtree;
    bool use_dtb;

    if (n < 1 || d <= 1)  stop("X is ill-shaped");
    if (M < 1 || M > n-1) stop("incorrect M");

    if (algorithm == "auto") {
        if (2 <= d && d <= 20) {
            if (Comp_get_max_threads() == 1 && d <= 3 && M <= 2)
                algorithm = "kd_tree_dual";
            else
                algorithm = "kd_tree_single";
        }
        else
            algorithm = "brute";
    }

    if (algorithm == "kd_tree_dual" || algorithm == "kd_tree_single") {
        if (d < 2 || d > 20) stop("K-d trees can only be used for 2 <= d <= 20");
        use_kdtree = true;

        if (algorithm == "kd_tree_single") {
            if (max_leaf_size == 0)
                max_leaf_size = 32;  // the current default
            if (first_pass_max_brute_size == 0)
                first_pass_max_brute_size = 32;  // the current default
            use_dtb = false;
        }
        else {
            if (max_leaf_size == 0)
                max_leaf_size = 8;  // the current default
            if (first_pass_max_brute_size == 0)
                first_pass_max_brute_size = 32;  // the current default
            use_dtb = true;
        }

        if (max_leaf_size <= 0)
            stop("max_leaf_size must be positive");
        if (first_pass_max_brute_size <= 0)
            stop("first_pass_max_brute_size must be positive");
    }
    else if (algorithm == "brute")
        use_kdtree = false;
    else
        stop("invalid 'algorithm'");


    std::vector<FLOAT> XC(n*d);
    Py_ssize_t j = 0;
    for (Py_ssize_t i=0; i<n; ++i)
        for (Py_ssize_t u=0; u<d; ++u)
            XC[j++] = (FLOAT)_X(i, u);  // row-major


    std::vector<Py_ssize_t> mst_ind((n-1)*2);    // C-order
    std::vector<FLOAT>      mst_dist(n-1);       // TODO: use out_dist
    std::vector<Py_ssize_t> nn_ind((M==1)?0:(n*(M-1)));
    std::vector<FLOAT>      nn_dist((M==1)?0:(n*(M-1)));

    if (use_kdtree)
        Cmst_euclid_kdtree(
            XC.data(), n, d, M, mst_dist.data(), mst_ind.data(),
            (M==1)?nullptr:nn_dist.data(), (M==1)?nullptr:nn_ind.data(),
            max_leaf_size, first_pass_max_brute_size, use_dtb,
            dcore_dist_adj, verbose
        );
    else
        Cmst_euclid_brute(
            XC.data(), n, d, M, mst_dist.data(), mst_ind.data(),
            (M==1)?nullptr:nn_dist.data(), (M==1)?nullptr:nn_ind.data(),
            dcore_dist_adj, verbose
        );

    Rcpp::IntegerMatrix out_mst_ind(n-1, 2);
    Rcpp::NumericVector out_mst_dist(n-1);
    for (Py_ssize_t i=0; i<n-1; ++i) {
        out_mst_ind(i, 0)  = mst_ind[2*i+0]+1.0;  // R-based indexing
        out_mst_ind(i, 1)  = mst_ind[2*i+1]+1.0;  // R-based indexing
        out_mst_dist(i)    = mst_dist[i];
    }

    if (M == 1) {
        return Rcpp::List::create(
            Rcpp::Named("mst.index")=out_mst_ind,
            Rcpp::Named("mst.dist") =out_mst_dist
        );
    }
    else {
        Rcpp::IntegerMatrix out_nn_ind(n, M-1);
        Rcpp::NumericMatrix out_nn_dist(n, M-1);
        Py_ssize_t u=0;
        for (Py_ssize_t i=0; i<n; ++i) {
            for (Py_ssize_t j=0; j<M-1; ++j) {
                out_nn_ind(i, j)  = nn_ind[u]+1;  // 1-based indexing
                out_nn_dist(i, j) = nn_dist[u];
                ++u;
            }
        }
        return Rcpp::List::create(
            Rcpp::Named("mst.index")=out_mst_ind,
            Rcpp::Named("mst.dist") =out_mst_dist,
            Rcpp::Named("nn.index") =out_nn_ind,
            Rcpp::Named("nn.dist")  =out_nn_dist
        );
    }
}
