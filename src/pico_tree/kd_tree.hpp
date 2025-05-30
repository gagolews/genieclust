#pragma once

#include "pico_tree/internal/box.hpp"
#include "pico_tree/internal/kd_tree_builder.hpp"
#include "pico_tree/internal/kd_tree_search.hpp"
#include "pico_tree/internal/point_wrapper.hpp"
#include "pico_tree/internal/search_visitor.hpp"
#include "pico_tree/internal/space_wrapper.hpp"
#include "pico_tree/internal/type_traits.hpp"
#include "pico_tree/metric.hpp"

namespace pico_tree {

//! \brief A kd_tree is a binary tree that partitions space using hyper planes.
//! \details https://en.wikipedia.org/wiki/K-d_tree
//! \tparam Space_ Type of space.
//! \tparam Metric_ Type of metric. Determines how distances are measured.
//! \tparam Index_ Type of index.
template <
    typename Space_,
    typename Metric_ = metric_l2_squared,
    typename Index_ = int>
class kd_tree {
  static_assert(
      std::is_same_v<std::remove_cv_t<Space_>, Space_>,
      "SPACE_TYPE_MUST_BE_NON-CONST_NON-VOLATILE");

  using space_wrapper_type =
      internal::space_wrapper<internal::remove_reference_wrapper_t<Space_>>;
  //! \brief Node type based on Metric_::space_category.
  using node_type = typename internal::kd_tree_space_tag_traits<
      typename Metric_::space_category>::
      template node_type<Index_, typename space_wrapper_type::scalar_type>;
  using kd_tree_data_type =
      internal::kd_tree_data<node_type, space_wrapper_type::dim>;

 public:
  //! \brief Size type.
  using size_type = size_t;
  //! \brief Index type.
  using index_type = Index_;
  //! \brief Scalar type.
  using scalar_type = typename space_wrapper_type::scalar_type;
  //! \brief kd_tree dimension. It equals pico_tree::dynamic_extent in case
  //! dim is only known at run-time.
  static size_type constexpr dim = space_wrapper_type::dim;
  //! \brief Point set or adaptor type.
  using space_type = Space_;
  //! \brief The metric used for various searches.
  using metric_type = Metric_;
  //! \brief Neighbor type of various search results.
  using neighbor_type = neighbor<index_type, scalar_type>;
  //! \brief Leaf index range type.
  using leaf_range_type = typename kd_tree_data_type::leaf_range_type;

  //! \brief Creates a kd_tree given \p space and \p stop_condition.
  //! \details The kd_tree takes \p space by value. This allows it to take
  //! ownership of the point cloud. To avoid creating a copy of the input:
  //!
  //! \li Use move semantics: kd_tree tree(std::move(space), stop_condition);
  //! \li Use an std::reference_wrapper<space_type> as the space_type.
  //!
  //! The value of \p stop_condition influences the height and performance of
  //! the tree. The splitting mechanism determines data locality within the
  //! leafs. The exact effect it has depends on the tree splitting mechanism.
  //!
  //! \param space The input point set.
  //! \param stop_condition One of max_leaf_size_t or max_leaf_depth_t.
  //! \param start_bounds One of bounds_from_space_t or bounds_t<Point_>.
  //! \param rule One of median_max_side_t, midpoint_max_side_t, or
  //! sliding_midpoint_max_side_t.
  template <
      typename Stop_,
      typename Bounds_ = bounds_from_space_t,
      typename Rule_ = sliding_midpoint_max_side_t>
  kd_tree(
      space_type space,
      splitter_stop_condition_t<Stop_> const& stop_condition,
      splitter_start_bounds_t<Bounds_> const& start_bounds = Bounds_{},
      splitter_rule_t<Rule_> const& rule = Rule_{})
      : space_(std::move(space)),
        metric_(),
        data_(
            internal::build_kd_tree<kd_tree_data_type, dim>()(
                space_wrapper_type(space_),
                stop_condition,
                start_bounds,
                rule)) {}

  //! \brief The kd_tree cannot be copied.
  //! \details The kd_tree uses pointers to nodes and copying pointers is not
  //! the same as creating a deep copy.
  kd_tree(kd_tree const&) = delete;

  //! \brief Move constructor of the kd_tree.
  kd_tree(kd_tree&&) = default;

  //! \brief kd_tree copy assignment.
  kd_tree& operator=(kd_tree const& other) = delete;

  //! \brief kd_tree move assignment.
  kd_tree& operator=(kd_tree&& other) = default;

  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor .
  template <typename P_, typename V_>
  inline void search_nearest(P_ const& x, V_& visitor) const {
    static_assert(
        std::is_same_v<
            scalar_type,
            typename internal::point_wrapper<P_>::scalar_type>,
        "POINT_AND_TREE_SCALAR_TYPES_DIFFER");
    static_assert(
        dim == internal::point_wrapper<P_>::dim || dim == dynamic_extent ||
            internal::point_wrapper<P_>::dim == dynamic_extent,
        "POINT_AND_TREE_DIMS_DIFFER");

    internal::point_wrapper<P_> p(x);
    search_nearest(p, visitor, typename Metric_::space_category());
  }

  //! \brief Searches for the nearest neighbor of point \p x.
  //! \details Interpretation of the output distance depends on the metric type.
  //! The default metric_l2_squared results in a squared distance.
  template <typename P_>
  inline void search_nn(P_ const& x, neighbor_type& nn) const {
    internal::search_nn<neighbor_type> v(nn);
    search_nearest(x, v);
  }

  //! \brief Searches for the approximate nearest neighbor of point \p x.
  //! \details Nodes in the tree are skipped by scaling down the search
  //! distance and as a result the true nearest neighbor may not be found. An
  //! approximate nearest neighbor will at most be a factor of distance ratio \p
  //! e farther from the query point than the true nearest neighbor:
  //! max_ann_distance = true_nn_distance * e.
  //!
  //! Interpretation of both the input error ratio and output distances
  //! depend on the metric type. The default metric_l2_squared calculates
  //! squared distances. Using this metric, the input error ratio should be the
  //! squared error ratio and the output distances will be squared distances
  //! scaled by the inverse error ratio.
  //!
  //! Example:
  //! \code{.cpp}
  //! // A max error of 15%. I.e. max 15% farther away from the true nn.
  //! scalar_type max_error = scalar_type(0.15);
  //! scalar_type e = tree.metric()(scalar_type(1.0) + max_error);
  //! neighbor<index_type, scalar_type> nn;
  //! tree.search_nn(p, e, nn);
  //! // Optionally scale back to the actual metric distance.
  //! nn.second *= e;
  //! \endcode
  template <typename P_>
  inline void search_nn(
      P_ const& x, scalar_type const e, neighbor_type& nn) const {
    internal::search_approximate_nn<neighbor_type> v(e, nn);
    search_nearest(x, v);
  }

  //! \brief Searches for the k nearest neighbors of point \p x, where k equals
  //! std::distance(begin, end). It is expected that the value type of the
  //! iterator equals neighbor<index_type, scalar_type>.
  //! \details Interpretation of the output distances depend on the metric type.
  //! The default metric_l2_squared results in squared distances.
  //! \tparam P_ Point type.
  //! \tparam RandomAccessIterator_ Iterator type.
  template <typename P_, typename RandomAccessIterator_>
  inline void search_knn(
      P_ const& x,
      RandomAccessIterator_ begin,
      RandomAccessIterator_ end) const {
    static_assert(
        std::is_same_v<
            typename std::iterator_traits<RandomAccessIterator_>::value_type,
            neighbor_type>,
        "ITERATOR_VALUE_TYPE_DOES_NOT_EQUAL_NEIGHBOR_TYPE");

    internal::search_knn<RandomAccessIterator_> v(begin, end);
    search_nearest(x, v);
  }

  //! \brief Searches for the \p k nearest neighbors of point \p x and stores
  //! the results in output vector \p knn.
  //! \tparam P_ Point type.
  //! \see template <typename P_, typename RandomAccessIterator_> void
  //! search_knn(P_ const&, RandomAccessIterator_, RandomAccessIterator_) const
  template <typename P_>
  inline void search_knn(
      P_ const& x, size_type const k, std::vector<neighbor_type>& knn) const {
    // If it happens that the point set has less points than k we just return
    // all points in the set.
    knn.resize(std::min(k, space_wrapper_type(space_).size()));
    search_knn(x, knn.begin(), knn.end());
  }

  //! \brief Searches for the k approximate nearest neighbors of point \p x,
  //! where k equals std::distance(begin, end). It is expected that the value
  //! type of the iterator equals neighbor<index_type, scalar_type>.
  //! \see template <typename P_, typename RandomAccessIterator_> void
  //! search_knn(P_ const&, RandomAccessIterator_, RandomAccessIterator_) const
  //! \see template <typename P_, typename RandomAccessIterator_> void
  //! search_nn(P_ const&, scalar_type, neighbor_type&) const
  template <typename P_, typename RandomAccessIterator_>
  inline void search_knn(
      P_ const& x,
      scalar_type const e,
      RandomAccessIterator_ begin,
      RandomAccessIterator_ end) const {
    static_assert(
        std::is_same_v<
            typename std::iterator_traits<RandomAccessIterator_>::value_type,
            neighbor_type>,
        "ITERATOR_VALUE_TYPE_DOES_NOT_EQUAL_NEIGHBOR_TYPE");

    internal::search_approximate_knn<RandomAccessIterator_> v(e, begin, end);
    search_nearest(x, v);
  }

  //! \brief Searches for the \p k approximate nearest neighbors of point \p x
  //! and stores the results in output vector \p knn.
  //! \tparam P_ Point type.
  //! \see template <typename P_, typename RandomAccessIterator_> void
  //! search_knn(P_ const&, RandomAccessIterator_, RandomAccessIterator_) const
  //! \see template <typename P_, typename RandomAccessIterator_> void
  //! search_nn(P_ const&, scalar_type, neighbor_type&) const
  template <typename P_>
  inline void search_knn(
      P_ const& x,
      size_type const k,
      scalar_type const e,
      std::vector<neighbor_type>& knn) const {
    // If it happens that the point set has less points than k we just return
    // all points in the set.
    knn.resize(std::min(k, space_wrapper_type(space_).size()));
    search_knn(x, e, knn.begin(), knn.end());
  }

  //! \brief Searches for all the neighbors of point \p x that are within radius
  //! \p radius and stores the results in output vector \p n.
  //! \details Interpretation of the in and output distances depend on the
  //! metric type. The default metric_l2_squared results in squared distances.
  //! \tparam P_ Point type.
  //! \param x Input point.
  //! \param radius Search radius.
  //! \code{.cpp}
  //! scalar_type distance = -2.0;
  //! // E.g., metric_l1: 2.0, metric_l2_squared: 4.0
  //! scalar_type metric_distance = kdtree.metric()(distance);
  //! std::vector<neighbor<index_type, scalar_type>> n;
  //! tree.search_radius(p, metric_distance, n);
  //! \endcode
  //! \param n Output points.
  //! \param sort If true, the result set is sorted from closest to farthest
  //! distance with respect to the query point.
  template <typename P_>
  inline void search_radius(
      P_ const& x,
      scalar_type const radius,
      std::vector<neighbor_type>& n,
      bool const sort = false) const {
    internal::search_radius<neighbor_type> v(radius, n);
    search_nearest(x, v);

    if (sort) {
      v.sort();
    }
  }

  //! \brief Searches for all approximate neighbors of point \p x that are
  //! within radius \p radius and stores the results in output vector \p n.
  //! \see template <typename P_, typename RandomAccessIterator_> void
  //! search_radius(P_ const&, scalar_type, std::vector<neighbor_type>&, bool)
  //! const
  //! \see template <typename P_, typename RandomAccessIterator_> void
  //! search_nn(P_ const&, scalar_type, neighbor_type&) const
  template <typename P_>
  inline void search_radius(
      P_ const& x,
      scalar_type const radius,
      scalar_type const e,
      std::vector<neighbor_type>& n,
      bool const sort = false) const {
    internal::search_approximate_radius<neighbor_type> v(e, radius, n);
    search_nearest(x, v);

    if (sort) {
      v.sort();
    }
  }

  //! \brief Returns all points within the box defined by \p min and \p max.
  //! Query time is bounded by O(n^(1-1/dimension)+k).
  //! \tparam P_ Point type.
  template <typename P_>
  inline void search_box(
      P_ const& min, P_ const& max, std::vector<index_type>& idxs) const {
    idxs.clear();
    space_wrapper_type space(space_);
    // Note that it's never checked if the bounding box intersects at all. For
    // now it is assumed that this check is not worth it: If there is any
    // overlap then the search is slower. So unless many queries don't intersect
    // there is no point in adding it.
    using search_box_type =
        internal::search_box<space_wrapper_type, Metric_, index_type>;
    using box_map_type = typename search_box_type::box_map_type;

    search_box_type(
        space,
        metric_,
        data_.indices,
        data_.root_box,
        box_map_type(
            internal::point_wrapper<P_>(min).begin(),
            internal::point_wrapper<P_>(max).begin(),
            space.sdim()),
        idxs)(data_.root_node);
  }

  //! \brief Returns the index range for all non-empty leaves in the kd_tree.
  //! \details The leaf order follows from a depth-first traversal of the
  //! kd_tree. If the input bounds is a hyper cube, then the leaf order is
  //! identical to that of the N-order curve (a rotated version of the Z-order
  //! curve that splits nodes in reversed order).
  inline std::vector<leaf_range_type> leaf_ranges() const {
    return data_.leaf_ranges();
  }

  //! \brief Point set used by the tree.
  inline space_type const& space() const { return space_; }

  //! \brief Metric used for search queries.
  inline metric_type const& metric() const { return metric_; }

  //! \brief Loads the tree in binary from file.
  static kd_tree load(space_type space, std::string const& filename) {
    std::fstream stream =
        internal::open_stream(filename, std::ios::in | std::ios::binary);
    return load(std::move(space), stream);
  }

  //! \brief Loads the tree in binary from \p stream .
  //! \details This is considered a convenience function to be able to save and
  //! load a kd_tree on a single machine.
  //! \li Does not take memory endianness into account.
  //! \li Does not check if the stored tree structure is valid for the given
  //! point set.
  //! \li Does not check if the stored tree structure is valid for the given
  //! template arguments.
  static kd_tree load(space_type space, std::iostream& stream) {
    internal::stream_wrapper s(stream);
    return kd_tree(std::move(space), s);
  }

  //! \brief Saves the tree in binary to file.
  static void save(kd_tree const& tree, std::string const& filename) {
    std::fstream stream =
        internal::open_stream(filename, std::ios::out | std::ios::binary);
    save(tree, stream);
  }

  //! \brief Saves the tree in binary to \p stream .
  //! \details This is considered a convenience function to be able to save and
  //! load a kd_tree on a single machine.
  //! \li Does not take memory endianness into account.
  //! \li Stores the tree structure but not the space.
  static void save(kd_tree const& tree, std::iostream& stream) {
    internal::stream_wrapper s(stream);
    kd_tree_data_type::save(tree.data_, s);
  }

 private:
  //! \brief Constructs a kd_tree by reading its indexing and leaf information
  //! from a stream.
  kd_tree(space_type space, internal::stream_wrapper& stream)
      : space_(std::move(space)),
        metric_(),
        data_(kd_tree_data_type::load(stream)) {}

  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor for node \p node.
  template <typename PointWrapper_, typename Visitor_>
  inline void search_nearest(
      PointWrapper_ point, Visitor_& visitor, euclidean_space_tag) const {
    internal::search_nearest_euclidean<
        space_wrapper_type,
        Metric_,
        PointWrapper_,
        Visitor_,
        index_type>(
        space_wrapper_type(space_), metric_, data_.indices, point, visitor)(
        data_.root_node);
  }

  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor for node \p node.
  template <typename PointWrapper_, typename Visitor_>
  inline void search_nearest(
      PointWrapper_ point, Visitor_& visitor, topological_space_tag) const {
    internal::search_nearest_topological<
        space_wrapper_type,
        Metric_,
        PointWrapper_,
        Visitor_,
        index_type>(
        space_wrapper_type(space_), metric_, data_.indices, point, visitor)(
        data_.root_node);
  }

  //! \brief Point set used for querying point data.
  space_type space_;
  //! \brief Metric used for comparing distances.
  metric_type metric_;
  //! \brief Data structure of the kd_tree.
  kd_tree_data_type data_;
};

template <typename Space_, typename... Args>
kd_tree(Space_, Args...) -> kd_tree<Space_, metric_l2_squared, int>;

template <
    typename Metric_ = metric_l2_squared,
    typename Index_ = int,
    typename Bounds_ = bounds_from_space_t,
    typename Rule_ = sliding_midpoint_max_side_t,
    typename Space_,
    typename Stop_>
kd_tree<std::decay_t<Space_>, Metric_, Index_> make_kd_tree(
    Space_&& space,
    splitter_stop_condition_t<Stop_> const& stop_condition,
    splitter_start_bounds_t<Bounds_> const& start_bounds = Bounds_{},
    splitter_rule_t<Rule_> const& rule = Rule_{}) {
  return kd_tree<std::decay_t<Space_>, Metric_, Index_>(
      std::forward<Space_>(space), stop_condition, start_bounds, rule);
}

}  // namespace pico_tree
