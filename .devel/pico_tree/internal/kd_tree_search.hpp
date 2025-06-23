#pragma once

#include <vector>

#include "pico_tree/internal/box.hpp"
#include "pico_tree/internal/kd_tree_node.hpp"
#include "pico_tree/internal/point.hpp"
#include "pico_tree/metric.hpp"

namespace pico_tree::internal {

//! \brief This class provides a search nearest function for Euclidean spaces.
//! \details S. Arya and D. M. Mount, Algorithms for fast vector quantization,
//! In IEEE Data Compression Conference, pp. 381–390, March 1993.
//! https://www.cs.umd.edu/~mount/Papers/DCC.pdf
//! This paper describes the "Incremental Distance Calculation" technique  to
//! speed up nearest neighbor queries.
template <
    typename SpaceWrapper_,
    typename Metric_,
    typename PointWrapper_,
    typename Visitor_,
    typename Index_>
class search_nearest_euclidean {
 public:
  using index_type = Index_;
  using scalar_type = typename SpaceWrapper_::scalar_type;
  using point_type = point<scalar_type, SpaceWrapper_::dim>;
  //! \brief Node type supported by search_nearest_euclidean.
  using node_type = kd_tree_node_euclidean<index_type, scalar_type>;

  inline search_nearest_euclidean(
      SpaceWrapper_ space,
      Metric_ metric,
      std::vector<index_type> const& indices,
      PointWrapper_ query,
      Visitor_& visitor)
      : space_(space),
        metric_(metric),
        indices_(indices),
        query_(query),
        node_box_offset_(point_type::from_size(space_.sdim())),
        visitor_(visitor) {}

  //! \brief Search nearest neighbors starting from \p node.
  inline void operator()(node_type const* const node) {
    node_box_offset_.fill(scalar_type(0.0));
    search_nearest(node, scalar_type(0.0));
  }

 private:
  inline void search_nearest(
      node_type const* const node, scalar_type node_box_distance) {
    if (node->is_leaf()) {
      auto begin = indices_.begin() + node->data.leaf.begin_idx;
      auto const end = indices_.begin() + node->data.leaf.end_idx;
      for (; begin < end; ++begin) {
        visitor_(*begin, metric_(query_.begin(), query_.end(), space_[*begin]));
      }
    } else {
      // Go left or right and then check if we should still go down the other
      // side based on the current minimum distance.
      size_t const split_dim = static_cast<size_t>(node->data.branch.split_dim);
      scalar_type const v = query_[split_dim];
      scalar_type new_offset;
      node_type const* node_1st;
      node_type const* node_2nd;

      // On equals we would possibly need to go left as well. However, this is
      // handled by the if statement below this one: the check that max search
      // radius still hits the split value after having traversed the first
      // branch.
      // If left_max - v > 0, this means that the query is inside the left node,
      // if right_min - v < 0 it's inside the right one. For the area in between
      // we just pick the closest one by summing them.
      if ((node->data.branch.left_max + node->data.branch.right_min - v - v) >
          0) {
        node_1st = node->left;
        node_2nd = node->right;
        new_offset = metric_(node->data.branch.right_min - v);
      } else {
        node_1st = node->right;
        node_2nd = node->left;
        new_offset = metric_(node->data.branch.left_max - v);
      }

      // The distance and offset for node_1st is the same as that of its parent.
      search_nearest(node_1st, node_box_distance);

      // Calculate the distance to node_2nd.
      // NOTE: This method only works with Lp norms to which the exponent is not
      // applied.
      scalar_type const old_offset = node_box_offset_[split_dim];
      node_box_distance = node_box_distance - old_offset + new_offset;

      // The value visitor->max() contains the current nearest neighbor distance
      // or otherwise current maximum search distance. When testing against the
      // split value we determine if we should go into the neighboring node.
      if (visitor_.max() >= node_box_distance) {
        node_box_offset_[split_dim] = new_offset;
        search_nearest(node_2nd, node_box_distance);
        node_box_offset_[split_dim] = old_offset;
      }
    }
  }

  SpaceWrapper_ space_;
  Metric_ metric_;
  std::vector<index_type> const& indices_;
  PointWrapper_ query_;
  point_type node_box_offset_;
  Visitor_& visitor_;
};

//! \brief This class provides a search nearest function for topological spaces.
template <
    typename SpaceWrapper_,
    typename Metric_,
    typename PointWrapper_,
    typename Visitor_,
    typename Index_>
class search_nearest_topological {
 public:
  using index_type = Index_;
  using scalar_type = typename SpaceWrapper_::scalar_type;
  using point_type = point<scalar_type, SpaceWrapper_::dim>;
  //! \brief Node type supported by search_nearest_topological.
  using node_type = kd_tree_node_topological<index_type, scalar_type>;

  inline search_nearest_topological(
      SpaceWrapper_ space,
      Metric_ metric,
      std::vector<index_type> const& indices,
      PointWrapper_ query,
      Visitor_& visitor)
      : space_(space),
        metric_(metric),
        indices_(indices),
        query_(query),
        node_box_offset_(point_type::from_size(space_.sdim())),
        visitor_(visitor) {}

  //! \brief Search nearest neighbors starting from \p node.
  inline void operator()(node_type const* const node) {
    node_box_offset_.fill(scalar_type(0.0));
    search_nearest(node, scalar_type(0.0));
  }

 private:
  inline void search_nearest(
      node_type const* const node, scalar_type node_box_distance) {
    if (node->is_leaf()) {
      auto begin = indices_.begin() + node->data.leaf.begin_idx;
      auto const end = indices_.begin() + node->data.leaf.end_idx;
      for (; begin < end; ++begin) {
        visitor_(*begin, metric_(query_.begin(), query_.end(), space_[*begin]));
      }
    } else {
      // Go left or right and then check if we should still go down the other
      // side based on the current minimum distance.
      size_t const split_dim = static_cast<size_t>(node->data.branch.split_dim);
      scalar_type const v = query_[split_dim];
      // Determine the distance to the boxes of the children of this node.
      scalar_type const d1 = box_distance(
          node->data.branch.left_min,
          node->data.branch.left_max,
          v,
          node->data.branch.split_dim);
      scalar_type const d2 = box_distance(
          node->data.branch.right_min,
          node->data.branch.right_max,
          v,
          node->data.branch.split_dim);
      node_type const* node_1st;
      node_type const* node_2nd;
      scalar_type new_offset;

      // Visit the closest child/box first.
      if (d1 < d2) {
        node_1st = node->left;
        node_2nd = node->right;
        new_offset = d2;
      } else {
        node_1st = node->right;
        node_2nd = node->left;
        new_offset = d1;
      }

      search_nearest(node_1st, node_box_distance);

      scalar_type const old_offset = node_box_offset_[split_dim];
      node_box_distance = node_box_distance - old_offset + new_offset;

      // The value visitor->max() contains the current nearest neighbor distance
      // or otherwise current maximum search distance. When testing against the
      // split value we determine if we should go into the neighboring node.
      if (visitor_.max() >= node_box_distance) {
        node_box_offset_[split_dim] = new_offset;
        search_nearest(node_2nd, node_box_distance);
        node_box_offset_[split_dim] = old_offset;
      }
    }
  }

  scalar_type box_distance(
      scalar_type min, scalar_type max, scalar_type v, int dim) const {
    scalar_type d;
    auto bd = [&](auto one_space) { d = box_distance(min, max, v, one_space); };
    metric_.apply_dim_space(dim, bd);
    return metric_(d);
  }

  scalar_type box_distance(
      scalar_type min, scalar_type max, scalar_type v, one_space_r1) const {
    return segment_r1<scalar_type>(min, max).distance(v);
  }

  scalar_type box_distance(
      scalar_type min, scalar_type max, scalar_type v, one_space_s1) const {
    return segment_s1<scalar_type>(min, max).distance(v);
  }

  SpaceWrapper_ space_;
  Metric_ metric_;
  std::vector<index_type> const& indices_;
  PointWrapper_ query_;
  point_type node_box_offset_;
  Visitor_& visitor_;
};

//! \brief A functor that provides range searches for both Euclidean and
//! topological spaces. Query time is bounded by O(n^(1-1/dimension)+k).
//! \details Many tree nodes are excluded by checking if they intersect with the
//! box of the query. We don't store the bounding box of each node but calculate
//! them at run time. This slows down search_box in favor of having faster
//! nearest neighbor searches.
template <typename SpaceWrapper_, typename Metric_, typename Index_>
class search_box {
  using space_category = typename Metric_::space_category;
  static constexpr bool is_euclidean_space_v =
      std::is_same_v<space_category, euclidean_space_tag>;

 public:
  using index_type = Index_;
  using scalar_type = typename SpaceWrapper_::scalar_type;
  static size_t constexpr dim = SpaceWrapper_::dim;
  using box_type = box<scalar_type, dim>;
  using box_map_type = std::conditional_t<
      is_euclidean_space_v,
      box_map<scalar_type const, dim>,
      metric_box_map<scalar_type const, dim, Metric_>>;
  using node_type = typename kd_tree_space_tag_traits<
      space_category>::template node_type<index_type, scalar_type>;

  inline search_box(
      SpaceWrapper_ space,
      Metric_ metric,
      std::vector<index_type> const& indices,
      box_type const& root_box,
      box_map_type const& query,
      std::vector<index_type>& idxs)
      : space_(space),
        metric_(metric),
        indices_(indices),
        box_(root_box),
        query_(query),
        idxs_(idxs) {}

  //! \brief Range search starting from \p node.
  inline void operator()(node_type const* const node) {
    if (node->is_leaf()) {
      auto begin = indices_.begin() + node->data.leaf.begin_idx;
      auto const end = indices_.begin() + node->data.leaf.end_idx;
      for (; begin < end; ++begin) {
        if (contains(*begin)) {
          idxs_.push_back(*begin);
        }
      }
    } else {
      size_t const split_dim = static_cast<size_t>(node->data.branch.split_dim);
      scalar_type old_value = box_.max(split_dim);
      box_.max(split_dim) = node->data.branch.left_max;

      // Check if the left node is fully contained. If true, report all its
      // indices. Else, if its partially contained, continue the range search
      // down the left node.
      if (query_.contains(box_)) {
        report_node(node->left);
      } else if (intersects_left(split_dim, node)) {
        operator()(node->left);
      }

      box_.max(split_dim) = old_value;
      old_value = box_.min(split_dim);
      box_.min(split_dim) = node->data.branch.right_min;

      // Same as the left side.
      if (query_.contains(box_)) {
        report_node(node->right);
      } else if (intersects_right(split_dim, node)) {
        operator()(node->right);
      }

      box_.min(split_dim) = old_value;
    }
  }

 private:
  bool contains(index_type const idx) const {
    return query_.contains(space_[idx]);
  }

  bool contains() const { return query_.contains(box_); }

  bool intersects_left(
      size_t const split_dim, node_type const* const node) const {
    if constexpr (is_euclidean_space_v) {
      return query_.min(split_dim) <= node->data.branch.left_max;
    } else {
      return query_.min(split_dim) <= node->data.branch.left_max ||
             query_.max(split_dim) >= node->data.branch.left_min;
    }
  }

  bool intersects_right(
      size_t const split_dim, node_type const* const node) const {
    if constexpr (is_euclidean_space_v) {
      return query_.max(split_dim) >= node->data.branch.right_min;
    } else {
      return query_.max(split_dim) >= node->data.branch.right_min ||
             query_.min(split_dim) <= node->data.branch.right_max;
    }
  }

  //! \brief Reports all indices contained by \p node.
  inline void report_node(node_type const* const node) const {
    index_type begin;
    index_type end;

    if (node->is_leaf()) {
      begin = node->data.leaf.begin_idx;
      end = node->data.leaf.end_idx;
    } else {
      // Nodes and index pointers (begin_idx and end_idx) are ordered left to
      // right. This means that for any node, its left-most and right-most leaf
      // node descendants will respectively store the begin index and end index
      // of the entire range of points contained by that node.
      begin = report_left(node->left);
      end = report_right(node->right);
    }

    std::copy(
        indices_.cbegin() + begin,
        indices_.cbegin() + end,
        std::back_inserter(idxs_));
  }

  inline index_type report_left(node_type const* const node) const {
    if (node->is_leaf()) {
      return node->data.leaf.begin_idx;
    } else {
      return report_left(node->left);
    }
  }

  inline index_type report_right(node_type const* const node) const {
    if (node->is_leaf()) {
      return node->data.leaf.end_idx;
    } else {
      return report_right(node->right);
    }
  }

  SpaceWrapper_ space_;
  Metric_ metric_;
  std::vector<index_type> const& indices_;
  // This variable is used for maintaining a running bounding box.
  box_type box_;
  box_map_type const& query_;
  std::vector<index_type>& idxs_;
};

}  // namespace pico_tree::internal
