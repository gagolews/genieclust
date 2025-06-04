/*
 * PicoTree: a C++ header only library for fast nearest neighbor
 * and range searches using a KdTree.
 *
 * <https://github.com/Jaybro/pico_tree>
 *
 * Version 1.0.0 (c5f719837df9707ee12d94cb0108aa0c34bfe96f)
 *
 * Copyright (c) 2025 Jonathan Broere
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */




#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

#include "pico_tree/internal/box.hpp"
#include "pico_tree/internal/kd_tree_data.hpp"
#include "pico_tree/internal/kd_tree_node.hpp"
#include "pico_tree/internal/point_wrapper.hpp"
#include "pico_tree/metric.hpp"

namespace pico_tree {

template <typename Derived_>
struct splitter_rule_t {
  Derived_ const& derived() const {
    return *static_cast<Derived_ const*>(this);
  }

 protected:
  constexpr explicit splitter_rule_t() = default;
  constexpr explicit splitter_rule_t(splitter_rule_t const&) = default;
  constexpr explicit splitter_rule_t(splitter_rule_t&&) = default;
};

//! \brief Splits a node on the median point along the dimension of the node's
//! box longest side. This rule is also known as the standard split rule.
//! \details This rule builds a tree in O(n log n) time on average. It's
//! generally slower compared to sliding_midpoint_max_side_t but results in a
//! balanced kd_tree.
struct median_max_side_t : public splitter_rule_t<median_max_side_t> {
  constexpr explicit median_max_side_t() = default;
};

//! \brief Splits a node's box halfway the dimension of its longest side. The
//! first dimension is chosen if multiple sides share being the longest. This
//! splitting rule can result in empty nodes.
//! \details The aspect ratio of the split is at most 2:1.
struct midpoint_max_side_t : public splitter_rule_t<midpoint_max_side_t> {
  constexpr explicit midpoint_max_side_t() = default;
};

//! \brief Splits a node's box halfway the dimension of its longest side. The
//! first dimension is chosen if multiple sides share being the longest. In case
//! the split results in an empty sub-node, the split is adjusted to include a
//! single point into that sub-node.
//! \details Based on the paper "It's okay to be skinny, if your friends are
//! fat". The aspect ratio of the split is at most 2:1 unless that results in an
//! empty sub-node.
//!
//! * http://www.cs.umd.edu/~mount/Papers/cgc99-smpack.pdf
//!
//! This splitter can be used to answer an approximate nearest neighbor query in
//! O(1/e^d log n) time.
//!
//! The tree is build in O(n log n) time and results in a tree that is both
//! faster to build and generally faster to query as compared to
//! median_max_side_t.
struct sliding_midpoint_max_side_t
    : public splitter_rule_t<sliding_midpoint_max_side_t> {
  constexpr explicit sliding_midpoint_max_side_t() = default;
};

//! \copydoc median_max_side_t
inline constexpr median_max_side_t median_max_side{};

//! \copydoc midpoint_max_side_t
inline constexpr midpoint_max_side_t midpoint_max_side{};

//! \copydoc sliding_midpoint_max_side_t
inline constexpr sliding_midpoint_max_side_t sliding_midpoint_max_side{};

template <typename Derived_>
struct splitter_stop_condition_t {
  Derived_ const& derived() const {
    return *static_cast<Derived_ const*>(this);
  }

 protected:
  constexpr explicit splitter_stop_condition_t() = default;
  constexpr explicit splitter_stop_condition_t(
      splitter_stop_condition_t const&) = default;
  constexpr explicit splitter_stop_condition_t(splitter_stop_condition_t&&) =
      default;
};

//! \brief The maximum number of points allowed in a leaf node.
struct max_leaf_size_t : public splitter_stop_condition_t<max_leaf_size_t> {
  constexpr max_leaf_size_t(size_t v) : value(v) { assert(value > 0); }

  size_t value;
};

//! \brief The maximum depth allowed for a leaf node. A depth of zero means that
//! the root node becomes a leaf node.
struct max_leaf_depth_t : public splitter_stop_condition_t<max_leaf_depth_t> {
  constexpr max_leaf_depth_t(size_t v) : value(v) {}

  size_t value;
};

template <typename Derived_>
struct splitter_start_bounds_t {
  Derived_ const& derived() const {
    return *static_cast<Derived_ const*>(this);
  }

 protected:
  constexpr explicit splitter_start_bounds_t() = default;
  constexpr explicit splitter_start_bounds_t(splitter_start_bounds_t const&) =
      default;
  constexpr explicit splitter_start_bounds_t(splitter_start_bounds_t&&) =
      default;
};

struct bounds_from_space_t
    : public splitter_start_bounds_t<bounds_from_space_t> {
  constexpr explicit bounds_from_space_t() = default;
};

inline constexpr bounds_from_space_t bounds_from_space{};

template <typename Point_>
struct bounds_t : public splitter_start_bounds_t<bounds_t<Point_>> {
  constexpr explicit bounds_t(Point_ const& min, Point_ const& max)
      : min_(min), max_(max) {}

  constexpr Point_ const& min() const { return min_; }
  constexpr Point_ const& max() const { return max_; }

 private:
  Point_ min_;
  Point_ max_;
};

namespace internal {

//! \copydoc median_max_side_t
template <typename SpaceWrapper_>
class splitter_median_max_side {
  using scalar_type = typename SpaceWrapper_::scalar_type;
  using size_type = size_t;
  using box_type = box<scalar_type, SpaceWrapper_::dim>;

 public:
  splitter_median_max_side(SpaceWrapper_ space) : space_{space} {}

  template <typename RandomAccessIterator_>
  inline void operator()(
      typename std::iterator_traits<
          RandomAccessIterator_>::value_type const,  // depth
      RandomAccessIterator_ begin,
      RandomAccessIterator_ end,
      box_type const& box,
      RandomAccessIterator_& split,
      size_type& split_dim,
      scalar_type& split_val) const {
    scalar_type max_delta;
    box.max_side(split_dim, max_delta);

    split = begin + (end - begin) / 2;

    std::nth_element(
        begin,
        split,
        end,
        [this, &split_dim](auto const index_a, auto const index_b) -> bool {
          return space_[index_a][split_dim] < space_[index_b][split_dim];
        });

    split_val = space_[*split][split_dim];
  }

 private:
  SpaceWrapper_ space_;
};

//! \copydoc midpoint_max_side_t
template <typename SpaceWrapper_>
class splitter_midpoint_max_side {
  using scalar_type = typename SpaceWrapper_::scalar_type;
  using size_type = size_t;
  using box_type = box<scalar_type, SpaceWrapper_::dim>;

 public:
  splitter_midpoint_max_side(SpaceWrapper_ space) : space_{space} {}

  template <typename RandomAccessIterator_>
  inline void operator()(
      typename std::iterator_traits<
          RandomAccessIterator_>::value_type const,  // depth
      RandomAccessIterator_ begin,
      RandomAccessIterator_ end,
      box_type const& box,
      RandomAccessIterator_& split,
      size_type& split_dim,
      scalar_type& split_val) const {
    scalar_type max_delta;
    box.max_side(split_dim, max_delta);
    split_val = max_delta * scalar_type(0.5) + box.min(split_dim);

    // Everything smaller than split_val goes left, the rest right.
    auto const comp = [this, &split_dim, &split_val](auto const index) -> bool {
      return space_[index][split_dim] < split_val;
    };

    split = std::partition(begin, end, comp);
  }

 private:
  SpaceWrapper_ space_;
};

//! \copydoc sliding_midpoint_max_side_t
template <typename SpaceWrapper_>
class splitter_sliding_midpoint_max_side {
  using scalar_type = typename SpaceWrapper_::scalar_type;
  using size_type = size_t;
  using box_type = box<scalar_type, SpaceWrapper_::dim>;

 public:
  splitter_sliding_midpoint_max_side(SpaceWrapper_ space) : space_{space} {}

  template <typename RandomAccessIterator_>
  inline void operator()(
      typename std::iterator_traits<
          RandomAccessIterator_>::value_type const,  // depth
      RandomAccessIterator_ begin,
      RandomAccessIterator_ end,
      box_type const& box,
      RandomAccessIterator_& split,
      size_type& split_dim,
      scalar_type& split_val) const {
    scalar_type max_delta;
    box.max_side(split_dim, max_delta);
    split_val = max_delta / scalar_type(2.0) + box.min(split_dim);

    // Everything smaller than split_val goes left, the rest right.
    auto const comp = [this, &split_dim, &split_val](auto const index) -> bool {
      return space_[index][split_dim] < split_val;
    };

    split = std::partition(begin, end, comp);

    // If it happens that either all points are on the left side or right
    // side, one point slides to the other side and we split on the first
    // right value instead of the middle split. In these two cases the split
    // value is unknown and a partial sort is required to obtain it, but also
    // to rearrange all other indices such that they are on their
    // corresponding left or right side.
    if (split == end) {
      split--;
      std::nth_element(
          begin,
          split,
          end,
          [this, &split_dim](auto const index_a, auto const index_b) -> bool {
            return space_[index_a][split_dim] < space_[index_b][split_dim];
          });
      split_val = space_[*split][split_dim];
    } else if (split == begin) {
      split++;
      std::nth_element(
          begin,
          split,
          end,
          [this, &split_dim](auto const index_a, auto const index_b) -> bool {
            return space_[index_a][split_dim] < space_[index_b][split_dim];
          });
      split_val = space_[*split][split_dim];
    }
  }

 private:
  SpaceWrapper_ space_;
};

template <typename Rule_>
struct splitter_rule_traits;

template <>
struct splitter_rule_traits<median_max_side_t> {
  template <typename SpaceWrapper_>
  using splitter_type = splitter_median_max_side<SpaceWrapper_>;
};

template <>
struct splitter_rule_traits<midpoint_max_side_t> {
  template <typename SpaceWrapper_>
  using splitter_type = splitter_midpoint_max_side<SpaceWrapper_>;
};

template <>
struct splitter_rule_traits<sliding_midpoint_max_side_t> {
  template <typename SpaceWrapper_>
  using splitter_type = splitter_sliding_midpoint_max_side<SpaceWrapper_>;
};

//! \brief This class provides the build algorithm of the kd_tree. How the
//! kd_tree will be build depends on the Splitter template argument.
template <
    typename SpaceWrapper_,
    typename Stop_,
    typename Rule_,
    typename KdTreeData_>
class build_kd_tree_impl {
 public:
  using index_type = typename KdTreeData_::index_type;
  using scalar_type = typename KdTreeData_::scalar_type;
  using size_type = size_t;
  using space_type = SpaceWrapper_;
  using box_type = box<scalar_type, KdTreeData_::dim>;
  using splitter_type = typename splitter_rule_traits<
      Rule_>::template splitter_type<SpaceWrapper_>;
  using kd_tree_data_type = KdTreeData_;
  using node_type = typename kd_tree_data_type::node_type;
  using node_allocator_type = typename kd_tree_data_type::node_allocator_type;

  build_kd_tree_impl(
      space_type const& space,
      size_type const stop_value,
      std::vector<index_type>& indices,
      node_allocator_type& allocator)
      : space_(space),
        stop_value_(static_cast<index_type>(stop_value)),
        splitter_(space_),
        indices_(indices),
        allocator_(allocator) {}

  //! \brief Creates a tree for a range of indices and returns its root node.
  //! Each time a node is created, its corresponding index range is split in
  //! two. This happens recursively for each sub set of indices until the stop
  //! condition is reached.
  //! \details While descending the tree we split nodes based on the root box
  //! until leaf nodes are reached. Inside the leaf nodes the boxes are updated
  //! to be the bounding boxes of the points they contain. While unwinding the
  //! recursion we update the split information for each branch node based on
  //! merging leaf nodes. Since the updated split information based on the leaf
  //! nodes can have smaller bounding boxes than the original ones, we can
  //! improve query times.
  inline node_type* operator()(box_type const& root_box) {
    box_type box(root_box);
    return create_node(0, indices_.begin(), indices_.end(), box);
  }

 private:
  template <typename RandomAccessIterator_>
  inline node_type* create_node(
      index_type const depth,
      RandomAccessIterator_ begin,
      RandomAccessIterator_ end,
      box_type& box) const {
    node_type* node = allocator_.allocate();

    if (is_leaf(depth, begin, end)) {
      node->set_leaf(
          static_cast<index_type>(begin - indices_.begin()),
          static_cast<index_type>(end - indices_.begin()));
      // Keep the original box in case it is empty. This can only happen with
      // the midpoint split.
      if constexpr (std::is_same_v<Rule_, midpoint_max_side_t>) {
        if (begin < end) {
          compute_bounding_box(begin, end, box);
        }
      } else {
        compute_bounding_box(begin, end, box);
      }
    } else {
      // split equals end for the left branch and begin for the right branch.
      RandomAccessIterator_ split;
      size_type split_dim;
      scalar_type split_val;
      splitter_(depth, begin, end, box, split, split_dim, split_val);

      box_type right = box;
      // Argument box will function as the left bounding box until we merge
      // left and right again at the end of this code section.
      box.max(split_dim) = split_val;
      right.min(split_dim) = split_val;

      node->left = create_node(depth + 1, begin, split, box);
      node->right = create_node(depth + 1, split, end, right);

      node->set_branch(box, right, split_dim);

      // Merges both child boxes. We can expect any of the min max values to
      // change except for the ones of split_dim.
      box.fit(right);
    }

    return node;
  }

  template <typename RandomAccessIterator_>
  inline void compute_bounding_box(
      RandomAccessIterator_ begin,
      RandomAccessIterator_ end,
      box_type& box) const {
    box.fill_inverse_max();
    for (; begin < end; ++begin) {
      box.fit(space_[*begin]);
    }
  }

  template <typename RandomAccessIterator_>
  inline bool is_leaf(
      index_type depth,
      RandomAccessIterator_ begin,
      RandomAccessIterator_ end) const {
    if constexpr (std::is_same_v<Stop_, max_leaf_size_t>) {
      return (end - begin) <= stop_value_;
    } else {
      // Either stop when the depth is reached or when the amount of points that
      // remain <= 1. In the latter case it becomes impossible to split a branch
      // further. A forced split for a leaf size is 1 would just add another
      // empty node, but it would also be a problem for the
      // sliding_midpoint_max_side_t splitter rule, where none of the resulting
      // branches is allowed to be empty.
      return (depth == stop_value_) || ((end - begin) <= 1);
    }
  }

  space_type const& space_;
  index_type const stop_value_;
  splitter_type splitter_;
  std::vector<index_type>& indices_;
  node_allocator_type& allocator_;
};

//! \brief kd_tree meta information depending on the SpaceTag_ template
//! argument.
template <typename SpaceTag_>
struct kd_tree_space_tag_traits;

//! \brief kd_tree meta information for the euclidean_space_tag.
template <>
struct kd_tree_space_tag_traits<euclidean_space_tag> {
  //! \brief Supported node type.
  template <typename Index_, typename Scalar_>
  using node_type = kd_tree_node_euclidean<Index_, Scalar_>;
};

//! \brief kd_tree meta information for the topological_space_tag.
template <>
struct kd_tree_space_tag_traits<topological_space_tag> {
  //! \brief Supported node type.
  template <typename Index_, typename Scalar_>
  using node_type = kd_tree_node_topological<Index_, Scalar_>;
};

template <typename KdTreeData_, size_t Dim_>
class build_kd_tree {
  using index_type = typename KdTreeData_::index_type;
  using scalar_type = typename KdTreeData_::scalar_type;
  using node_type = typename KdTreeData_::node_type;
  using box_type = box<scalar_type, Dim_>;

 public:
  using kd_tree_data_type = KdTreeData_;

  //! \brief Construct a kd_tree.
  template <
      typename SpaceWrapper_,
      typename Stop_,
      typename Bounds_,
      typename Rule_>
  kd_tree_data_type operator()(
      SpaceWrapper_ space,
      splitter_stop_condition_t<Stop_> const& stop_condition,
      splitter_start_bounds_t<Bounds_> const& start_bounds,
      splitter_rule_t<Rule_> const&) {
    static_assert(
        std::is_same_v<scalar_type, typename SpaceWrapper_::scalar_type>);
    static_assert(Dim_ == SpaceWrapper_::dim);
    assert(space.size() > 0);

    using build_kd_tree_impl_type =
        build_kd_tree_impl<SpaceWrapper_, Stop_, Rule_, kd_tree_data_type>;
    using node_allocator_type = typename kd_tree_data_type::node_allocator_type;

    std::vector<index_type> indices(space.size());
    std::iota(indices.begin(), indices.end(), 0);
    box_type root_box = box_from_bounds(space, start_bounds.derived());
    node_allocator_type allocator;
    node_type* root_node = build_kd_tree_impl_type{
        space, stop_condition.derived().value, indices, allocator}(root_box);

    return kd_tree_data_type{
        std::move(indices), root_box, std::move(allocator), root_node};
  }

 private:
  template <typename SpaceWrapper_>
  box_type box_from_bounds(SpaceWrapper_ space, bounds_from_space_t) const {
    return space.compute_bounding_box();
  }

  template <typename SpaceWrapper_, typename Point_>
  box_type box_from_bounds(
      SpaceWrapper_ space, bounds_t<Point_> const& bounds) const {
    internal::point_wrapper<Point_> min(bounds.min());
    internal::point_wrapper<Point_> max(bounds.max());
    box_type bbox = box_type::make_inverse_max(space.sdim());
    bbox.fit(min.begin());
    bbox.fit(max.begin());
    return bbox;
  }
};

}  // namespace internal

}  // namespace pico_tree
