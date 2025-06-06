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

namespace pico_tree::internal {

//!\brief Binary node base.
template <typename Derived_>
struct kd_tree_node_base {
  //! \brief Returns if the current node is a branch.
  inline bool is_branch() const { return left != nullptr && right != nullptr; }
  //! \brief Returns if the current node is a leaf.
  inline bool is_leaf() const { return left == nullptr && right == nullptr; }

  template <typename Index_>
  inline void set_leaf(Index_ begin_idx, Index_ end_idx) {
    derived().data.leaf.begin_idx = begin_idx;
    derived().data.leaf.end_idx = end_idx;
    left = nullptr;
    right = nullptr;
  }

  inline Derived_& derived() { return *static_cast<Derived_*>(this); }

  //! \brief Left child.
  Derived_* left;
  //! \brief Right child.
  Derived_* right;
};

//! \brief Tree leaf data.
template <typename Index_>
struct kd_tree_leaf {
  //! \brief Returns true if the leaf is empty.
  constexpr bool empty() const { return begin_idx == end_idx; }

  //! \brief Begin of an index range.
  Index_ begin_idx;
  //! \brief End of an index range.
  Index_ end_idx;
};

//! \brief Tree branch data that stores one boundary per child.
template <typename Scalar_>
struct kd_tree_branch_single {
  //! \brief Split coordinate / index of the kd_tree spatial dimension.
  int split_dim;
  //! \brief Maximum coordinate value of the left node box for split_dim.
  Scalar_ left_max;
  //! \brief Minimum coordinate value of the right node box for split_dim.
  Scalar_ right_min;
};

//! \brief Tree branch data that stores two boundaries per child.
//! \details Storing both boundaries per child allows the use of identifications
//! (wrapping around).
template <typename Scalar_>
struct kd_tree_branch_double {
  //! \brief Split coordinate / index of the kd_tree spatial dimension.
  int split_dim;
  //! \brief Minimum coordinate value of the left node box for split_dim.
  Scalar_ left_min;
  //! \brief Maximum coordinate value of the left node box for split_dim.
  Scalar_ left_max;
  //! \brief Minimum coordinate value of the right node box for split_dim.
  Scalar_ right_min;
  //! \brief Maximum coordinate value of the right node box for split_dim.
  Scalar_ right_max;
};

//! \brief NodeData is used to either store branch or leaf information. Which
//! union member is used can be tested with is_branch() or is_leaf().
template <typename Leaf_, typename Branch_>
union kd_tree_node_data {
  //! \brief Union branch data.
  Branch_ branch;
  //! \brief Union leaf data.
  Leaf_ leaf;
};

//! \brief kd_tree node for a Euclidean space.
template <typename Index_, typename Scalar_>
struct kd_tree_node_euclidean
    : public kd_tree_node_base<kd_tree_node_euclidean<Index_, Scalar_>> {
  using index_type = Index_;
  using scalar_type = Scalar_;

  template <typename Box_>
  inline void set_branch(
      Box_ const& left_box, Box_ const& right_box, size_t const split_dim) {
    data.branch.split_dim = static_cast<int>(split_dim);
    data.branch.left_max = left_box.max(split_dim);
    data.branch.right_min = right_box.min(split_dim);
  }

  //! \brief Node data as a union of a leaf and branch.
  kd_tree_node_data<kd_tree_leaf<Index_>, kd_tree_branch_single<Scalar_>> data;
};

//! \brief kd_tree node for a topological space.
template <typename Index_, typename Scalar_>
struct kd_tree_node_topological
    : public kd_tree_node_base<kd_tree_node_topological<Index_, Scalar_>> {
  using index_type = Index_;
  using scalar_type = Scalar_;

  template <typename Box_>
  inline void set_branch(
      Box_ const& left_box, Box_ const& right_box, size_t const split_dim) {
    data.branch.split_dim = static_cast<int>(split_dim);
    data.branch.left_min = left_box.min(split_dim);
    data.branch.left_max = left_box.max(split_dim);
    data.branch.right_min = right_box.min(split_dim);
    data.branch.right_max = right_box.max(split_dim);
  }

  //! \brief Node data as a union of a leaf and branch.
  kd_tree_node_data<kd_tree_leaf<Index_>, kd_tree_branch_double<Scalar_>> data;
};

}  // namespace pico_tree::internal
