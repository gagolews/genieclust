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

#include <iterator>
#include <vector>

#include "pico_tree/core.hpp"
#include "pico_tree/internal/memory.hpp"
#include "pico_tree/internal/stream_wrapper.hpp"

namespace pico_tree::internal {

//! \brief The data structure that represents a kd_tree.
template <typename Node_, size_t Dim_>
class kd_tree_data {
  template <typename Iterator_>
  class iterator_range {
   public:
    using iterator_type = Iterator_;
    using difference_type =
        typename std::iterator_traits<Iterator_>::difference_type;

    constexpr iterator_range(Iterator_ begin, Iterator_ end)
        : begin_(begin), end_(end) {}

    constexpr Iterator_ begin() const { return begin_; }
    constexpr Iterator_ end() const { return end_; }

   private:
    Iterator_ begin_;
    Iterator_ end_;
  };

 public:
  using index_type = typename Node_::index_type;
  using scalar_type = typename Node_::scalar_type;
  static size_t constexpr dim = Dim_;
  using box_type = internal::box<scalar_type, dim>;
  using node_type = Node_;
  using node_allocator_type = chunk_allocator<node_type, 256>;
  using leaf_range_type =
      iterator_range<typename std::vector<index_type>::const_iterator>;

  static kd_tree_data load(stream_wrapper& stream) {
    typename box_type::size_type sdim;
    stream.read(sdim);

    kd_tree_data kd_tree_data{
        {}, box_type(sdim), node_allocator_type(), nullptr};
    kd_tree_data.read(stream);

    return kd_tree_data;
  }

  static void save(kd_tree_data const& data, stream_wrapper& stream) {
    // Write sdim.
    stream.write(data.root_box.size());
    data.write(stream);
  }

  inline std::vector<leaf_range_type> leaf_ranges() const {
    std::vector<leaf_range_type> ranges;
    insert_leaf_range(root_node, ranges);
    return ranges;
  }

  //! \brief Sorted indices that refer to points inside points_.
  std::vector<index_type> indices;
  //! \brief Bounding box of the root node.
  box_type root_box;
  //! \brief Memory allocator for tree nodes.
  node_allocator_type allocator;
  //! \brief Root of the kd_tree.
  node_type* root_node;

 private:
  inline void insert_leaf_range(
      node_type const* const node, std::vector<leaf_range_type>& ranges) const {
    if (node->is_leaf() && !node->data.leaf.empty()) {
      using difference_type = typename leaf_range_type::difference_type;
      ranges.push_back(leaf_range_type(
          indices.begin() + difference_type(node->data.leaf.begin_idx),
          indices.begin() + difference_type(node->data.leaf.end_idx)));
    } else {
      insert_leaf_range(node->left, ranges);
      insert_leaf_range(node->right, ranges);
    }
  }

  //! \brief Recursively reads the Node and its descendants.
  inline node_type* read_node(stream_wrapper& stream) {
    node_type* node = allocator.allocate();
    bool is_leaf;
    stream.read(is_leaf);

    if (is_leaf) {
      stream.read(node->data.leaf);
      node->left = nullptr;
      node->right = nullptr;
    } else {
      stream.read(node->data.branch);
      node->left = read_node(stream);
      node->right = read_node(stream);
    }

    return node;
  }

  //! \brief Recursively writes the Node and its descendants.
  inline void write_node(
      node_type const* const node, stream_wrapper& stream) const {
    if (node->is_leaf()) {
      stream.write(true);
      stream.write(node->data.leaf);
    } else {
      stream.write(false);
      stream.write(node->data.branch);
      write_node(node->left, stream);
      write_node(node->right, stream);
    }
  }

  inline void read(stream_wrapper& stream) {
    stream.read(indices);
    // The root box gets the correct size from the kd_tree constructor.
    stream.read(root_box.size(), root_box.min());
    stream.read(root_box.size(), root_box.max());
    root_node = read_node(stream);
  }

  inline void write(stream_wrapper& stream) const {
    stream.write(indices);
    stream.write(root_box.min(), root_box.size());
    stream.write(root_box.max(), root_box.size());
    write_node(root_node, stream);
  }
};

}  // namespace pico_tree::internal
