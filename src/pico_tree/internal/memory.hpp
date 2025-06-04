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

#include <type_traits>

namespace pico_tree::internal {

//! \brief An instance of list_pool_resource constructs fixed size chunks of
//! memory and stores these in a list. Memory is only released when the resource
//! is destructed or when calling the release() method.
//! \details A list_pool_resource is mainly useful for monotonically
//! constructing objects of a single type when the total number to be created
//! cannot be known up front.
//! <p/>
//! A previous memory manager implementation was based on the std::deque. The
//! chunk size that it uses can vary across different implementations of the C++
//! standard, resulting in an unreliable performance of PicoTree.
//! <p/>
//! https://en.wikipedia.org/wiki/Memory_pool
template <typename T_, std::size_t ChunkSize_>
class list_pool_resource {
 private:
  struct node;

 public:
  static_assert(std::is_trivial_v<T_>, "TYPE_T_IS_NOT_TRIVIAL");
  static_assert(
      std::is_trivially_destructible_v<T_>,
      "TYPE_T_IS_NOT_TRIVIALLY_DESTRUCTIBLE");

  //! \brief Value type allocated by the list_pool_resource.
  using value_type = T_;
  //! \brief Chunk type allocated by the list_pool_resource.
  using chunk = typename node::chunk;

 public:
  //! \brief list_pool_resource constructor.
  list_pool_resource() : head_(nullptr) {}

  //! \brief A list_pool_resource instance cannot be copied.
  //! \details Just no!
  list_pool_resource(list_pool_resource const&) = delete;

  //! \private
  list_pool_resource(list_pool_resource&& other) : head_(other.head_) {
    // So we don't accidentally delete things twice.
    other.head_ = nullptr;
  }

  //! \private
  list_pool_resource& operator=(list_pool_resource const& other) = delete;

  //! \private
  list_pool_resource& operator=(list_pool_resource&& other) {
    head_ = other.head_;
    other.head_ = nullptr;
    return *this;
  }

  //! \brief list_pool_resource destructor.
  virtual ~list_pool_resource() { release(); }

  //! \brief Allocates a chunk of memory and returns a pointer to it.
  inline chunk* allocate() {
    node* n = new node;
    n->prev = head_;
    head_ = n;
    return &head_->data;
  }

  //! \brief Release all memory allocated by this list_pool_resource.
  void release() {
    // Suppose node was contained by an std::unique_ptr, then it may happen
    // that we hit a recursion limit depending on how many nodes are destructed.
    while (head_ != nullptr) {
      node* n = head_->prev;
      delete head_;
      head_ = n;
    }
  }

 private:
  node* head_;
};

//! \brief Node containing a chunk of memory.
template <typename T_, std::size_t ChunkSize_>
struct list_pool_resource<T_, ChunkSize_>::node {
  //! \brief Chunk type allocated by the list_pool_resource.
  using chunk = std::array<T_, ChunkSize_>;

  node* prev;
  chunk data;
};

//! \brief An instance of chunk_allocator constructs objects. It does so in
//! chunks of size ChunkSize_ to reduce memory fragmentation.
template <typename T_, std::size_t ChunkSize_>
class chunk_allocator final {
 private:
  using resource = list_pool_resource<T_, ChunkSize_>;
  using chunk = typename resource::chunk;

 public:
  //! \brief Value type allocated by the chunk_allocator.
  using value_type = T_;

  //! \brief chunk_allocator constructor.
  chunk_allocator() : object_index_(ChunkSize_) {}

  //! \brief Create an object of type T_ and return a pointer to it.
  inline value_type* allocate() {
    if (object_index_ == ChunkSize_) {
      chunk_ = resource_.allocate();
      object_index_ = 0;
    }

    value_type* object = &(*chunk_)[object_index_];
    object_index_++;

    return object;
  }

 private:
  resource resource_;
  std::size_t object_index_;
  chunk* chunk_;
};

}  // namespace pico_tree::internal
