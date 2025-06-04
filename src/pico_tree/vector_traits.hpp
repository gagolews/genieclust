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

#include <vector>

#include "core.hpp"
#include "point_traits.hpp"
#include "space_traits.hpp"

namespace pico_tree {

//! \brief Provides an interface for std::vector<> and points supported by
//! PointTraits.
//! \tparam Point_ Any of the point types supported by PointTraits.
//! \tparam Allocator_ Allocator type used by the std::vector.
template <typename Point_, typename Allocator_>
struct space_traits<std::vector<Point_, Allocator_>> {
  //! \brief The space_type of these traits.
  using space_type = std::vector<Point_, Allocator_>;
  //! \brief The point type used by space_type.
  using point_type = Point_;
  //! \brief The scalar type of point coordinates.
  using scalar_type = typename point_traits<Point_>::scalar_type;
  //! \brief The size and index type of point coordinates.
  using size_type = size_t;
  //! \brief Compile time spatial dimension.
  static size_type constexpr dim = point_traits<Point_>::dim;

  static_assert(
      dim != dynamic_extent, "VECTOR_OF_POINT_DOES_NOT_SUPPORT_DYNAMIC_DIM");

  //! \brief Returns the point at \p idx from \p space.
  template <typename Index_>
  inline static Point_ const& point_at(space_type const& space, Index_ idx) {
    return space[static_cast<size_type>(idx)];
  }

  //! \brief Returns number of points contained by \p space.
  inline static size_type size(space_type const& space) { return space.size(); }

  //! \brief Returns the number of coordinates or spatial dimension of each
  //! point.
  inline static size_type constexpr sdim(space_type const&) { return dim; }
};

}  // namespace pico_tree
