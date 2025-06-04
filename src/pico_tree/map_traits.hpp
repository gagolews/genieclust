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

//! \file map_traits.hpp
//! \brief Provides an interface for spaces and points when working with raw
//! pointers.

#include "map.hpp"
#include "space_traits.hpp"

namespace pico_tree {

template <typename Scalar_, size_t Dim_>
struct point_traits<point_map<Scalar_, Dim_>> {
  using point_type = point_map<Scalar_, Dim_>;
  using scalar_type = typename point_type::scalar_type;
  using size_type = typename point_type::size_type;
  static size_type constexpr dim = Dim_;

  inline static scalar_type const* data(point_type const& point) {
    return point.data();
  }

  inline static size_type size(point_type const& point) { return point.size(); }
};

//! \brief MapTraits provides an interface for spaces and points when working
//! with a space_map.
template <typename Point_>
struct space_traits<space_map<Point_>> {
  using space_type = space_map<Point_>;
  using point_type = typename space_type::point_type;
  using scalar_type = typename space_type::scalar_type;
  using size_type = typename space_type::size_type;
  static size_type constexpr dim = space_type::dim;

  template <typename Index_>
  inline static decltype(auto) point_at(space_type const& space, Index_ idx) {
    return space[static_cast<size_type>(idx)];
  }

  inline static size_type size(space_type const& space) { return space.size(); }

  inline static size_type sdim(space_type const& space) { return space.sdim(); }
};

}  // namespace pico_tree
