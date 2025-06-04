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

#include "pico_tree/core.hpp"
#include "pico_tree/point_traits.hpp"

namespace pico_tree::internal {

//! \brief The point_wrapper class wraps makes working with any point type
//! through its respective point_traits a bit easier and it allows for the
//! addition of extra convenience methods.
//! \details The internals of PicoTree never use the specializations of the
//! point_traits class directly, but interface with any point type through this
//! wrapper interface.
template <typename Point_>
class point_wrapper {
  using point_traits_type = point_traits<Point_>;
  using point_type = Point_;

 public:
  using scalar_type = typename point_traits_type::scalar_type;
  using size_type = size_t;
  static constexpr size_type dim = point_traits_type::dim;

  inline explicit point_wrapper(point_type const& point) : point_(point) {}

  inline scalar_type const& operator[](std::size_t index) const {
    return data()[index];
  }

  inline auto begin() const { return data(); }

  inline auto end() const { return data() + size(); }

 private:
  inline scalar_type const* data() const {
    return point_traits_type::data(point_);
  }

  constexpr size_type size() const {
    if constexpr (dim != dynamic_extent) {
      return dim;
    } else {
      return point_traits_type::size(point_);
    }
  }

  point_type const& point_;
};

}  // namespace pico_tree::internal
