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

#include <array>

#include "core.hpp"
#include "point_traits.hpp"

namespace pico_tree {

//! \brief Point interface for Scalar_[Dim_].
template <typename Scalar_, std::size_t Dim_>
struct point_traits<Scalar_[Dim_]> {
  using point_type = Scalar_[Dim_];
  using scalar_type = Scalar_;
  using size_type = size_t;
  static constexpr size_type dim = static_cast<size_type>(Dim_);

  //! \brief Returns a pointer to the coordinates of the input point.
  inline static constexpr scalar_type const* data(point_type const& point) {
    return point;
  }

  //! \brief Returns the number of coordinates or spatial dimension of each
  //! point.
  inline static constexpr size_type size(point_type const&) { return dim; }
};

//! \brief Point interface for std::array<Scalar_, Dim_>.
template <typename Scalar_, std::size_t Dim_>
struct point_traits<std::array<Scalar_, Dim_>> {
  using point_type = std::array<Scalar_, Dim_>;
  using scalar_type = Scalar_;
  using size_type = size_t;
  static constexpr size_type dim = static_cast<size_type>(Dim_);

  //! \brief Returns a pointer to the coordinates of the input point.
  inline static constexpr scalar_type const* data(point_type const& point) {
    return point.data();
  }

  //! \brief Returns the number of coordinates or spatial dimension of each
  //! point.
  inline static constexpr size_type size(point_type const&) { return dim; }
};

}  // namespace pico_tree
