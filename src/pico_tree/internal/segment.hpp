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

#include <cassert>
#include <cmath>

#include "pico_tree/distance.hpp"

namespace pico_tree::internal {

template <typename Scalar_>
struct segment_base {
  using scalar_type = Scalar_;

  segment_base(scalar_type min, scalar_type max) : min(min), max(max) {}

  scalar_type min;
  scalar_type max;
};

template <typename Scalar_>
struct segment_r1 : protected segment_base<Scalar_> {
  using segment_base<Scalar_>::segment_base;
  using segment_base<Scalar_>::max;
  using segment_base<Scalar_>::min;
  using typename segment_base<Scalar_>::scalar_type;

  constexpr segment_r1(scalar_type min, scalar_type max)
      : segment_base<Scalar_>(min, max) {
    assert(min <= max);
  }

  constexpr bool contains(scalar_type x) const { return min <= x && x <= max; }

  constexpr bool contains(segment_r1<scalar_type> const& x) const {
    return min <= x.min && x.max <= max;
  }

  constexpr scalar_type distance(scalar_type x) const {
    if (x < min) {
      return min - x;
    } else if (x > max) {
      return x - max;
    } else {
      return scalar_type(0.0);
    }
  }

  constexpr scalar_type extent() const { return max - min; }
};

template <typename Scalar_>
struct segment_s1 : protected segment_base<Scalar_> {
  using segment_base<Scalar_>::segment_base;
  using segment_base<Scalar_>::max;
  using segment_base<Scalar_>::min;
  using typename segment_base<Scalar_>::scalar_type;

  constexpr segment_s1(scalar_type min, scalar_type max)
      : segment_base<Scalar_>(min, max) {}

  constexpr bool contains(scalar_type x) const {
    if (linear()) {
      return min <= x && x <= max;
    } else {
      return x >= min || x <= max;
    }
  }

  constexpr bool contains(segment_r1<scalar_type> const& x) const {
    if (linear()) {
      return min <= x.min && x.max <= max;
    } else {
      return x.min >= min || x.max <= max;
    }
  }

  constexpr scalar_type distance_min_max(scalar_type x) const {
    if (x < min || x > max) {
      return std::min(s1_distance(x, min), s1_distance(x, max));
    } else {
      return scalar_type(0.0);
    }
  }

  constexpr scalar_type distance_max_min(scalar_type x) const {
    if (x < max || x > min) {
      return scalar_type(0.0);
    } else {
      return std::min(s1_distance(x, min), s1_distance(x, max));
    }
  }

  constexpr scalar_type distance(scalar_type x) const {
    if (linear()) {
      return distance_min_max(x);
    } else {
      return distance_max_min(x);
    }
  }

  constexpr bool linear() const { return min <= max; }
};

}  // namespace pico_tree::internal
