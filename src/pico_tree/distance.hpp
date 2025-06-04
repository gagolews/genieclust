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

namespace pico_tree {

class one_space_r1 {};

class one_space_s1 {};

namespace internal {

//! \brief Simply the number one as a constant.
template <typename T_>
inline T_ constexpr one_v = T_(1.0);

}  // namespace internal

//! \brief Calculates the distance between two coordinates.
template <typename Scalar_>
constexpr Scalar_ r1_distance(Scalar_ x, Scalar_ y) {
  return std::abs(x - y);
}

//! \brief Calculates the distance between two coordinates on the unit circle
//! s1. The values for \p x or \p y must lie within the range of [0...1].
template <typename Scalar_>
constexpr Scalar_ s1_distance(Scalar_ x, Scalar_ y) {
  Scalar_ const d = std::abs(x - y);
  return std::min(d, internal::one_v<Scalar_> - d);
}

//! \brief Calculates the square of a number.
template <typename Scalar_>
constexpr Scalar_ squared(Scalar_ x) {
  return x * x;
}

//! \brief Calculates the squared distance between two coordinates.
template <typename Scalar_>
constexpr Scalar_ squared_r1_distance(Scalar_ x, Scalar_ y) {
  return squared(x - y);
}

//! \brief Calculates the squared distance between two coordinates on the unit
//! circle s1. The values for \p x or \p y must lie within the range of [0...1].
template <typename Scalar_>
constexpr Scalar_ squared_s1_distance(Scalar_ x, Scalar_ y) {
  return squared(s1_distance(x, y));
}

}  // namespace pico_tree
