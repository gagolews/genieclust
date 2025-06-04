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

#include <cmath>
#include <iterator>
#include <limits>

#include "core.hpp"
#include "distance.hpp"

namespace pico_tree {

namespace internal {

//! \brief Calculates the distance between two coordinates.
struct distance_fn {
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x, Scalar_ y) const {
    return r1_distance(x, y);
  }
};

//! \brief Calculates the squared distance between two coordinates.
struct squared_r1_distance_fn {
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x, Scalar_ y) const {
    return squared_r1_distance(x, y);
  }
};

//! \private
template <
    typename InputIterator1_,
    typename InputSentinel1_,
    typename InputIterator2_,
    typename BinaryOperator>
constexpr auto sum(
    InputIterator1_ begin1,
    InputSentinel1_ end1,
    InputIterator2_ begin2,
    BinaryOperator op) {
  using scalar_type =
      typename std::iterator_traits<InputIterator1_>::value_type;

  scalar_type d{};

  for (; begin1 != end1; ++begin1, ++begin2) {
    d += op(*begin1, *begin2);
  }

  return d;
}

}  // namespace internal

//! \brief This tag indicates that a metric to supports the most generic space
//! that can be used with PicoTree's search structures.
//! \details A space tag is used by PicoTree to select the correct algorithms
//! for use with a particular space.
//!
//! Usings the topological_space_tag for metrics allows support for
//! identifications in point sets. A practical example is that of the unit
//! circle represented by the interval [0, 1]. Here, 0 and 1 are the same point
//! on the circle and performing a radius query around both values should result
//! in the same point set.
class topological_space_tag {};

//! \brief This tag indicates that a metric supports the Euclidean space with
//! PicoTree's search structures.
//! \details Supports the fastest queries but doesn't support identifications.
//! \see topological_space_tag
class euclidean_space_tag : public topological_space_tag {};

//! \brief metric_l1 metric for measuring the Taxicab or Manhattan distance
//! between points.
//! \details For more details:
//! * https://en.wikipedia.org/wiki/Metric_space
//! * https://en.wikipedia.org/wiki/Lp_space
struct metric_l1 {
  //! \brief This tag specifies the supported space by this metric.
  using space_category = euclidean_space_tag;

  template <
      typename InputIterator1_,
      typename InputSentinel1_,
      typename InputIterator2_>
  constexpr auto operator()(
      InputIterator1_ begin1,
      InputSentinel1_ end1,
      InputIterator2_ begin2) const {
    return internal::sum(begin1, end1, begin2, internal::distance_fn());
  }

  //! \brief Returns the absolute value of \p x.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x) const {
    return std::abs(x);
  }
};

//! \brief The metric_l2_squared semi-metric measures squared Euclidean
//! distances between points. It does not satisfy the triangle inequality.
//! \see metric_l1
struct metric_l2_squared {
  //! \brief This tag specifies the supported space by this metric.
  using space_category = euclidean_space_tag;

  template <
      typename InputIterator1_,
      typename InputSentinel1_,
      typename InputIterator2_>
  constexpr auto operator()(
      InputIterator1_ begin1,
      InputSentinel1_ end1,
      InputIterator2_ begin2) const {
    return internal::sum(
        begin1, end1, begin2, internal::squared_r1_distance_fn());
  }

  //! \brief Returns the squared value of \p x.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x) const {
    return squared(x);
  }
};

struct metric_lpinf {
  //! \brief This tag specifies the supported space by this metric.
  using space_category = euclidean_space_tag;

  template <
      typename InputIterator1_,
      typename InputSentinel1_,
      typename InputIterator2_>
  constexpr auto operator()(
      InputIterator1_ begin1,
      InputSentinel1_ end1,
      InputIterator2_ begin2) const {
    using scalar_type =
        typename std::iterator_traits<InputIterator1_>::value_type;

    scalar_type d{};

    for (; begin1 != end1; ++begin1, ++begin2) {
      d = std::max(d, r1_distance(*begin1, *begin2));
    }

    return d;
  }

  //! \brief Returns the absolute value of \p x.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x) const {
    return std::abs(x);
  }
};

struct metric_lninf {
  //! \brief This tag specifies the supported space by this metric.
  using space_category = euclidean_space_tag;

  template <
      typename InputIterator1_,
      typename InputSentinel1_,
      typename InputIterator2_>
  constexpr auto operator()(
      InputIterator1_ begin1,
      InputSentinel1_ end1,
      InputIterator2_ begin2) const {
    using scalar_type =
        typename std::iterator_traits<InputIterator1_>::value_type;

    scalar_type d = std::numeric_limits<scalar_type>::max();

    for (; begin1 != end1; ++begin1, ++begin2) {
      d = std::min(d, r1_distance(*begin1, *begin2));
    }

    return d;
  }

  //! \brief Returns the absolute value of \p x.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x) const {
    return std::abs(x);
  }
};

//! \brief The metric_so2 measures distances on the unit circle S1. The
//! coordinate of each point is expected to be within the range [0...1].
//! \details  It is the intrinsic metric of points in R2 on S1 given by the
//! great-circle distance. Named after the Special Orthogonal Group of
//! dimension 2. The circle S1 is represented by the range [0...1] / 0 ~ 1.
//!
//! For more details:
//! * https://en.wikipedia.org/wiki/Intrinsic_metric
//! * https://en.wikipedia.org/wiki/Great-circle_distance
struct metric_so2 {
  //! \brief This tag specifies the supported space by this metric.
  using space_category = topological_space_tag;

  template <
      typename InputIterator1_,
      typename InputSentinel1_,
      typename InputIterator2_>
  constexpr auto operator()(
      InputIterator1_ begin1, InputSentinel1_, InputIterator2_ begin2) const {
    return s1_distance(*begin1, *begin2);
  }

  //! \brief Returns the absolute value of \p x.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x) const {
    return std::abs(x);
  }

  template <typename UnaryPredicate_>
  void apply_dim_space([[maybe_unused]] int dim, UnaryPredicate_ p) const {
    p(one_space_s1{});
  }
};

//! \brief The metric_se2_squared measures distances in Euclidean space between
//! Euclidean motions. The third coordinate of each point is expected to be
//! within the range [0...1].
//! \details Named after the Special Euclidean group of dimension 2.
//! For more details:
//! * https://en.wikipedia.org/wiki/Euclidean_group
struct metric_se2_squared {
  //! \brief This tag specifies the supported space by this metric.
  using space_category = topological_space_tag;

  template <
      typename InputIterator1_,
      typename InputSentinel1_,
      typename InputIterator2_>
  constexpr auto operator()(
      InputIterator1_ begin1, InputSentinel1_, InputIterator2_ begin2) const {
    return internal::sum(
               begin1, begin1 + 2, begin2, internal::squared_r1_distance_fn()) +
           squared_s1_distance(*(begin1 + 2), *(begin2 + 2));
  }

  //! \brief Returns the squared value of \p x.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x) const {
    return squared(x);
  }

  template <typename UnaryPredicate_>
  void apply_dim_space(int dim, UnaryPredicate_ p) const {
    if (dim < 2) {
      p(one_space_r1{});
    } else {
      p(one_space_s1{});
    }
  }
};

}  // namespace pico_tree
