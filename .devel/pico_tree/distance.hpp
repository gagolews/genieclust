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
