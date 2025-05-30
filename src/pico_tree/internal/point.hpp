#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <vector>

#include "pico_tree/core.hpp"

namespace pico_tree::internal {

//! \details The non-specialized class knows its dimension at compile-time and
//! uses an std::array for storing its data. Faster than using the std::vector
//! in practice.
template <typename Scalar_, size_t Dim_>
struct point_storage_traits {
  using type = std::array<Scalar_, Dim_>;

  static constexpr auto from_size([[maybe_unused]] size_t size) {
    assert(size == Dim_);
    return type();
  }
};

//! \details The specialized class doesn't knows its dimension at compile-time
//! and uses an std::vector for storing its data so it can be resized.
template <typename Scalar_>
struct point_storage_traits<Scalar_, dynamic_extent> {
  using type = std::vector<Scalar_>;

  static constexpr auto from_size(size_t size) { return type(size); }
};

//! \brief A point is a container that stores a contiguous array of elements as
//! an aggregate type. The storage is either as an std::array or an std::vector.
//! Using the storage, elems_, is considered undefined behavior.
//! \details Having elems_ public goes against the against the encapsulation
//! principle but gives us aggregate initialization in return.
template <typename Scalar_, size_t Dim_>
struct point {
  static_assert(
      Dim_ == dynamic_extent || Dim_ > 0, "DIM_MUST_BE_DYNAMIC_OR_>_0");

  //! \private Using Pst__ is considered undefined behavior.
  using Pst__ = point_storage_traits<Scalar_, Dim_>;
  //! \private Using Elems__ is considered undefined behavior.
  using Elems__ = typename Pst__::type;

  using scalar_type = Scalar_;
  using size_type = size_t;
  static size_type constexpr dim = Dim_;

  //! \brief Creates a point from a size.
  static constexpr point from_size(size_t size) {
    return {Pst__::from_size(size)};
  }

  //! \brief Fills the storage with value \p v.
  constexpr void fill(scalar_type v) {
    std::fill(elems_.begin(), elems_.end(), v);
  }

  //! \brief Normalize point in place to unit length.
  void normalize() {
    scalar_type l2 = scalar_type(0);
    for (auto& e : elems_) l2 += e * e;

    l2 = scalar_type(1) / std::sqrt(l2);
    for (auto& e : elems_) e *= l2;
  }

  //! \brief Access the container data.
  constexpr scalar_type& operator[](size_type i) noexcept { return elems_[i]; }

  //! \brief Access the container data.
  constexpr scalar_type const& operator[](size_type i) const noexcept {
    return elems_[i];
  }

  constexpr scalar_type const* data() const noexcept { return elems_.data(); }

  constexpr scalar_type* data() noexcept { return elems_.data(); }

  //! \brief Returns the size of the container.
  constexpr size_type size() const noexcept { return elems_.size(); }

  //! \private Using elems_ is considered undefined behavior.
  Elems__ elems_;
};

}  // namespace pico_tree::internal
