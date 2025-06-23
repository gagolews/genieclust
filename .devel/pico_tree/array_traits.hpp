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
